"""Aparta reseñas similares entre particiones, conservando las de mayor prioridad.

La comparación es Jaccard exacto de conjuntos de trigramas, con un índice de
prefijos para evitar comparar todos los pares. No entrena ni evalúa modelos.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import unicodedata
import uuid
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

try:
    from .prepare_dataset import COLUMNS, LABELS, PRIORITY, ROOT, digest
except ImportError:  # Ejecución directa desde PowerShell.
    from prepare_dataset import COLUMNS, LABELS, PRIORITY, ROOT, digest

INPUT = ROOT / 'data/processed/amazon_three_levels'
OUTPUT = ROOT / 'data/processed/amazon_three_levels_clean'
POLICY = ROOT / 'data/similarity_policy.json'
MATCH_COLUMNS = ('id', 'original_split', 'reference_id', 'reference_split',
                 'intersection', 'union', 'jaccard')


def grams(text: str) -> set[str]:
    normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    if not normalized:
        raise ValueError('No se admite texto vacío.')
    return {normalized[i:i + 3] for i in range(len(normalized) - 2)} if len(normalized) >= 3 else {normalized}


def prefix_length(size: int, threshold: Fraction) -> int:
    # ceil(threshold * size), sin redondeos de coma flotante.
    required = (threshold.numerator * size + threshold.denominator - 1) // threshold.denominator
    return size - required + 1


class SimilarityIndex:
    """Búsqueda exhaustiva por umbral, sin muestreo ni aproximación aleatoria.

    Si J(A,B)>=t, la intersección contiene al menos ceil(t*|A|) y
    ceil(t*|B|) elementos. Sus prefijos, en un mismo orden global, deben
    compartir algún elemento. Solo estos candidatos necesitan verificación.
    """

    def __init__(self, references: list[dict], threshold: Fraction):
        if not 0 < threshold <= 1:
            raise ValueError('El umbral debe estar entre 0 y 1.')
        self.threshold = threshold
        self.references = references
        frequencies = Counter()
        for row in references:
            frequencies.update(grams(row['text']))
        ordered = sorted(frequencies, key=lambda token: (frequencies[token], token))
        self.ranks = {token: rank for rank, token in enumerate(ordered)}
        self.tokens: list[tuple[int, ...]] = []
        self.postings: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(references):
            tokens = tuple(sorted(self.ranks[token] for token in grams(row['text'])))
            self.tokens.append(tokens)
            for token in tokens[:prefix_length(len(tokens), threshold)]:
                self.postings[token].append(index)

    def matches(self, text: str, allowed_ids: set[str]):
        tokens = grams(text)
        size = len(tokens)
        known = {self.ranks[token] for token in tokens if token in self.ranks}
        unknown_count = size - len(known)
        # Los tokens ausentes en todas las referencias tienen frecuencia cero y
        # preceden a los conocidos; no generan candidatos en el índice.
        prefix_size = prefix_length(size, self.threshold) - unknown_count
        if prefix_size <= 0:
            return []
        numerator, denominator = self.threshold.numerator, self.threshold.denominator
        minimum = (numerator * size + denominator - 1) // denominator
        maximum = denominator * size // numerator
        candidates = set()
        for token in sorted(known)[:prefix_size]:
            for index in self.postings.get(token, ()):
                if (self.references[index]['id'] in allowed_ids
                        and minimum <= len(self.tokens[index]) <= maximum):
                    candidates.add(index)
        found = []
        for index in sorted(candidates):
            overlap = len(known.intersection(self.tokens[index]))
            union = size + len(self.tokens[index]) - overlap
            if overlap * denominator >= numerator * union:
                found.append((index, overlap, union))
        return found


def read_rows(path: Path, split: str):
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f'Cabecera inesperada en {path.name}.')
        for row in reader:
            if (None in row or any(value is None for value in row.values())
                    or row['original_split'] != split or row['valuation'] not in LABELS
                    or not row['id'] or not row['text'].strip()):
                raise ValueError(f'Registro inválido en {path.name}.')
            yield row


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding='utf-8'))
    if (policy.get('priority') != list(PRIORITY) or policy.get('character_ngram_size') != 3
            or policy.get('normalization') != 'NFKC, casefold, collapsed whitespace; punctuation and negations preserved'
            or policy.get('action') != 'exclude_lower_priority_against_retained_higher_priority'
            or policy.get('compare_within_split') is not False
            or policy.get('use_labels_for_decision') is not False):
        raise ValueError('Política de similitud incompatible con el auditor.')
    threshold = Fraction(policy['threshold'])
    if not 0 < threshold <= 1:
        raise ValueError('Umbral inválido.')
    return policy


def audit(input_dir: Path, output_dir: Path, policy_path: Path = POLICY, progress=None) -> dict:
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if (output_dir == input_dir or output_dir.is_relative_to(input_dir)
            or input_dir.is_relative_to(output_dir)):
        raise ValueError('Las rutas de entrada y salida deben estar separadas.')
    if output_dir.exists():
        raise FileExistsError(f'{output_dir} ya existe; usa otro --output-dir.')
    manifest_path = input_dir / 'manifest.json'
    manifest_hash, policy_hash = digest(manifest_path), digest(policy_path)
    prepared = json.loads(manifest_path.read_text(encoding='utf-8'))
    policy = load_policy(policy_path)
    paths = {split: input_dir / f'{split}.csv' for split in PRIORITY}
    input_hashes = {split: digest(path) for split, path in paths.items()}
    for split in PRIORITY:
        if input_hashes[split] != prepared['outputs'][split]['sha256']:
            raise ValueError(f'La huella de {split} no coincide con el manifiesto preparado.')
    references = [row for split in ('test', 'validation') for row in read_rows(paths[split], split)]
    if progress:
        progress('Construyendo índice de test y validación; no se muestran textos.')
    index = SimilarityIndex(references, Fraction(policy['threshold']))
    allowed = {row['id'] for row in references if row['original_split'] == 'test'}
    seen_ids: set[str] = set()
    output_stats, removed, pair_counts = {}, {}, Counter()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # mkdir hereda permisos del padre en Windows; el destino solo aparece al finalizar.
    staging = output_dir.parent / f'.{output_dir.name}-{uuid.uuid4().hex}.tmp'
    staging.mkdir()
    try:
        with (staging / 'matches.csv').open('w', encoding='utf-8', newline='') as report:
            report_writer = csv.DictWriter(report, fieldnames=MATCH_COLUMNS, lineterminator='\n')
            report_writer.writeheader()
            for split in PRIORITY:
                counts, scanned, excluded = Counter(), 0, 0
                accepted_ids = set()
                with (staging / f'{split}.csv').open('w', encoding='utf-8', newline='') as output:
                    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator='\n')
                    writer.writeheader()
                    for row in read_rows(paths[split], split):
                        if row['id'] in seen_ids:
                            raise ValueError('Hay identificadores repetidos entre las entradas.')
                        seen_ids.add(row['id'])
                        scanned += 1
                        matches = [] if split == 'test' else index.matches(row['text'], allowed)
                        if matches:
                            excluded += 1
                            for reference_index, overlap, union in matches:
                                reference = references[reference_index]
                                pair_counts[f"{split}->{reference['original_split']}"] += 1
                                report_writer.writerow({
                                    'id': row['id'], 'original_split': split,
                                    'reference_id': reference['id'], 'reference_split': reference['original_split'],
                                    'intersection': overlap, 'union': union, 'jaccard': f'{overlap / union:.10f}',
                                })
                        else:
                            writer.writerow(row)
                            counts[row['valuation']] += 1
                            if split == 'validation':
                                accepted_ids.add(row['id'])
                        if progress and scanned % 10000 == 0:
                            progress(f'{split}: {scanned} revisados; {excluded} apartados.')
                if scanned != prepared['outputs'][split]['records']:
                    raise ValueError(f'El conteo de {split} no coincide con el manifiesto.')
                if set(counts) != set(LABELS):
                    raise ValueError(f'{split} no conserva las tres clases.')
                allowed.update(accepted_ids)
                output_stats[split] = {'records': sum(counts.values()),
                                       'by_valuation': {label: counts[label] for label in LABELS},
                                       'sha256': digest(staging / f'{split}.csv')}
                removed[split] = excluded
                if progress:
                    progress(f'{split}: terminado; {sum(counts.values())} conservados, {excluded} apartados.')
        for split, path in paths.items():
            if digest(path) != input_hashes[split]:
                raise ValueError('Una entrada cambió durante la auditoría.')
        if digest(manifest_path) != manifest_hash or digest(policy_path) != policy_hash:
            raise ValueError('La configuración cambió durante la auditoría.')
        # test se copia sin exclusiones ni cambios de formato.
        if output_stats['test']['sha256'] != input_hashes['test']:
            raise ValueError('La salida de test debe conservarse byte por byte.')
        manifest = {
            'stage': 'cross_split_similarity_filter', 'policy': policy,
            'policy_sha256': policy_hash, 'prepared_manifest_sha256': manifest_hash,
            'inputs_sha256': input_hashes, 'outputs': output_stats,
            'excluded_records': removed, 'matching_pairs': dict(sorted(pair_counts.items())),
            'matches_sha256': digest(staging / 'matches.csv'),
            'near_duplicates_checked': True,
            'cross_split_pairs_at_or_above_threshold_remaining': 0,
            'evaluation_status': 'No models fitted; no predictions or classification metrics computed.',
            'limitations': 'Lexical heuristic, not semantic equivalence. May exclude different opinions with similar wording and miss paraphrases. Within-split near duplicates are not compared. Rating/text consistency and actual language remain to be examined using train/validation.',
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                              encoding='utf-8', newline='\n')
        if output_dir.exists():
            raise FileExistsError('El destino apareció durante la auditoría; no se sobrescribió.')
        staging.rename(output_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = audit(args.input_dir, args.output_dir, progress=lambda message: print(message, flush=True))
    print(json.dumps({key: manifest[key] for key in ('outputs', 'excluded_records', 'matching_pairs')},
                     ensure_ascii=False, indent=2))
    print(f'CSV y auditoría: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
