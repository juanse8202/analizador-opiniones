"""Agrupa valoraciones y prepara CSV sin alterar los originales ni entrenar modelos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

try:
    from .download_amazon import verify_file
except ImportError:  # Ejecución directa desde la terminal.
    from download_amazon import verify_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/amazon_source.json'
POLICY = ROOT / 'data/label_mapping.json'
RAW = ROOT / 'data/raw/amazon_reviews_multi/es'
OUTPUT = ROOT / 'data/processed/amazon_three_levels'
PRIORITY = ('test', 'validation', 'train')
LABELS = ('desfavorable', 'intermedia', 'favorable')
COLUMNS = ('id', 'text', 'valuation', 'original_label', 'source', 'language', 'original_split')


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def text_key(text: str) -> bytes:
    # Solo control de duplicados; no sustituye el futuro procesamiento NLP.
    normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha256(normalized.encode('utf-8')).digest()


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding='utf-8'))
    expected = {'0': 'desfavorable', '1': 'desfavorable', '2': 'intermedia',
                '3': 'favorable', '4': 'favorable'}
    if (policy.get('mapping') != expected or policy.get('labels') != list(LABELS)
            or policy.get('target_column') != 'valuation'
            or policy.get('human_sentiment_annotation') is not False):
        raise ValueError('La política no coincide con la opción 1 documentada.')
    return policy


def convert_record(record: dict, split: str, policy: dict) -> dict:
    label, text, identifier = record.get('label'), record.get('text'), record.get('id')
    if type(label) is not int or label not in range(5):
        raise ValueError('La etiqueta original debe ser un entero de 0 a 4.')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('El texto debe ser una cadena no vacía.')
    if not isinstance(identifier, str) or not identifier.startswith('es_'):
        raise ValueError('El identificador debe corresponder al subconjunto español.')
    if str(record.get('label_text')) != str(label):
        raise ValueError('label_text no coincide con label.')
    return {'id': identifier, 'text': text,
            'valuation': policy['mapping'][str(label)], 'original_label': label,
            'source': 'mteb/amazon_reviews_multi', 'language': 'es', 'original_split': split}


def records(path: Path):
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('Se esperaba un objeto JSON.')
                yield record
            except ValueError as exc:
                raise ValueError(f'{path.name}, línea {number}: JSON inválido.') from exc


def prepare(input_dir: Path, output_dir: Path, source_path: Path = SOURCE,
            policy_path: Path = POLICY) -> dict:
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if output_dir == input_dir or output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir):
        raise ValueError('Las rutas de originales y resultados deben estar separadas.')
    if output_dir.exists():
        raise FileExistsError(f'{output_dir} ya existe; utiliza otro --output-dir para conservarlo.')
    source = json.loads(source_path.read_text(encoding='utf-8'))
    policy = load_policy(policy_path)
    if set(source['files']) != set(PRIORITY):
        raise ValueError('Se necesitan train, validation y test.')
    inputs = {}
    paths = {}
    for split in PRIORITY:
        metadata = source['files'][split]
        paths[split] = input_dir / Path(metadata['path']).name
        inputs[split] = verify_file(paths[split], metadata)

    # Cada huella pertenece a la primera partición que la contiene según PRIORITY.
    # Una contradicción de clases dentro de esa partición deja al grupo apartado.
    owners = {}
    identifiers = set()
    input_counts = {}
    for split in PRIORITY:
        labels = Counter()
        count = 0
        for record in records(paths[split]):
            row = convert_record(record, split, policy)
            if row['id'] in identifiers:
                raise ValueError('Se encontró un identificador repetido en los originales.')
            identifiers.add(row['id'])
            key = text_key(row['text'])
            if key not in owners:
                owners[key] = [split, row['valuation'], False]
            elif owners[key][0] == split and owners[key][1] != row['valuation']:
                owners[key][1] = None
            count += 1
            labels[str(row['original_label'])] += 1
        if count != source['files'][split]['expected_records']:
            raise ValueError(f'Cantidad inesperada en {split}.')
        input_counts[split] = {'records': count, 'original_label_counts': dict(sorted(labels.items()))}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_stats = {}
    removed = {}
    # Construir primero en una carpeta temporal para no publicar resultados incompletos.
    with tempfile.TemporaryDirectory(prefix='.amazon-prepare-', dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        with (staging / 'exclusions.csv').open('w', encoding='utf-8', newline='') as excluded:
            exclusion_writer = csv.DictWriter(excluded, fieldnames=('id', 'original_split', 'reason'),
                                               lineterminator='\n')
            exclusion_writer.writeheader()
            for split in PRIORITY:
                counts, reasons = Counter(), Counter()
                with (staging / f'{split}.csv').open('w', encoding='utf-8', newline='') as destination:
                    writer = csv.DictWriter(destination, fieldnames=COLUMNS, lineterminator='\n')
                    writer.writeheader()
                    for record in records(paths[split]):
                        row = convert_record(record, split, policy)
                        owner = owners[text_key(row['text'])]
                        if owner[0] != split:
                            reason = 'duplicate_in_higher_priority_split'
                        elif owner[1] is None:
                            reason = 'conflicting_grouped_labels_in_owner_split'
                        elif owner[2]:
                            reason = 'duplicate_within_split'
                        else:
                            reason = None
                        if reason:
                            reasons[reason] += 1
                            exclusion_writer.writerow({'id': row['id'], 'original_split': split, 'reason': reason})
                        else:
                            writer.writerow(row)
                            owner[2] = True
                            counts[row['valuation']] += 1
                if set(counts) != set(LABELS):
                    raise ValueError(f'{split} no conserva las tres clases tras la preparación.')
                output_stats[split] = {'records': sum(counts.values()),
                                       'by_valuation': {label: counts[label] for label in LABELS},
                                       'sha256': digest(staging / f'{split}.csv')}
                removed[split] = dict(sorted(reasons.items()))

        # Detectar modificaciones concurrentes de entradas o configuración.
        for split in PRIORITY:
            if digest(paths[split]) != inputs[split]['sha256']:
                raise ValueError('Un archivo original cambió durante la preparación.')
        if load_policy(policy_path) != policy or json.loads(source_path.read_text(encoding='utf-8')) != source:
            raise ValueError('La configuración cambió durante la preparación.')
        manifest = {
            'repository': source['download_repository'], 'revision': source['revision'],
            'source_metadata_sha256': digest(source_path), 'label_policy_sha256': digest(policy_path),
            'policy_version': policy['version'], 'mapping': policy['mapping'],
            'columns': list(COLUMNS), 'inputs': inputs, 'input_counts': input_counts,
            'outputs': output_stats, 'excluded_by_reason': removed,
            'exclusions_sha256': digest(staging / 'exclusions.csv'),
            'duplicate_policy': {'priority': list(PRIORITY), 'normalization': 'NFKC, casefold y espacios colapsados; puntuación y negaciones conservadas',
                                 'owner_split_label_conflicts': 'Excluir todas las copias del grupo'},
            'near_duplicates_checked': False,
            'evaluation_status': 'No se entrenaron modelos ni se evaluaron predicciones.',
            'limitations': 'Etiquetas derivadas de valoraciones; intermedia puede incluir opiniones mixtas. No se garantiza equivalencia con sentimientos anotados por personas. El control exacto no descarta paráfrasis ni textos similares.',
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                              encoding='utf-8', newline='\n')
        # Evitar sustituciones de datos existentes.
        if output_dir.exists():
            raise FileExistsError(f'{output_dir} apareció durante la preparación; no se sobrescribió.')
        staging.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=RAW)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.input_dir, args.output_dir)
    print(json.dumps({split: manifest['outputs'][split] for split in ('train', 'validation', 'test')},
                     ensure_ascii=False, indent=2))
    print(f'CSV y manifiesto guardados en {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
