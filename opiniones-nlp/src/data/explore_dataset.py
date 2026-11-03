"""EDA reproducible de entrenamiento y validación; no abre test ni entrena ML."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import unicodedata
import uuid
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from spacy.lang.es.stop_words import STOP_WORDS

try:
    from .prepare_dataset import COLUMNS, LABELS, ROOT, digest, load_policy, text_key
except ImportError:  # Ejecución directa desde PowerShell.
    from prepare_dataset import COLUMNS, LABELS, ROOT, digest, load_policy, text_key

SPLITS = ('train', 'validation')
INPUT = ROOT / 'data/processed/amazon_three_levels_clean'
OUTPUT = ROOT / 'artifacts/eda'
POLICY = ROOT / 'data/label_mapping.json'
NEGATIONS = frozenset(('no', 'ni', 'sin', 'nunca', 'jamás', 'tampoco'))
FILTERED_STOP_WORDS = frozenset(STOP_WORDS) - NEGATIONS
TOKEN_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", flags=re.UNICODE)
FLAGS = ('short_text', 'no_word_tokens', 'negation', 'contrast_marker', 'url', 'email',
         'html_like', 'repeated_character', 'mostly_uppercase', 'other_unicode_symbol',
         'no_spanish_stopword_marker')
REVIEW_FIELDS = ('selection', 'id', 'original_split', 'valuation', 'original_label', 'text',
                 'signals', 'reviewer', 'language_observation', 'rating_text_relation', 'notes')


def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(unicodedata.normalize('NFKC', text).casefold())


def signals(text: str, words: list[str]) -> dict[str, bool]:
    vocabulary = set(words)
    letters = [char for char in text if char.isalpha()]
    return {
        'short_text': len(words) <= 3,
        'no_word_tokens': not words,
        'negation': bool(vocabulary & NEGATIONS),
        'contrast_marker': bool(vocabulary & {'pero', 'aunque'}) or 'sin embargo' in ' '.join(words),
        'url': bool(re.search(r'https?://|www\.', text, re.IGNORECASE)),
        'email': bool(re.search(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}', text)),
        'html_like': bool(re.search(r'<[^>]+>', text)),
        'repeated_character': bool(re.search(r'(.)\1{4,}', text)),
        'mostly_uppercase': len(letters) >= 10 and sum(c.isupper() for c in letters) / len(letters) >= 0.7,
        'other_unicode_symbol': any(unicodedata.category(c) == 'So' for c in text),
        'no_spanish_stopword_marker': len(words) >= 5 and not vocabulary.intersection(STOP_WORDS),
    }


def load_frames(input_dir: Path, policy_path: Path = POLICY):
    manifest_path = input_dir / 'manifest.json'
    manifest_hash, policy_hash = digest(manifest_path), digest(policy_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('stage') != 'cross_split_similarity_filter' or manifest.get('near_duplicates_checked') is not True:
        raise ValueError('El EDA requiere la preparación final con auditoría de similitud.')
    policy = load_policy(policy_path)
    frames, fingerprints = {}, {}
    identifiers = set()
    for split in SPLITS:  # Nunca construir una ruta de test ni explorar el directorio.
        path = input_dir / f'{split}.csv'
        fingerprints[split] = digest(path)
        if fingerprints[split] != manifest['outputs'][split]['sha256']:
            raise ValueError(f'Huella incorrecta en {split}.')
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if tuple(frame.columns) != COLUMNS or frame.empty:
            raise ValueError(f'Esquema o contenido inválido en {split}.')
        if (frame['id'].str.strip().eq('').any() or frame['text'].str.strip().eq('').any()
                or frame['id'].duplicated().any() or identifiers.intersection(frame['id'])
                or not frame['original_split'].eq(split).all()
                or not frame['language'].eq('es').all()
                or not frame['original_label'].isin(list(policy['mapping'])).all()
                or not frame['valuation'].eq(frame['original_label'].map(policy['mapping'])).all()):
            raise ValueError(f'Registros incompatibles con la preparación en {split}.')
        counts = {label: int(frame['valuation'].eq(label).sum()) for label in LABELS}
        if (len(frame) != manifest['outputs'][split]['records']
                or counts != manifest['outputs'][split]['by_valuation'] or not all(counts.values())):
            raise ValueError(f'Conteos incompatibles en {split}.')
        identifiers.update(frame['id'])
        frames[split] = frame
    return frames, fingerprints, manifest_hash, policy_hash


def write_csv(path: Path, rows: list[dict], columns):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def length_stats(values: pd.Series) -> dict:
    return {'count': int(len(values)), 'min': int(values.min()), 'max': int(values.max()),
            'mean': round(float(values.mean()), 4), 'std': round(float(values.std(ddof=0)), 4),
            **{f'p{percentile}': round(float(values.quantile(percentile / 100)), 4)
               for percentile in (5, 25, 50, 75, 95, 99)}}


def select_review(frames: dict, seed: int, sample_size: int) -> list[dict]:
    selected = {}
    for split, frame in frames.items():
        for label in LABELS:
            group = frame.loc[frame['valuation'].eq(label)].sort_values('id')
            for row in group.sample(n=min(sample_size, len(group)), random_state=seed).to_dict('records'):
                selected[row['id']] = (row, {'random_stratified'})
        # Casos dirigidos aparte de la muestra aleatoria; no estiman prevalencias.
        for flag in ('short_text', 'no_spanish_stopword_marker', 'contrast_marker',
                     'url', 'email', 'html_like', 'other_unicode_symbol'):
            candidates = frame.loc[frame[flag]].sort_values('id')
            for row in candidates.sample(n=min(2, len(candidates)), random_state=seed).to_dict('records'):
                if row['id'] not in selected:
                    selected[row['id']] = (row, set())
                selected[row['id']][1].add(f'flag:{flag}')
    return [{'selection': ';'.join(sorted(reasons)), **{key: row[key] for key in
             ('id', 'original_split', 'valuation', 'original_label', 'text')},
             'signals': ';'.join(flag for flag in FLAGS if row[flag]),
             'reviewer': '', 'language_observation': '', 'rating_text_relation': '', 'notes': ''}
            for _, (row, reasons) in sorted(selected.items())]


def make_plots(directory: Path, distribution: list[dict], lengths: list[dict],
               terms: list[dict], histograms: dict):
    # Evita ventanas y sitúa la caché en el proyecto, también al ejecutar en VS Code.
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parent / '.tools/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    palette = ('#b83c46', '#b08318', '#21836c')
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for ax, split in zip(axes, SPLITS):
            rows = [row for row in distribution if row['split'] == split]
            bars = ax.bar(LABELS, [row['percent'] for row in rows], color=palette)
            ax.bar_label(bars, labels=[f"{row['percent']:.2f}%\n(n={row['records']:,})" for row in rows], padding=3)
            ax.set(title=f'Valoraciones · {split}', ylabel='Porcentaje de reseñas', ylim=(0, 50))
        fig.savefig(directory / 'class_distribution.png', dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for ax, split in zip(axes, SPLITS):
            frame = histograms[split]
            # Mismo rango para ambos conjuntos; se limita solo la visualización.
            cap = max(1, int(histograms['train']['word_count'].quantile(0.99)))
            clipped = frame['word_count'].clip(upper=cap)
            ax.hist(clipped, bins=min(40, cap + 1), range=(0, cap + 1),
                    weights=[100 / len(frame)] * len(frame), color='#426ba3')
            ax.set(title=f'Extensión · {split}', xlabel=f'Palabras (≥ {cap} agrupadas en el extremo)',
                   ylabel='Porcentaje de reseñas')
        fig.savefig(directory / 'word_lengths.png', dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for ax, split in zip(axes, SPLITS):
            rows = [row for row in lengths if row['split'] == split and row['measure'] == 'words']
            ax.errorbar(range(3), [row['p50'] for row in rows],
                        yerr=([row['p50'] - row['p25'] for row in rows],
                              [row['p75'] - row['p50'] for row in rows]),
                        fmt='o', capsize=6, color='#426ba3')
            ax.set(xticks=range(3), xticklabels=LABELS, ylabel='Palabras', title=f'Mediana y rango P25–P75 · {split}')
        fig.savefig(directory / 'lengths_by_valuation.png', dpi=160)
        plt.close(fig)

        for split in SPLITS:
            fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharex=True, layout='constrained')
            for ax, label, color in zip(axes, LABELS, palette):
                rows = [row for row in terms if row['split'] == split and row['valuation'] == label
                        and row['view'] == 'without_stopwords'][:15][::-1]
                ax.barh([row['term'] for row in rows], [row['document_percent'] for row in rows], color=color)
                ax.set(title=label, xlabel='% de reseñas de la categoría')
            fig.suptitle(f'Términos frecuentes · {split}\nStop words descriptivas; negaciones conservadas')
            fig.savefig(directory / f'top_terms_{split}.png', dpi=160)
            plt.close(fig)
    return matplotlib.__version__


def run_eda(input_dir: Path, output_dir: Path, policy_path: Path = POLICY,
            seed: int = 42, sample_size: int = 5, top_n: int = 20, progress=None) -> dict:
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if output_dir == input_dir or output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir):
        raise ValueError('Entrada y salida deben estar separadas.')
    if output_dir.exists():
        raise FileExistsError(f'{output_dir} ya existe; usa otro --output-dir para conservar la revisión.')
    if sample_size < 1 or top_n < 1:
        raise ValueError('sample_size y top_n deben ser positivos.')
    frames, fingerprints, manifest_hash, policy_hash = load_frames(input_dir, policy_path)
    source_hash = digest(Path(__file__))
    distribution, lengths, terms, flag_rows, quality = [], [], [], [], {}
    normalized_keys = {}
    for split, frame in frames.items():
        if progress:
            progress(f'Analizando {split}: {len(frame)} reseñas.')
        keys = frame['text'].map(text_key)
        normalized_keys[split] = set(keys)
        quality[split] = {'missing_by_column': {column: int(frame[column].str.strip().eq('').sum()) for column in COLUMNS},
                          'normalized_duplicate_extra_rows': int(keys.duplicated().sum())}
        word_counts, flag_values = [], []
        total = {(label, view): Counter() for label in LABELS for view in ('raw', 'without_stopwords')}
        documents = {(label, view): Counter() for label in LABELS for view in ('raw', 'without_stopwords')}
        for text, label in frame[['text', 'valuation']].itertuples(index=False, name=None):
            words = tokens(text)
            word_counts.append(len(words))
            flag_values.append(signals(text, words))
            for view, selected in [('raw', words), ('without_stopwords',
                                                   [word for word in words if word not in FILTERED_STOP_WORDS])]:
                total[label, view].update(selected)
                documents[label, view].update(set(selected))
        frame['word_count'], frame['char_count'] = word_counts, frame['text'].str.len()
        for flag in FLAGS:
            frame[flag] = [values[flag] for values in flag_values]
        for label in LABELS:
            group = frame.loc[frame['valuation'].eq(label)]
            distribution.append({'split': split, 'valuation': label, 'records': len(group),
                                 'percent': round(100 * len(group) / len(frame), 4)})
            for measure, column in [('words', 'word_count'), ('characters', 'char_count')]:
                lengths.append({'split': split, 'valuation': label, 'measure': measure,
                                **length_stats(group[column])})
            for flag in FLAGS:
                count = int(group[flag].sum())
                flag_rows.append({'split': split, 'valuation': label, 'signal': flag,
                                  'records': count, 'percent': round(100 * count / len(group), 4)})
            for view in ('raw', 'without_stopwords'):
                order = sorted(documents[label, view], key=lambda term: (
                    -documents[label, view][term], -total[label, view][term], term))[:top_n]
                for term in order:
                    terms.append({'split': split, 'valuation': label, 'view': view, 'term': term,
                                  'frequency': total[label, view][term],
                                  'document_frequency': documents[label, view][term],
                                  'document_percent': round(100 * documents[label, view][term] / len(group), 4)})
    quality['train_validation_normalized_overlap'] = len(normalized_keys['train'] & normalized_keys['validation'])
    review = select_review(frames, seed, sample_size)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f'.{output_dir.name}-{uuid.uuid4().hex}.tmp'
    staging.mkdir()
    try:
        for filename, records in [('class_distribution.csv', distribution), ('length_summary.csv', lengths),
                                  ('top_terms.csv', terms), ('quality_signals.csv', flag_rows)]:
            write_csv(staging / filename, records, tuple(records[0]))
        write_csv(staging / 'review_sample.csv', review, REVIEW_FIELDS)
        matplotlib_version = make_plots(staging, distribution, lengths, terms, frames)
        for split in SPLITS:
            if digest(input_dir / f'{split}.csv') != fingerprints[split]:
                raise ValueError('Los datos cambiaron durante el análisis.')
        if digest(input_dir / 'manifest.json') != manifest_hash or digest(policy_path) != policy_hash:
            raise ValueError('El manifiesto o la política cambió durante el análisis.')
        summary = {
            'schema_version': 1, 'analysis_splits': list(SPLITS),
            'input_manifest_sha256': manifest_hash, 'inputs_sha256': fingerprints,
            'label_policy_sha256': policy_hash, 'code_sha256': source_hash,
            'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                         'matplotlib': matplotlib_version, 'spacy': spacy.__version__},
            'parameters': {'seed': seed, 'random_samples_per_class_and_split': sample_size,
                           'targeted_samples_per_signal_and_split': 2, 'top_n': top_n,
                           'short_text_max_words': 3, 'uppercase_min_letters': 10,
                           'uppercase_min_ratio': 0.7, 'character_repeat_min': 5,
                           'no_spanish_marker_min_words': 5, 'histogram_cap_train_percentile': 99},
            'stopwords': {'source': 'spacy.lang.es.stop_words.STOP_WORDS',
                          'protected_negations': sorted(NEGATIONS), 'effective_count': len(FILTERED_STOP_WORDS),
                          'sha256': hashlib.sha256('\n'.join(sorted(FILTERED_STOP_WORDS)).encode('utf-8')).hexdigest()},
            'records': {split: len(frame) for split, frame in frames.items()},
            'quality': quality, 'distribution': distribution, 'lengths': lengths,
            'review': {'records': len(review), 'random_stratified_records': sum(
                'random_stratified' in row['selection'] for row in review), 'annotation_status': 'unreviewed'},
            'outputs_sha256': {path.name: digest(path) for path in sorted(staging.iterdir())},
            'limitations': 'Exploratory lexical signals, not language identification or sentiment annotations. Test is not opened. No classifier fitted, no labels changed. Review sample has blank annotation fields; targeted rows do not estimate prevalence.',
        }
        (staging / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                                             encoding='utf-8', newline='\n')
        if output_dir.exists():
            raise FileExistsError('La salida apareció durante el análisis; no se sobrescribió.')
        staging.rename(output_dir)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_report(input_dir: Path, output_dir: Path) -> dict:
    """El notebook muestra agregados existentes solo si sus huellas coinciden."""
    summary = json.loads((output_dir / 'summary.json').read_text(encoding='utf-8'))
    if summary['analysis_splits'] != list(SPLITS):
        raise ValueError('El informe debe contener únicamente train y validation.')
    if digest(input_dir / 'manifest.json') != summary['input_manifest_sha256']:
        raise ValueError('El manifiesto de datos cambió: genera otro informe.')
    for split in SPLITS:
        if digest(input_dir / f'{split}.csv') != summary['inputs_sha256'][split]:
            raise ValueError(f'Cambió {split}: genera otro informe.')
    for name, expected in summary['outputs_sha256'].items():
        if Path(name).name != name or digest(output_dir / name) != expected:
            raise ValueError(f'El artefacto {name} cambió. Conserva las anotaciones en una copia separada.')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    summary = run_eda(args.input_dir, args.output_dir, seed=args.seed,
                      progress=lambda message: print(message, flush=True))
    print(json.dumps({key: summary[key] for key in ('records', 'quality', 'review')}, ensure_ascii=False, indent=2))
    print(f'EDA guardado en {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
