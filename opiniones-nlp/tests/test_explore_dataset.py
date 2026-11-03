import csv
import json
from pathlib import Path

import pytest

from src.data.explore_dataset import (FILTERED_STOP_WORDS, NEGATIONS, run_eda,
                                      signals, tokens, verify_report)
from src.data.prepare_dataset import COLUMNS, LABELS, digest

POLICY = Path(__file__).resolve().parents[1] / 'data/label_mapping.json'


def fixture_prepared(tmp_path):
    directory = tmp_path / 'clean'
    directory.mkdir()
    examples = {
        'train': [('Malo malo, no funciona.', 0), ('Envase roto.', 1),
                  ('Entrega normal, aunque el tamaño varía.', 2),
                  ('Bueno bueno. Muy práctico.', 3), ('Funciona bien 🙂', 4)],
        'validation': [('No sirve.', 0), ('La calidad es aceptable pero pesa.', 2),
                       ('Perfecto para mis viajes.', 4)],
    }
    mapping = json.loads(POLICY.read_text(encoding='utf-8'))['mapping']
    manifest = {'stage': 'cross_split_similarity_filter', 'near_duplicates_checked': True, 'outputs': {}}
    for split, items in examples.items():
        with (directory / f'{split}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
            writer.writeheader()
            for index, (text, original) in enumerate(items):
                writer.writerow({'id': f'es_{split}_{index}', 'text': text,
                                 'valuation': mapping[str(original)], 'original_label': original,
                                 'source': 'fixture', 'language': 'es', 'original_split': split})
        counts = {label: sum(mapping[str(original)] == label for _, original in items) for label in LABELS}
        manifest['outputs'][split] = {'records': len(items), 'by_valuation': counts,
                                      'sha256': digest(directory / f'{split}.csv')}
    # No existe test.csv: el análisis debe funcionar sin abrirlo.
    (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return directory


def csv_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def test_descriptive_tokens_keep_negations_and_unicode():
    assert tokens('  ¡ＮＯ funciona! Ni jamás compraría 123. 😞') == [
        'no', 'funciona', 'ni', 'jamás', 'compraría']
    assert NEGATIONS.isdisjoint(FILTERED_STOP_WORDS)
    assert signals('No sirve, pero tiene buena pinta.', tokens('No sirve, pero tiene buena pinta.'))['contrast_marker']
    assert signals('🙂!!!', tokens('🙂!!!'))['no_word_tokens']
    assert signals('🙂!!!', tokens('🙂!!!'))['other_unicode_symbol']
    assert signals('excellent durable lightweight stylish superb', tokens(
        'excellent durable lightweight stylish superb'))['no_spanish_stopword_marker']
    assert not signals('El producto es muy bueno', tokens('El producto es muy bueno'))['no_spanish_stopword_marker']


def test_eda_generates_consistent_reports_without_opening_test(tmp_path, monkeypatch):
    prepared = fixture_prepared(tmp_path)
    originals = {p.name: p.read_bytes() for p in prepared.iterdir()}
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.name == 'test.csv':
            raise AssertionError('El EDA intentó abrir test.csv.')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)
    output = tmp_path / 'eda'
    summary = run_eda(prepared, output, POLICY, sample_size=1, top_n=30)
    assert summary['records'] == {'train': 5, 'validation': 3}
    assert summary['analysis_splits'] == ['train', 'validation']
    assert summary['quality']['train_validation_normalized_overlap'] == 0
    assert all(value == 0 for value in summary['quality']['train']['missing_by_column'].values())
    distribution = csv_rows(output / 'class_distribution.csv')
    assert [int(row['records']) for row in distribution if row['split'] == 'train'] == [2, 1, 2]
    terms = csv_rows(output / 'top_terms.csv')
    repeated = next(row for row in terms if row['split'] == 'train' and row['valuation'] == 'desfavorable'
                    and row['view'] == 'raw' and row['term'] == 'malo')
    assert int(repeated['frequency']) == 2 and int(repeated['document_frequency']) == 1
    assert any(row['term'] == 'no' and row['view'] == 'without_stopwords' for row in terms)
    review = csv_rows(output / 'review_sample.csv')
    assert all(row['original_split'] in {'train', 'validation'} for row in review)
    assert all(not row['reviewer'] and not row['rating_text_relation'] for row in review)
    assert len({row['id'] for row in review}) == len(review)
    assert summary['review']['random_stratified_records'] == 6
    assert summary['review']['annotation_status'] == 'unreviewed'
    assert all((output / name).stat().st_size > 1000 for name in summary['outputs_sha256'] if name.endswith('.png'))
    assert verify_report(prepared, output) == summary
    assert originals == {p.name: p.read_bytes() for p in prepared.iterdir()}
    with pytest.raises(FileExistsError):
        run_eda(prepared, output, POLICY)
    with pytest.raises(ValueError, match='separadas'):
        run_eda(prepared, prepared / 'nested', POLICY)


def test_changed_input_is_rejected_without_output(tmp_path):
    prepared = fixture_prepared(tmp_path)
    with (prepared / 'train.csv').open('a', encoding='utf-8') as stream:
        stream.write('\n')
    output = tmp_path / 'eda'
    with pytest.raises(ValueError, match='Huella'):
        run_eda(prepared, output, POLICY)
    assert not output.exists()


def test_wrong_mapping_is_rejected_even_with_updated_hash(tmp_path):
    prepared = fixture_prepared(tmp_path)
    rows = csv_rows(prepared / 'validation.csv')
    rows[0]['valuation'] = 'favorable'
    with (prepared / 'validation.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    path = prepared / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['outputs']['validation']['sha256'] = digest(prepared / 'validation.csv')
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='incompatibles'):
        run_eda(prepared, tmp_path / 'eda', POLICY)


def test_artifact_tampering_is_detected_and_samples_are_reproducible(tmp_path):
    prepared = fixture_prepared(tmp_path)
    first, second = tmp_path / 'first', tmp_path / 'second'
    a = run_eda(prepared, first, POLICY, sample_size=1, top_n=5)
    b = run_eda(prepared, second, POLICY, sample_size=1, top_n=5)
    for filename in ('review_sample.csv', 'class_distribution.csv', 'length_summary.csv',
                     'top_terms.csv', 'quality_signals.csv'):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert a['quality'] == b['quality'] and a['review'] == b['review']
    with (first / 'class_distribution.csv').open('a', encoding='utf-8') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='artefacto'):
        verify_report(prepared, first)


def test_changed_input_during_analysis_does_not_publish_results(tmp_path):
    prepared = fixture_prepared(tmp_path)

    def change_input(message):
        if message.startswith('Analizando train:'):
            with (prepared / 'train.csv').open('a', encoding='utf-8') as stream:
                stream.write('\n')

    with pytest.raises(ValueError, match='cambiaron'):
        run_eda(prepared, tmp_path / 'eda', POLICY, progress=change_input)
    assert not (tmp_path / 'eda').exists()
    assert not list(tmp_path.glob('.eda-*.tmp'))
