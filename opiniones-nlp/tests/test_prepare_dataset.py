import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.data.prepare_dataset import LABELS, convert_record, load_policy, prepare, text_key

POLICY_PATH = Path(__file__).resolve().parents[1] / 'data/label_mapping.json'


def record(identifier, text, label):
    return {'id': identifier, 'text': text, 'label': label, 'label_text': str(label)}


@pytest.mark.parametrize('label,expected', [(0, 'desfavorable'), (1, 'desfavorable'),
                                          (2, 'intermedia'), (3, 'favorable'), (4, 'favorable')])
def test_grouping_preserves_original_label_and_text(label, expected):
    text = '¡No volvería! El envío llegó tarde 😞\nConservar este texto.'
    row = convert_record(record('es_example', text, label), 'train', load_policy(POLICY_PATH))
    assert row['valuation'] == expected
    assert row['original_label'] == label and row['text'] == text
    assert row['original_split'] == 'train'


@pytest.mark.parametrize('label', [-1, 5, True, '2', None])
def test_invalid_original_labels_are_rejected(label):
    with pytest.raises(ValueError, match='entero'):
        convert_record(record('es_example', 'Comentario', label), 'train', load_policy(POLICY_PATH))


def fixture_inputs(tmp_path):
    directory = tmp_path / 'raw'
    directory.mkdir()
    rows = {split: [record(f'es_{split}_{index}', f'Comentario único {split} número {index}', index)
                    for index in range(5)] for split in ('train', 'validation', 'test')}
    rows['train'].append(record('es_cross_duplicate', rows['test'][0]['text'], 4))
    rows['validation'][0]['text'] = rows['train'][1]['text']
    rows['train'].append(record('es_within_duplicate', '  COMENTARIO ÚNICO train número 0  ', 0))
    rows['train'].extend([record('es_conflict_a', 'Opinión contradictoria idéntica', 0),
                          record('es_conflict_b', 'Opinión contradictoria idéntica', 4)])
    source = {'download_repository': 'fixture', 'revision': 'fixture-version', 'files': {}}
    for split, items in rows.items():
        path = directory / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in items),
                        encoding='utf-8', newline='\n')
        source['files'][split] = {'path': f'es/{path.name}', 'bytes': path.stat().st_size,
                                  'expected_records': len(items), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    source_path = tmp_path / 'source.json'
    source_path.write_text(json.dumps(source), encoding='utf-8')
    return directory, source_path


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def test_preparation_blocks_leakage_and_preserves_raw_files(tmp_path):
    raw, source = fixture_inputs(tmp_path)
    originals = {path.name: path.read_bytes() for path in raw.iterdir()}
    output = tmp_path / 'prepared'
    manifest = prepare(raw, output, source, POLICY_PATH)
    assert {split: metadata['records'] for split, metadata in manifest['outputs'].items()} == {
        'test': 5, 'validation': 5, 'train': 4}
    fingerprints = set()
    for split in ('test', 'validation', 'train'):
        rows = read_csv(output / f'{split}.csv')
        assert {row['valuation'] for row in rows} == set(LABELS)
        for row in rows:
            fingerprint = text_key(row['text'])
            assert fingerprint not in fingerprints
            fingerprints.add(fingerprint)
            assert row['original_split'] == split
    exclusions = read_csv(output / 'exclusions.csv')
    excluded_ids = {row['id'] for row in exclusions}
    assert {'es_cross_duplicate', 'es_train_1', 'es_within_duplicate',
            'es_conflict_a', 'es_conflict_b'} <= excluded_ids
    assert manifest['excluded_by_reason']['train']['duplicate_in_higher_priority_split'] == 2
    assert manifest['excluded_by_reason']['train']['conflicting_grouped_labels_in_owner_split'] == 2
    assert manifest['near_duplicates_checked'] is False
    assert originals == {path.name: path.read_bytes() for path in raw.iterdir()}
    before = (output / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        prepare(raw, output, source, POLICY_PATH)
    assert (output / 'manifest.json').read_bytes() == before


def test_outputs_are_reproducible(tmp_path):
    raw, source = fixture_inputs(tmp_path)
    first, second = tmp_path / 'first', tmp_path / 'second'
    prepare(raw, first, source, POLICY_PATH)
    prepare(raw, second, source, POLICY_PATH)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}


def test_changed_input_is_rejected_before_publishing_outputs(tmp_path):
    raw, source = fixture_inputs(tmp_path)
    with (raw / 'train.jsonl').open('a', encoding='utf-8') as stream:
        stream.write('\n')
    output = tmp_path / 'prepared'
    with pytest.raises(ValueError):
        prepare(raw, output, source, POLICY_PATH)
    assert not output.exists()
