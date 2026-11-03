import csv
import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from src.data.audit_similarity import SimilarityIndex, audit, grams
from src.data.prepare_dataset import COLUMNS, LABELS, PRIORITY, digest

POLICY_PATH = Path(__file__).resolve().parents[1] / 'data/similarity_policy.json'


@pytest.mark.parametrize('threshold', ['0.25', '0.5', '0.85', '1'])
def test_prefix_index_matches_exhaustive_comparison(threshold):
    rng = random.Random(2026)
    base = 'El producto funciona correctamente y el envío llegó a tiempo. '
    texts = [base, base.upper(), 'a', 'ab', 'abc', '¡Sí!']
    for _ in range(65):
        text = list(base)
        for _ in range(rng.randrange(1, 12)):
            text[rng.randrange(len(text))] = rng.choice('xyz ñ')
        texts.append(''.join(text))
    references = [{'id': str(i), 'text': text} for i, text in enumerate(texts)]
    value = Fraction(threshold)
    index = SimilarityIndex(references, value)
    allowed = {row['id'] for row in references if int(row['id']) % 3 != 0}
    queries = texts + ['zzzzzz', 'これは全く別の文章です', base + ' PALABRAS NUEVAS 12345']
    for text in queries:
        query = grams(text)
        expected = []
        for i, row in enumerate(references):
            other = grams(row['text'])
            overlap, union = len(query & other), len(query | other)
            if row['id'] in allowed and Fraction(overlap, union) >= value:
                expected.append((i, overlap, union))
        assert index.matches(text, allowed) == expected


def test_normalization_preserves_negation_punctuation_and_unicode():
    assert grams('  ＨＯＬＡ\n mundo  ') == grams('hola mundo')
    assert grams('No sirve 😞!') != grams('Sirve 😞!')
    assert grams('Bueno!') != grams('Bueno?')
    with pytest.raises(ValueError, match='vacío'):
        grams(' \n ')


def test_exact_threshold_boundary_is_included_in_both_directions():
    short = ''.join(chr(33 + i) for i in range(19))  # 17 trigramas únicos.
    long = ''.join(chr(33 + i) for i in range(22))  # 20 trigramas únicos.
    for reference, query in [(short, long), (long, short)]:
        index = SimilarityIndex([{'id': 'reference', 'text': reference}], Fraction('0.85'))
        assert index.matches(query, {'reference'}) == [(0, 17, 20)]
    assert SimilarityIndex([{'id': 'reference', 'text': short}], Fraction('0.85')).matches(
        long + 'z', {'reference'}) == []


def fixture_prepared(tmp_path):
    root = tmp_path / 'prepared'
    root.mkdir()
    texts = {
        'test': ['El cargador no funciona y el paquete llegó roto.',
                 'La botella tiene una capacidad de setecientos mililitros.',
                 'Disfruto mucho usando estos auriculares en el autobús.'],
        'validation': ['La lámpara ilumina mal y tampoco ajusta a mi escritorio.',
                       'El cargador no funciona y el paquete llegó roto!',
                       'Este libro histórico narra una aventura maravillosa.',
                       'Está bastante bien', 'Está bastante bien.'],
        'train': ['Pésimo calzado, me hace daño al caminar por la calle.',
                  'El aparato ofrece una velocidad de impresión aceptable.',
                  'Me encanta este reloj para salir a correr cada mañana.',
                  'La lámpara ilumina mal y tampoco ajusta a mi escritorio!',
                  # Solo coincide con validation: el índice debe esperar al final
                  # de esa partición para activarla, sin excluir filas de su interior.
                  'Está bastante bien!'],
    }
    manifest = {'outputs': {}}
    for split in PRIORITY:
        with (root / f'{split}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
            writer.writeheader()
            for i, text in enumerate(texts[split]):
                label_index = 1 if split == 'validation' and i == 3 else i % 3
                writer.writerow({'id': f'es_{split}_{i}', 'text': text,
                                 'valuation': LABELS[label_index], 'original_label': str(i % 5),
                                 'source': 'fixture', 'language': 'es', 'original_split': split})
        manifest['outputs'][split] = {'records': len(texts[split]), 'sha256': digest(root / f'{split}.csv')}
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return root


def csv_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def test_filter_protects_test_and_prevents_cross_split_similarities(tmp_path):
    prepared = fixture_prepared(tmp_path)
    original = {path.name: path.read_bytes() for path in prepared.iterdir()}
    destination = tmp_path / 'clean'
    manifest = audit(prepared, destination, POLICY_PATH)
    assert manifest['excluded_records'] == {'test': 0, 'validation': 1, 'train': 2}
    assert (destination / 'test.csv').read_bytes() == original['test.csv']
    reports = csv_rows(destination / 'matches.csv')
    assert {row['id'] for row in reports} == {'es_validation_1', 'es_train_3', 'es_train_4'}
    assert all('text' not in column for column in reports[0])
    assert 'El cargador' not in (destination / 'matches.csv').read_text(encoding='utf-8')
    previous = []
    for split in PRIORITY:
        retained = csv_rows(destination / f'{split}.csv')
        assert {row['valuation'] for row in retained} == set(LABELS)
        for row in retained:
            for other in previous:
                a, b = grams(row['text']), grams(other['text'])
                assert Fraction(len(a & b), len(a | b)) < Fraction('0.85')
            assert row in csv_rows(prepared / f'{split}.csv')
        previous.extend(retained)
    assert original == {path.name: path.read_bytes() for path in prepared.iterdir()}
    assert manifest['near_duplicates_checked'] is True
    second = tmp_path / 'second'
    audit(prepared, second, POLICY_PATH)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    with pytest.raises(FileExistsError):
        audit(prepared, destination, POLICY_PATH)
    with pytest.raises(ValueError, match='separadas'):
        audit(prepared, prepared / 'nested', POLICY_PATH)


def test_changed_csv_is_rejected_before_publishing(tmp_path):
    prepared = fixture_prepared(tmp_path)
    with (prepared / 'train.csv').open('a', encoding='utf-8') as stream:
        stream.write('\n')
    destination = tmp_path / 'clean'
    with pytest.raises(ValueError, match='huella'):
        audit(prepared, destination, POLICY_PATH)
    assert not destination.exists()


def test_changed_input_during_audit_is_rejected_and_staging_removed(tmp_path):
    prepared = fixture_prepared(tmp_path)
    destination = tmp_path / 'clean'

    def change_input(message):
        if message.startswith('test: terminado'):
            with (prepared / 'test.csv').open('a', encoding='utf-8') as stream:
                stream.write('\n')

    with pytest.raises(ValueError, match='cambió'):
        audit(prepared, destination, POLICY_PATH, progress=change_input)
    assert not destination.exists()
    assert not list(tmp_path.glob('.clean-*.tmp'))


def rewrite_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def refresh_manifest(prepared):
    manifest = {'outputs': {split: {'records': len(csv_rows(prepared / f'{split}.csv')),
                                   'sha256': digest(prepared / f'{split}.csv')} for split in PRIORITY}}
    (prepared / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def test_exclusion_decisions_do_not_depend_on_ratings(tmp_path):
    prepared = fixture_prepared(tmp_path)
    audit(prepared, tmp_path / 'first', POLICY_PATH)
    for split in PRIORITY:
        rows = csv_rows(prepared / f'{split}.csv')
        for row in rows:
            row['valuation'] = LABELS[(LABELS.index(row['valuation']) + 1) % 3]
            row['original_label'] = str((int(row['original_label']) + 2) % 5)
        rewrite_csv(prepared / f'{split}.csv', rows)
    refresh_manifest(prepared)
    audit(prepared, tmp_path / 'second', POLICY_PATH)
    assert (tmp_path / 'first/matches.csv').read_bytes() == (tmp_path / 'second/matches.csv').read_bytes()


def test_removed_validation_does_not_remove_train_through_a_similarity_chain(tmp_path):
    prepared = fixture_prepared(tmp_path)
    a = ''.join(chr(33 + i) for i in range(60))
    b = '~~~' + a[3:]
    c = b[:-3] + '|||'
    for left, right, expected in [(a, b, True), (b, c, True), (a, c, False)]:
        x, y = grams(left), grams(right)
        assert (Fraction(len(x & y), len(x | y)) >= Fraction('0.85')) is expected
    for split, text in [('test', a), ('validation', b), ('train', c)]:
        rows = csv_rows(prepared / f'{split}.csv')
        rows.append(dict(rows[0], id=f'es_chain_{split}', text=text))
        rewrite_csv(prepared / f'{split}.csv', rows)
    refresh_manifest(prepared)
    output = tmp_path / 'clean'
    audit(prepared, output, POLICY_PATH)
    assert 'es_chain_validation' not in {row['id'] for row in csv_rows(output / 'validation.csv')}
    assert 'es_chain_train' in {row['id'] for row in csv_rows(output / 'train.csv')}
