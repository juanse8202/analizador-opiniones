"""Descarga y comprueba la copia MTEB de Amazon en español, sin asignar sentimientos."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/amazon_source.json'
DEFAULT_OUTPUT = ROOT / 'data/raw/amazon_reviews_multi/es'


def verify_file(path: Path, expected: dict) -> dict:
    size = path.stat().st_size
    if size != expected['bytes']:
        raise ValueError(f'Tamaño incorrecto de {path.name}: {size}.')
    sha256 = hashlib.sha256()
    # Los archivos pequeños son blobs Git; el archivo de entrenamiento usa SHA-256 LFS.
    blob = hashlib.sha1(f'blob {size}\0'.encode('ascii'))
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha256.update(chunk)
            blob.update(chunk)
    if expected.get('sha256') and sha256.hexdigest() != expected['sha256']:
        raise ValueError(f'SHA-256 incorrecto de {path.name}.')
    if expected.get('git_blob_sha1') and blob.hexdigest() != expected['git_blob_sha1']:
        raise ValueError(f'La huella Git de {path.name} no coincide con la revisión fijada.')
    return {'bytes': size, 'sha256': sha256.hexdigest()}


def download_file(source: dict, output: Path, split: str, offline: bool) -> tuple[str, Path, dict]:
    expected = source['files'][split]
    path = output / Path(expected['path']).name
    if path.exists():
        return split, path, verify_file(path, expected)
    if offline:
        raise FileNotFoundError(f'Falta {path}; ejecuta primero sin --offline.')
    url = f"{source['source_url']}/resolve/{source['revision']}/{expected['path']}"
    request = Request(url, headers={'User-Agent': 'Opiniones-Academic-Project/1.0'})
    temporary = path.with_suffix('.jsonl.part')
    # Conservar cualquier descarga parcial anterior para poder revisarla.
    if temporary.exists():
        raise FileExistsError(f'Existe una descarga parcial: {temporary}. Consérvala o muévela antes de reintentar.')
    print(f'Descargando {split}...', flush=True)
    with urlopen(request, timeout=60) as response, temporary.open('xb') as destination:
        shutil.copyfileobj(response, destination)
    fingerprints = verify_file(temporary, expected)
    temporary.rename(path)
    return split, path, fingerprints


def inspect_records(path: Path, expected_records: int) -> tuple[dict, set[str]]:
    counts = Counter()
    identifiers = set()
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            identifier = record.get('id')
            text = record.get('text')
            label = record.get('label')
            if not isinstance(identifier, str) or not identifier.startswith('es_'):
                raise ValueError(f'{path.name}, línea {line_number}: identificador inesperado.')
            if identifier in identifiers:
                raise ValueError(f'{path.name}, línea {line_number}: identificador duplicado.')
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'{path.name}, línea {line_number}: texto vacío.')
            if type(label) is not int or label not in range(5):
                raise ValueError(f'{path.name}, línea {line_number}: etiqueta inesperada.')
            if str(record.get('label_text')) != str(label):
                raise ValueError(f'{path.name}, línea {line_number}: label_text no coincide con label.')
            identifiers.add(identifier)
            counts[label] += 1
    if len(identifiers) != expected_records or set(counts) != set(range(5)):
        raise ValueError(f'Cantidad de registros o cobertura de etiquetas inesperada en {path.name}.')
    return {'records': len(identifiers), 'original_label_counts': dict(sorted(counts.items()))}, identifiers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--offline', action='store_true', help='Comprobar originales locales sin descargarlos.')
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = tuple(source['files'])
    with ThreadPoolExecutor(max_workers=3) as pool:
        downloaded = list(pool.map(lambda split: download_file(source, output, split, args.offline), names))
    reports = {}
    ids_by_split = {}
    for split, path, fingerprints in downloaded:
        records, identifiers = inspect_records(path, source['files'][split]['expected_records'])
        reports[split] = dict(file=path.name, **fingerprints, **records)
        ids_by_split[split] = identifiers
    for index, split in enumerate(names):
        for other in names[index + 1:]:
            if ids_by_split[split] & ids_by_split[other]:
                raise ValueError(f'Hay identificadores compartidos entre {split} y {other}.')
    manifest = {
        'repository': source['download_repository'],
        'revision': source['revision'],
        'language': source['language'],
        'source_metadata_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'files': reports,
        'status': 'raw_original_ratings',
        'limitations': 'Originales sin transformación ni entrenamiento; la agrupación se realiza por separado en prepare_dataset.py. La comprobación de identificadores no descarta textos duplicados o similares entre particiones. No se muestran textos de test.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: reports[name]['records'] for name in names}, ensure_ascii=False), flush=True)
    print(f'Originales y manifiesto verificados: {output}', flush=True)


if __name__ == '__main__':
    main()
