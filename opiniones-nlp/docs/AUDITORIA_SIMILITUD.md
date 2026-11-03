# Auditoría de similitud entre particiones

## Objetivo y referencia

Cerrar el control de contaminación léxica entre entrenamiento, validación y prueba antes del EDA y del entrenamiento. Se conserva el criterio del proyecto de tickets (`src/data/build_splits.py`): Jaccard de trigramas de caracteres ≥ 0,85 y prioridad de evaluación. El nuevo corpus tiene tres particiones: `test → validation → train`.

La regla se fija en [similarity_policy.json](../data/similarity_policy.json) antes de entrenar. Las valoraciones, predicciones y métricas no intervienen en las decisiones. Es una regla conservadora de exclusión para separar datos, no un detector de equivalencia de significado.

## Método

1. Verificar las huellas y conteos de los CSV contra el manifiesto de la primera preparación.
2. Normalizar únicamente para comparar: Unicode NFKC, `casefold` y espacios colapsados. Se conservan signos, emojis y negaciones. Esta conservación de signos adapta la normalización del proyecto de tickets al dominio de opiniones.
3. Obtener el conjunto de trigramas de caracteres de cada texto; para cadenas menores de tres caracteres, usar la cadena completa como un único elemento.
4. Calcular `J(A,B) = |A ∩ B| / |A ∪ B|`, con comparación entera exacta contra `17/20` para evitar redondeos en el límite.
5. Conservar todo el test. Apartar de validación las coincidencias con test; después apartar de entrenamiento las coincidencias con test o con la validación conservada. No mover filas, reetiquetar ni equilibrar clases. No comparar pares de la misma partición.
6. Publicar una carpeta nueva solo después de verificar de nuevo las huellas de entrada, las tres clases de cada salida y la conservación exacta del test.

La búsqueda usa un índice de prefijos con orden global de trigramas por frecuencia y un filtro de longitudes. Para un conjunto de tamaño `n`, el prefijo contiene `n - ceil(0,85*n) + 1` elementos. Una pareja que supera el umbral debe compartir un elemento de sus prefijos; los candidatos se verifican con Jaccard completo. No hay muestreo ni búsqueda aproximada. Las pruebas comparan este índice con todos los pares en corpus pequeños, incluyendo tokens desconocidos, textos cortos y varios umbrales.

## Ejecución reproducible

Desde la raíz del proyecto, después de descargar y preparar los datos:

```powershell
& ./opiniones-nlp/.venv312/Scripts/python.exe opiniones-nlp/src/data/audit_similarity.py
```

Entrada: `opiniones-nlp/data/processed/amazon_three_levels/`.

Salida: `opiniones-nlp/data/processed/amazon_three_levels_clean/`.

- `train.csv`, `validation.csv`, `test.csv`: versión final para EDA y modelado; mismo esquema que la primera preparación.
- `matches.csv`: una fila por pareja detectada, con identificadores, particiones, intersección, unión y Jaccard. Una reseña excluida puede producir varias parejas. No contiene textos ni valoraciones.
- `manifest.json`: política completa y huella, huella del manifiesto de entrada, huellas de CSV de entrada y salida, conteos por valoración, exclusiones y parejas por partición.

Los originales y la preparación previa permanecen intactos. Todos los datos y reportes generados siguen excluidos de Git. Para repetir, elegir una salida inexistente:

```powershell
& ./opiniones-nlp/.venv312/Scripts/python.exe opiniones-nlp/src/data/audit_similarity.py --output-dir opiniones-nlp/data/processed/amazon_three_levels_clean_v2
```

## Resultado local

| Partición | Antes del control | Apartadas por similitud | Conservadas | Desfavorable | Intermedia | Favorable |
|---|---:|---:|---:|---:|---:|---:|
| Entrenamiento | 199.635 | 54 | 199.581 | 79.907 | 39.915 | 79.759 |
| Validación | 4.998 | 2 | 4.996 | 1.999 | 1.000 | 1.997 |
| Prueba | 5.000 | 0 | 5.000 | 2.000 | 1.000 | 2.000 |

Parejas detectadas (pueden compartir reseñas excluidas):

- `train->test`: 31.
- `train->validation`: 23.
- `validation->test`: 2.

Huellas SHA-256 de la versión final:

| Archivo | SHA-256 |
|---|---|
| `train.csv` | `4bf36da145727261b9b819a901aae5d6d83ae1c69c0ee6d04d5e64c40ce5cf4a` |
| `validation.csv` | `de28102011ab41372943a51d0d0941ef157dffbd854b3bf58b6e44eddc824a86` |
| `test.csv` | `f8f3fb9c43d3f7988e873fde529108f430fe9a77d2ec5f12e2f9426cd1f4576b` |

Estos conteos son resultados de preparación, no métricas de clasificación. El conjunto de prueba mantiene 5.000 registros y la misma huella que la primera preparación. Entre las salidas conservadas no queda ninguna pareja entre particiones que alcance el umbral bajo esta representación; no se afirma ausencia de toda similitud semántica.

## Verificación y límites

```powershell
Set-Location opiniones-nlp
& ./.venv312/Scripts/python.exe -m pytest -q
Set-Location ..
```

Las pruebas cubren equivalencia del índice con comparación exhaustiva, prioridad de particiones, conservación de texto y etiquetas, negaciones, ausencia de textos en el informe de parejas, reproducibilidad, rechazo de sobrescrituras y entradas modificadas, y limpieza de salidas incompletas.

La heurística puede apartar opiniones distintas con redacción parecida y omitir paráfrasis con vocabulario distinto. No detecta reseñas falsas ni duplicados por usuario o producto: esos metadatos no están en los archivos utilizados. No se comparan similitudes internas de una partición. Tampoco se verifica automáticamente el idioma real o que la valoración represente fielmente el texto.

El siguiente paso es `feat/eda`: analizar clases, longitudes, términos frecuentes y ruido con **train y validation de la versión final**, y revisar muestras de esos conjuntos. No mostrar ni explorar textos de test para elegir transformaciones, modelos o umbrales.
