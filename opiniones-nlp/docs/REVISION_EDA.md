# Revisión asistida de la muestra del EDA

## Alcance y responsable

Se revisaron individualmente las 47 reseñas completas de la muestra de entrenamiento y validación del paso 4. Responsable: **Codex (IA; revisión asistida)**. No se atribuye la revisión al estudiante ni se presenta como una evaluación humana independiente. Esta última seguirá siendo una actividad separada del paso 9.

Se completaron `reviewer`, `language_observation`, `rating_text_relation` y `notes` en una copia local `artifacts/eda/review_annotations.csv`. Los identificadores, textos, valoraciones, particiones y señales originales se conservaron. `review_sample.csv` mantiene sus campos vacíos como plantilla reproducible del generador.

## Criterios

- **compatible:** El texto permite una valoración global coherente con la categoría; no certifica la puntuación original.
- **ambigua:** Hay varias lecturas o intensidades globales plausibles; se requiere juicio adicional.
- **discrepante:** El juicio aparente del texto contrasta con la categoría asignada; candidato, no error confirmado.
- **insuficiente:** Falta un juicio evaluativo global suficiente para contrastar la categoría.

## Resultado

Los 47 textos revisados se observaron predominantemente en español, incluidos casos telegráficos y con vocabulario técnico. Esta observación de la muestra no valida automáticamente el idioma de todo el corpus.

| Relación texto–valoración | Muestra aleatoria estratificada | Casos dirigidos adicionales | Total |
|---|---:|---:|---:|
| compatible | 26 | 13 | 39 |
| ambigua | 3 | 3 | 6 |
| discrepante | 1 | 0 | 1 |
| insuficiente | 0 | 1 | 1 |

Los grupos contienen 30 reseñas aleatorias estratificadas (cinco por clase y conjunto) y 17 casos dirigidos adicionales. Los conteos describen únicamente esta muestra y estos criterios asistidos; no son métricas del clasificador, tasas de errores del corpus ni acuerdo entre anotadores.

Se detectó un candidato de discrepancia: una reseña con categoría favorable describe una falla reiterada y una solución indirecta sin elogios claros. Los casos ambiguos incluyen satisfacción de intensidad poco clara, opiniones mixtas y quejas sobre aspectos concretos sin valoración global explícita. El caso insuficiente confirma recepción sin aportar un juicio global suficiente. Las notas individuales quedan en el CSV local para facilitar una comprobación humana posterior.

La ausencia de stop words españolas produjo candidatos que seguían siendo español. Los dos fragmentos parecidos a HTML correspondían a una expresión entre signos y a la notación de una tecla. No se eliminaron filas por estas señales ni se modificaron etiquetas a partir del juicio asistido.

## Trazabilidad

- SHA-256 de `review_sample.csv`: `b64f8b0f8b4b2826e3d745cc49d7a200a90ac10e6a8b2ee1f38b52012dd6b476`.
- SHA-256 de `review_annotations.csv`: `30a98fd23df1ce9fa36712804d6f5dadd0880b6d4be5a0b08c1aab9972eebb83`.
- `artifacts/eda/review_annotations_summary.json` registra criterios, responsable, fecha UTC y conteos, sin duplicar textos.
- Los archivos de anotaciones permanecen locales y excluidos de Git; este informe versiona el resultado agregado.
- La verificación comprueba correspondencia uno a uno de 47 identificadores, conservación de todas las columnas originales y completitud de los cuatro campos de revisión.

## Cierre del paso 4

El EDA y la revisión asistida solicitada quedan documentados. Se mantiene pendiente la evaluación humana independiente del paso 9. El siguiente objetivo de implementación es el preprocesamiento NLP en `feat/preprocesamiento`, conservando las negaciones y las etiquetas de referencia.
