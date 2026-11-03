# Amazon Reviews Multi — español y tres niveles de valoración

## Procedencia y versión

- **Corpus original:** Multilingual Amazon Reviews Corpus, de Phillip Keung, Yichao Lu, György Szarvas y Noah A. Smith (2020).
- **Copia de descarga:** [mteb/amazon_reviews_multi](https://huggingface.co/datasets/mteb/amazon_reviews_multi).
- **Subconjunto:** `es`; `all_languages` reúne varios idiomas.
- **Revisión fijada:** `c379a6705fec24a2493fa68e011692605f44e119`.
- **Fecha de revisión de la fuente:** 2 de noviembre de 2026.
- **Configuración y huellas:** [amazon_source.json](../data/amazon_source.json).
- **Política de agrupación:** [label_mapping.json](../data/label_mapping.json), versión `amazon-three-levels-v1`.

El dominio de la primera versión comprende reseñas de productos y experiencias de compra, entrega y atención asociadas. El objetivo es estimar la valoración global agrupada a partir del texto.

## Disponibilidad y condiciones documentadas

La [ficha del corpus original](https://huggingface.co/datasets/defunct-datasets/amazon_reviews_multi) informa de que su distribución original fue retirada por decisión de sus proveedores. Este proyecto identifica explícitamente la copia pública de MTEB utilizada.

El [cargador de esa copia](https://huggingface.co/datasets/mteb/amazon_reviews_multi/blob/c379a6705fec24a2493fa68e011692605f44e119/amazon_reviews_multi.py) reproduce condiciones propias de Amazon para investigación académica, con restricciones de republicación y uso comercial. No se presenta como un dataset MIT o CC. Los originales y los resultados preparados quedan excluidos de Git; se versionan los scripts, la política y las referencias necesarias para reproducirlos.

## Archivos originales

| Partición | Archivo | Registros originales |
|---|---|---:|
| Entrenamiento | `train.jsonl` | 200.000 |
| Validación | `validation.jsonl` | 5.000 |
| Prueba | `test.jsonl` | 5.000 |
| **Total** | | **210.000** |

Formato JSON Lines UTF-8, con campos `id`, `text`, `label` y `label_text`. Los originales se conservan byte por byte; el preparador solo los lee. Sus tamaños y huellas esperados se verifican antes de generar resultados.

## Adaptación aprobada: opción 1

| `label` original | Nueva etiqueta `valuation` | Interpretación |
|---|---|---|
| `0` o `1` | `desfavorable` | Dos niveles inferiores de valoración. |
| `2` | `intermedia` | Punto medio de la escala original. |
| `3` o `4` | `favorable` | Dos niveles superiores de valoración. |

La política se aplica por igual a entrenamiento, validación y prueba. Se conserva `original_label` para mantener la trazabilidad. No se ajusta el mapeo con predicciones, resultados de modelos ni inspección manual del test.

**Intermedia no equivale a neutral.** Puede incluir comentarios mixtos, moderados o de satisfacción parcial. Las etiquetas se derivan de la valoración del cliente y no de una nueva anotación humana del sentimiento del texto. Las opiniones mixtas conservan el grupo correspondiente a la valoración original; no se excluyen por ser mixtas ni se reasignan por palabras aisladas.

Las métricas futuras medirán acuerdo con estas categorías de valoración. Los desacuerdos entre texto y valoración forman parte del ruido de supervisión y deben describirse al interpretar los errores.

Antes de limpiar duplicados, la agrupación produce:

| Partición | Desfavorable | Intermedia | Favorable |
|---|---:|---:|---:|
| Entrenamiento | 80.000 | 40.000 | 80.000 |
| Validación | 2.000 | 1.000 | 2.000 |
| Prueba | 2.000 | 1.000 | 2.000 |

No se altera la distribución del test para equilibrarla artificialmente. Los conteos efectivos posteriores se registran en el manifiesto preparado.

### Resultado de la primera preparación local (antes de similitudes)

| Partición | Desfavorable | Intermedia | Favorable | Total conservado |
|---|---:|---:|---:|---:|
| Entrenamiento | 79.930 | 39.917 | 79.788 | 199.635 |
| Validación | 1.999 | 1.000 | 1.999 | 4.998 |
| Prueba | 2.000 | 1.000 | 2.000 | 5.000 |

Se excluyeron 367 registros: 190 copias dentro de entrenamiento, 59 copias presentes en particiones de mayor prioridad y 118 registros de grupos con valoraciones agrupadas contradictorias dentro de entrenamiento. Estas cifras describen la preparación de datos; no son resultados de clasificación. Los 210.000 originales permanecen sin cambios.

## Descarga y comprobación

Desde la raíz `analizador-opiniones`, con el entorno del paso 2:

```powershell
& ./opiniones-nlp/.venv312/Scripts/python.exe opiniones-nlp/src/data/download_amazon.py
& ./opiniones-nlp/.venv312/Scripts/python.exe opiniones-nlp/src/data/download_amazon.py --offline
```

Se obtienen los originales en `opiniones-nlp/data/raw/amazon_reviews_multi/es/`, junto con un manifiesto local de huellas, conteos y distribución de las cinco etiquetas.

La descarga usa una revisión fija y conserva descargas parciales si hay errores. Los archivos existentes que no coinciden se rechazan sin sobrescribirlos. El script no requiere dependencias externas ni ejecuta código del repositorio descargado.

## Preparación de los CSV

```powershell
& ./opiniones-nlp/.venv312/Scripts/python.exe opiniones-nlp/src/data/prepare_dataset.py
```

Salida: `opiniones-nlp/data/processed/amazon_three_levels/`.

- `train.csv`, `validation.csv`, `test.csv`: registros conservados y etiquetas agrupadas.
- `exclusions.csv`: identificador, partición y motivo de cada exclusión, sin textos.
- `manifest.json`: huellas de entradas, configuración y salidas; conteos y exclusiones por motivo.

Columnas: `id,text,valuation,original_label,source,language,original_split`. El campo `text` se conserva íntegro. La normalización usada para comparar duplicados no se escribe sobre el texto ni constituye el pipeline NLP.

El directorio de salida debe ser nuevo. Para repetir conservando los resultados anteriores, usar otro destino:

```powershell
& ./opiniones-nlp/.venv312/Scripts/python.exe opiniones-nlp/src/data/prepare_dataset.py --output-dir opiniones-nlp/data/processed/amazon_three_levels_v2
```

## Duplicados y protección de particiones

Se conservan las particiones originales: las filas nunca se trasladan entre conjuntos. La preparación no hace muestreo aleatorio ni requiere una nueva semilla.

1. Comparar textos mediante Unicode NFKC, `casefold` y espacios colapsados; conservar signos, números y negaciones.
2. Aplicar prioridad `test → validation → train` para evitar que copias de evaluación permanezcan en entrenamiento.
3. Dentro de una partición, conservar la primera copia si las categorías agrupadas coinciden.
4. Si un grupo de textos idénticos de la partición prioritaria tiene categorías agrupadas contradictorias, excluir todas sus copias, incluidas las de menor prioridad, y registrar los motivos.

La lectura automática del test solo comprueba integridad y evita duplicación; no muestra sus textos ni calcula rendimiento de modelos. La exclusión de inconsistencias puede reducir su tamaño, de forma registrada y previa al entrenamiento.

## Comprobaciones y limitaciones

El preparador valida identificadores, texto no vacío, rango de etiquetas, huellas y conteos de entrada. Publica los resultados únicamente cuando completa la preparación y comprueba que cada partición conserva las tres clases. Rechaza originales modificados y directorios de salida existentes.

Las pruebas comprueban agrupación, conservación de textos y etiquetas originales, protección de particiones, tratamiento de duplicados contradictorios, reproducibilidad y rechazo de sobrescrituras:

```powershell
Set-Location opiniones-nlp
& ./.venv312/Scripts/python.exe -m pytest -q
Set-Location ..
```

La auditoría posterior está implementada en `src/data/audit_similarity.py` y documentada en [AUDITORIA_SIMILITUD.md](AUDITORIA_SIMILITUD.md). Lee la primera preparación y genera `data/processed/amazon_three_levels_clean/`; esta es la versión que deben consumir EDA y modelos. El manifiesto original conserva `near_duplicates_checked: false` porque describe una etapa previa; el nuevo manifiesto registra `true` junto con el alcance y umbral exactos de la comprobación.

**Pendiente para el EDA:** revisar consistencia de valoraciones, idioma real y cobertura con entrenamiento y validación. La detección léxica no garantiza ausencia de paráfrasis, ni compara similitudes dentro de cada partición. No se han entrenado modelos ni utilizado el test para seleccionar configuraciones.
