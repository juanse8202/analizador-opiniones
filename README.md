# Analizador de opiniones — NLP y Machine Learning

Proyecto académico individual para clasificar **reseñas de productos en español** como **desfavorables, intermedias o favorables**, utilizando procesamiento de lenguaje natural y comparando Multinomial Naive Bayes y Regresión Logística. Las etiquetas de referencia se obtienen agrupando las valoraciones originales del dataset.

El proyecto sigue la organización del clasificador de tickets: preparación de datos, análisis exploratorio, procesamiento NLP, entrenamiento, evaluación, API REST y frontend. La salida prevista será la valoración global estimada del comentario, las probabilidades de las tres clases y una visualización del procesamiento aplicado.

## Estado actual

**Paso 3 completado: fuente, agrupación de valoraciones, preparación y control de similitud entre particiones.**

La preparación se implementó en `feat/dataset` y su auditoría adicional en `feat/auditoria-dataset`. El siguiente paso es `feat/eda`.

El alcance y el entorno están configurados. Se dispone de Amazon Reviews Multi en español: 200.000 registros originales de entrenamiento, 5.000 de validación y 5.000 de prueba. La opción 1 adoptada agrupa sus cinco niveles en tres categorías de valoración.

El descargador verifica los originales y el preparador genera CSV conservando texto, identificador, etiqueta original y partición. Después se apartan coincidencias entre particiones con Jaccard de trigramas ≥ 0,85. Se registran exclusiones y huellas. Este control léxico puede omitir paráfrasis; no demuestra equivalencia semántica.

El procesamiento NLP para modelado, los clasificadores, la API, la interfaz y el script de inicio corresponden a etapas posteriores. Todavía no hay modelos entrenados ni métricas de clasificación.

## Finalidad y dominio

El dominio inicial comprende reseñas de productos y experiencias de compra, entrega y atención asociadas. La unidad de análisis es el texto completo de cada reseña en español.

El objetivo específico es **estimar la categoría de valoración que acompaña al texto**. La supervisión procede de la escala original del cliente; los comentarios no han recibido una nueva anotación humana de sentimientos.

El alcance general de clasificación de opiniones se mantiene, con una definición operacional basada en valoraciones. Las futuras métricas medirán acuerdo con esas valoraciones agrupadas y no demostrarán, por sí solas, reconocimiento de todos los matices emocionales del texto.

## Etiquetas y criterios de clasificación

La política reproducible está en [label_mapping.json](opiniones-nlp/data/label_mapping.json).

| `label` original | Etiqueta `valuation` | Definición |
|---|---|---|
| `0` o `1` | `desfavorable` | Agrupa los dos niveles inferiores de valoración. |
| `2` | `intermedia` | Representa el punto medio de la escala original. |
| `3` o `4` | `favorable` | Agrupa los dos niveles superiores de valoración. |

**Intermedia no significa ausencia de sentimiento.** Puede incluir satisfacción parcial, críticas moderadas u opiniones mixtas. Los registros se agrupan por su valoración original, aunque el contenido del texto pueda discrepar con ella.

### Opiniones mixtas y casos especiales

- Una reseña como «El producto funciona bien, pero la batería dura poco» puede tener distintas valoraciones según su autor. Durante la preparación conserva la categoría que corresponde a su etiqueta original; no se reetiqueta automáticamente por palabras del texto.
- La valoración original puede contener ruido o contradicciones con el comentario. Se revisarán muestras de entrenamiento y validación, documentando desacuerdos sin convertir las etiquetas derivadas en supuestas anotaciones humanas.
- Las negaciones, emojis y signos expresivos pueden aportar información para estimar la valoración. Su tratamiento se evaluará en la fase de NLP.
- Los saludos, entradas vacías, texto insuficiente y mensajes fuera del dominio requerirán validación y reglas de aclaración que se implementarán en fases posteriores.
- La detección automática de sarcasmo o ambigüedad no está implementada. Cualquier umbral de confianza requerirá experimentación.

La definición anterior de `positivo`, `negativo` y `neutral` se sustituye por esta política. La primera versión tampoco predice una categoría específica de sentimiento mixto ni clasifica aspectos por separado.

## Alcance funcional previsto

- Preparar y documentar datos de reseñas en español, con procedencia y valoración de referencia.
- Analizar distribución de clases, longitudes, términos frecuentes, faltantes y duplicados.
- Implementar y comparar alternativas de procesamiento de texto.
- Comparar Multinomial Naive Bayes y Regresión Logística, y justificar el modelo seleccionado.
- Guardar el pipeline completo para aplicar las mismas transformaciones en entrenamiento e inferencia.
- Ofrecer una API REST para predicciones, métricas y estado del modelo.
- Crear una interfaz para ingresar un comentario y mostrar valoración estimada, probabilidades y trazas de procesamiento.
- Verificar el comportamiento con pruebas y documentar errores y limitaciones.

Quedan fuera de esta versión el análisis por aspecto, la clasificación multilingüe, la integración con redes sociales y la respuesta automática a clientes. La carga masiva y el seguimiento histórico podrán considerarse posteriormente.

## Datos y preparación

La fuente es **Amazon Reviews Multi, subconjunto español (`es`), desde la copia pública de MTEB**. La revisión, procedencia y condiciones documentadas se describen en [Dataset](opiniones-nlp/docs/DATASET.md). Las huellas de los originales están en [amazon_source.json](opiniones-nlp/data/amazon_source.json).

La ficha original del corpus indica que su distribución fue retirada. Se registra explícitamente la copia utilizada y sus condiciones de investigación académica. Los originales, los CSV y los manifiestos generados permanecen locales y excluidos de Git.

Desde la raíz `analizador-opiniones`, usando el entorno del paso 2:

```powershell
& ./opiniones-nlp/.venv312/Scripts/python.exe opiniones-nlp/src/data/download_amazon.py
& ./opiniones-nlp/.venv312/Scripts/python.exe opiniones-nlp/src/data/download_amazon.py --offline
& ./opiniones-nlp/.venv312/Scripts/python.exe opiniones-nlp/src/data/prepare_dataset.py
& ./opiniones-nlp/.venv312/Scripts/python.exe opiniones-nlp/src/data/audit_similarity.py
```

El preparador lee los originales de `opiniones-nlp/data/raw/amazon_reviews_multi/es/` y genera `train.csv`, `validation.csv`, `test.csv`, `exclusions.csv` y `manifest.json` en `opiniones-nlp/data/processed/amazon_three_levels/`.

El auditor lee esa primera preparación y produce una nueva versión en `opiniones-nlp/data/processed/amazon_three_levels_clean/`, con los tres CSV, `matches.csv` (identificadores y similitudes, sin textos) y su propio manifiesto. **El EDA y los futuros modelos deben usar esta versión final.** Los resultados y el método se documentan en [Auditoría de similitudes](opiniones-nlp/docs/AUDITORIA_SIMILITUD.md).

El directorio de salida debe ser nuevo. Para repetir la preparación conservando resultados previos:

```powershell
& ./opiniones-nlp/.venv312/Scripts/python.exe opiniones-nlp/src/data/prepare_dataset.py --output-dir opiniones-nlp/data/processed/amazon_three_levels_v2
```

Esquema de los CSV:

| Campo | Contenido |
|---|---|
| `id` | Identificador original. |
| `text` | Texto original, conservado sin aplicar NLP. |
| `valuation` | `desfavorable`, `intermedia` o `favorable`. |
| `original_label` | Etiqueta original de `0` a `4`. |
| `source` | Repositorio de procedencia. |
| `language` | Subconjunto de origen: `es`. |
| `original_split` | Partición original: `train`, `validation` o `test`. |

Se mantienen las particiones de origen, sin mover filas entre ellas ni efectuar un nuevo muestreo. Para detectar duplicados se usa NFKC, minúsculas mediante `casefold` y espacios colapsados, conservando puntuación y negaciones. Esta normalización se utiliza únicamente para comparar textos.

La prioridad es `test → validation → train`, como criterio de protección de evaluación del proyecto de tickets. Las copias en particiones de menor prioridad se excluyen. Dentro de una partición se conserva la primera copia si las categorías agrupadas coinciden; si el grupo de la partición prioritaria contiene categorías contradictorias, se apartan todas sus copias. Las exclusiones se registran por identificador y motivo, sin mostrar textos de prueba.

El control adicional de [similarity_policy.json](opiniones-nlp/data/similarity_policy.json) utiliza Jaccard de conjuntos de trigramas de caracteres con umbral fijo 0,85 y prioridad `test → validation → train`, tomando el proyecto de tickets como referencia. Se excluyen las reseñas similares a filas conservadas de mayor prioridad, sin consultar sus valoraciones. El texto y las etiquetas de las filas conservadas no se modifican. El test de esta ejecución se conserva byte por byte.

**La similitud léxica no cubre todas las paráfrasis ni prueba ausencia de fuga semántica.** No se comparan similitudes dentro de una misma partición. La consistencia entre texto y valoración, el idioma real y la cobertura del dominio se revisarán con entrenamiento y validación durante el EDA; los textos de prueba siguen reservados.

## Pipeline NLP previsto

1. **Normalización:** evaluar mayúsculas, Unicode, tildes, números, URLs, puntuación y emojis, justificando su efecto sobre la predicción de valoración.
2. **Tokenización:** separar el texto en unidades y revisar casos frecuentes del dominio.
3. **Stop words:** evaluar su retirada conservando negaciones relevantes.
4. **Stemming y lematización:** implementar ambas alternativas y comparar sus resultados.
5. **N-gramas y TF-IDF:** comparar unigramas frente a unigramas y bigramas.
6. **Clasificación:** comparar los modelos con el mismo procedimiento de evaluación.

El pipeline seleccionado se guardará completo. La API reutilizará sus transformaciones y trazas para mantener el mismo procesamiento que en entrenamiento.

## Modelos y evaluación previstos

Se compararán **Multinomial Naive Bayes** y **Regresión Logística**, usando F1 macro, métricas por clase, matriz de confusión, exactitud y análisis de errores. Los resultados representarán acuerdo con la valoración original agrupada.

La selección de configuraciones utilizará entrenamiento y validación; TF-IDF se ajustará dentro de cada partición de entrenamiento de la validación cruzada. El test se mantiene reservado para evaluar el modelo fijado. Su lectura durante la preparación sirve para comprobar integridad y bloquear duplicados, sin evaluar predicciones ni mostrar sus textos.

La agrupación produce una distribución inicial aproximada de 40 % desfavorable, 20 % intermedia y 40 % favorable. El manifiesto preparado registra los conteos efectivos tras las exclusiones; no se equilibra artificialmente el test.

Una evaluación humana complementaria deberá documentar sus criterios y la diferencia entre juicio sobre el texto y valoración original del cliente. Las probabilidades se presentarán como estimaciones del clasificador; cualquier afirmación de calibración requerirá evaluación específica.

**No hay un modelo seleccionado ni métricas disponibles en esta etapa.** Los resultados del clasificador de tickets corresponden a otra tarea.

## Arquitectura y estructura

Tecnologías previstas: **Python y scikit-learn** para modelos; **NLTK y spaCy** para procesamiento en español; **FastAPI** para la API; y **React con Vite** para la interfaz. Las dependencias base conservan las versiones del proyecto de tickets. Se fijan en `opiniones-nlp/requirements.txt` y `opiniones-frontend/package.json`; `requirements-eda.txt` añade herramientas de análisis y notebooks. El backend referencia las dependencias de NLP mediante su propio `requirements.txt`.

```text
analizador-opiniones/
├── opiniones-nlp/
│   ├── data/
│   │   ├── raw/          # Datos originales
│   │   ├── processed/    # Datos preparados y particiones
│   │   ├── templates/    # Plantillas de captura y revisión
│   │   ├── amazon_source.json # Fuente, revisión y huellas de originales
│   │   ├── label_mapping.json # Regla de agrupación de valoraciones
│   │   └── similarity_policy.json # Regla fija de similitud entre particiones
│   ├── src/
│   │   ├── data/         # Preparación y validación
│   │   ├── nlp/          # Procesamiento compartido
│   │   └── models/       # Entrenamiento, comparación e inferencia
│   ├── artifacts/        # Pipelines e informes generados
│   ├── notebooks/        # Análisis exploratorio
│   ├── docs/             # Documentación técnica y de datos
│   ├── tests/            # Pruebas de datos, NLP y modelos
│   ├── requirements.txt  # Dependencias base de NLP y API
│   └── requirements-eda.txt # Dependencias adicionales de EDA
├── opiniones-backend/
│   ├── api/              # API REST
│   ├── tests/            # Pruebas del backend
│   └── requirements.txt  # Referencia a las dependencias compartidas
├── opiniones-frontend/
│   ├── src/              # Interfaz web
│   ├── package.json      # Dependencias y comprobación inicial
│   ├── package-lock.json # Versiones resueltas
│   └── .npmrc            # Caché local de npm
├── .gitignore
├── .python-version
├── .node-version
└── README.md
```

Las carpetas existen como estructura inicial. Sus responsabilidades describen el código que se añadirá en las siguientes fases.

El flujo previsto será: **comentario → interfaz → API → procesamiento NLP y TF-IDF → clasificador → resultado y trazas en la interfaz**.

La API tendrá como contrato inicial propuesto:

| Endpoint | Responsabilidad prevista |
|---|---|
| `POST /predict` | Recibir un comentario y devolver valoración agrupada, probabilidades estimadas y trazas de procesamiento. |
| `GET /metrics` | Consultar las métricas del modelo y la comparación de configuraciones disponibles. |
| `GET /health` | Informar el estado de la API y la disponibilidad del modelo. |

## Instalación y ejecución

### Requisitos

- **Python 3.12**; versión verificada: **3.12.16**, registrada en `.python-version`.
- **Node.js 22 o superior**; versión verificada: **24.19.0**, registrada en `.node-version`.
- **npm**; versión utilizada en la comprobación: **11.17.0**.

Los archivos de versión documentan el entorno; por sí solos no instalan Python ni Node.js. Ejecutar los comandos siguientes desde la raíz `analizador-opiniones`, usando PowerShell.

### Python: entorno propio y dependencias

Si Python 3.12 está registrado en el lanzador de Windows:

```powershell
py -3.12 -m venv opiniones-nlp/.venv312
```

En esta copia se preparó una instalación local de Python 3.12.16 en `.tools/python/`, porque el lanzador de Windows solo tenía registrada otra versión. Se puede crear de nuevo el entorno con esa instalación:

```powershell
& ./.tools/python/cpython-3.12.16-windows-x86_64-none/python.exe -m venv opiniones-nlp/.venv312
```

En otro equipo, instalar Python 3.12 o utilizar la ruta de su ejecutable para crear el entorno. La instalación local `.tools/` y el entorno `.venv312/` se excluyen de Git; cada equipo prepara los suyos.

Instalar las dependencias base y el modelo lingüístico de spaCy:

```powershell
& ./opiniones-nlp/.venv312/Scripts/python.exe -m pip install -r opiniones-backend/requirements.txt
& ./opiniones-nlp/.venv312/Scripts/python.exe -m pip install --no-deps https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.8.0/es_core_news_sm-3.8.0-py3-none-any.whl
```

El modelo `es_core_news_sm` permite el procesamiento lingüístico en español. El clasificador de valoraciones agrupadas se entrenará posteriormente con los datos del nuevo proyecto.

Para preparar también el análisis exploratorio y los notebooks:

```powershell
& ./opiniones-nlp/.venv312/Scripts/python.exe -m pip install -r opiniones-nlp/requirements-eda.txt
```

Este archivo incluye las dependencias base y añade versiones fijadas de pandas, matplotlib, ipykernel y sus dependencias. No es necesario activar el entorno para utilizar los comandos con su ruta completa.

### Frontend: instalación de dependencias

```powershell
Set-Location opiniones-frontend
npm.cmd ci
npm.cmd run check:deps
Set-Location ..
```

Se mantienen **React 19.1.0** y **Vite 6.3.5**, como en el proyecto de tickets. `package-lock.json` registra las dependencias resueltas y `.npmrc` ubica la caché en `.npm-cache/`, excluida de Git junto con `node_modules/`.

En este paso solo se comprueba que React, React DOM y Vite estén disponibles. Los archivos de la interfaz y sus comandos de desarrollo y compilación se incorporarán en `feat/interfaz`.

### Comprobación del entorno

Después de instalar las dependencias base, el modelo de spaCy y las herramientas de EDA:

```powershell
& ./opiniones-nlp/.venv312/Scripts/python.exe --version
& ./opiniones-nlp/.venv312/Scripts/python.exe -m pip check
& ./opiniones-nlp/.venv312/Scripts/python.exe -c "import sklearn, nltk, spacy, fastapi, uvicorn, pandas, matplotlib, ipykernel; spacy.load('es_core_news_sm'); print('Entorno Python disponible')"
```

Estas comprobaciones verifican dependencias y carga del modelo lingüístico. Las pruebas de preparación y similitud ya están disponibles:

```powershell
Set-Location opiniones-nlp
& ./.venv312/Scripts/python.exe -m pytest -q
Set-Location ..
```

Las pruebas de NLP, modelos y API se incorporarán con esas implementaciones. El entrenamiento y el arranque completo se documentarán en sus respectivas etapas.

## Desarrollo por etapas

| Paso | Rama | Resultado esperado |
|---|---|---|
| 1 | `docs/alcance` | Objetivo, dominio, etiquetas y criterios para casos especiales documentados. |
| 2 | `chore/entorno` | Dependencias, `.gitignore` e instalación reproducible. |
| 3 | `feat/dataset` + `feat/auditoria-dataset` | Fuente documentada, datos validados, particiones preparadas y control de similitud registrado. |
| 4 | `feat/eda` | Análisis exploratorio con resultados y conclusiones. |
| 5 | `feat/preprocesamiento` | Transformaciones NLP implementadas y verificadas. |
| 6 | `feat/modelos` | Modelos comparados y pipeline seleccionado con métricas. |
| 7 | `feat/api` | API de predicción, métricas y estado verificada. |
| 8 | `feat/interfaz` | Interfaz conectada a resultados y trazas reales. |
| 9 | `feat/evaluacion-final` | Evaluación independiente y análisis de errores del modelo fijado. |
| 10 | `chore/entrega` | Inicio integrado, documentación final y comprobación de reproducibilidad. |

Cada etapa se desarrollará en su rama con commits que describan avances reales y revisables. Se integrará en `main` mediante `git merge` o un Pull Request antes de crear la siguiente rama desde la versión actualizada.

## Limitaciones y pendientes

- Revisar el dominio y la correspondencia entre textos y valoraciones agrupadas.
- Analizar desbalance, ruido, similitudes y representatividad de los comentarios.
- Medir las dificultades con negaciones, sarcasmo, opiniones mixtas y mensajes fuera del dominio.
- Implementar validaciones y comprobar los casos en que corresponde solicitar aclaración.
- Mantener una evaluación independiente y explicar que las métricas miden valoraciones agrupadas.
- Completar cada fase antes de presentar sus funciones como disponibles.
