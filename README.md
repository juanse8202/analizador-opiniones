# Analizador de opiniones — NLP y Machine Learning

Proyecto académico individual para clasificar **opiniones de clientes escritas en español** como **positivas, negativas o neutrales**. El objetivo es implementar y explicar un pipeline de procesamiento de lenguaje natural, comparar modelos de clasificación e integrar el modelo seleccionado en una aplicación web.

El proyecto seguirá una organización similar a la del clasificador de tickets: preparación de datos, análisis exploratorio, procesamiento NLP, entrenamiento, evaluación, API REST y frontend. La salida prevista será el sentimiento estimado del comentario, las probabilidades de las tres clases y una visualización del procesamiento aplicado.

## Estado actual

**Paso 2: configuración del entorno y dependencias, en la rama `chore/entorno`.**

La definición del alcance está completada. El proyecto incorpora dependencias de Python y JavaScript, versiones de referencia, reglas de exclusión de Git e instrucciones de instalación. El entorno Python es propio de `opiniones-nlp`; backend y NLP comparten sus dependencias, siguiendo la organización del clasificador de tickets.

El dataset todavía no está seleccionado. El procesamiento, los modelos, la API, la interfaz y el script de inicio se implementarán en las siguientes etapas. Las carpetas de código conservan sus archivos `.gitkeep`.

La instalación de dependencias y las comprobaciones del entorno están verificadas en Windows con PowerShell. Las funciones de clasificación siguen siendo objetivos del proyecto: todavía no hay resultados de entrenamiento ni métricas de evaluación.

## Finalidad y dominio

El caso de uso inicial comprende **comentarios de clientes sobre productos y servicios de una empresa genérica**: satisfacción con un producto, calidad de atención, experiencia de compra o entrega. Este dominio orienta la primera versión; en la etapa de datos se concretará su cobertura según la fuente elegida y se documentará cualquier ajuste del alcance.

La unidad de análisis será un comentario completo en español. Se buscará reconocer la valoración expresada en el texto para facilitar la revisión de opiniones y ofrecer una demostración del funcionamiento del pipeline NLP.

La primera versión propondrá una sola etiqueta global por comentario. Los casos con valoraciones contradictorias, contexto insuficiente o significado incierto tendrán criterios de revisión específicos.

## Etiquetas y criterios de clasificación

| Etiqueta | Criterio operacional inicial | Ejemplo ilustrativo |
|---|---|---|
| `positivo` | Expresa una valoración favorable, satisfacción o recomendación. | «La atención fue excelente y resolvieron mi consulta rápidamente». |
| `negativo` | Expresa una valoración desfavorable, insatisfacción o crítica. | «El producto llegó roto y la atención fue pésima». |
| `neutral` | Describe información del dominio sin expresar una valoración favorable o desfavorable identificable. | «El paquete contiene dos unidades y un manual». |

Los ejemplos de este README son **ilustrativos**; no constituyen un dataset ni resultados del modelo. Las etiquetas se asignarán considerando el significado del comentario completo. Una palabra aislada, la ausencia de palabras positivas o una puntuación de tres estrellas no determinan automáticamente una etiqueta neutral.

### Opiniones mixtas y casos especiales

| Caso | Tratamiento previsto |
|---|---|
| «La comida estuvo excelente, pero la atención fue pésima». | Registrar como opinión mixta y apartar para revisión; no convertir automáticamente a `neutral`. |
| «El empaque es sencillo, pero el producto funciona muy bien; lo recomiendo». | Permitir `positivo` cuando la conclusión global favorable sea clara y se confirme en la revisión. |
| «No estoy satisfecho con el servicio». | Considerar la negación y el sentido de insatisfacción; conservar esta información durante el procesamiento. |
| «Hola», texto vacío, solo espacios o caracteres sin contenido interpretable. | Validar la entrada y solicitar un comentario suficiente, sin forzar una de las tres clases. |
| Sarcasmo, ironía, referencias incompletas o texto fuera del dominio. | Marcar para revisión cuando no pueda establecerse una interpretación consistente. |
| Emojis o signos expresivos. | Evaluar cómo conservar su información antes de definir la normalización. |

En la preparación del dataset, se asignará una de las tres etiquetas únicamente cuando exista una valoración global clara o una descripción neutral identificable. Los casos mixtos sin orientación dominante se conservarán aparte, con el motivo de revisión, y se contabilizarán en el informe de datos. Esta separación limita la cobertura del clasificador y deberá reflejarse en la evaluación.

La primera versión no incorpora una cuarta clase de sentimiento mixto. La detección automática de ambigüedad y las reglas para solicitar aclaración deberán implementarse y evaluarse antes de afirmarlas como comportamiento disponible. No se fijará un umbral de confianza sin experimentación.

## Alcance funcional previsto

- Preparar un dataset de opiniones en español con etiquetas y procedencia documentadas.
- Explorar la distribución de clases, las longitudes, los términos frecuentes, los faltantes y los duplicados.
- Implementar y comparar alternativas de procesamiento de texto.
- Entrenar Multinomial Naive Bayes y Regresión Logística, y justificar la selección del modelo.
- Guardar el pipeline completo para aplicar las mismas transformaciones en entrenamiento e inferencia.
- Ofrecer una API REST para consultar predicciones, métricas y estado del modelo.
- Crear una interfaz web para ingresar un comentario y visualizar el sentimiento, las probabilidades estimadas y las trazas de procesamiento.
- Verificar el comportamiento con pruebas y documentar resultados, errores y limitaciones.

Quedan fuera de esta primera versión el análisis de sentimiento por aspecto —por ejemplo, comida y atención por separado—, el análisis multilingüe, la integración con redes sociales y la respuesta automática a clientes. La carga masiva de comentarios y el seguimiento histórico podrán considerarse como ampliaciones posteriores.

## Datos y protocolo inicial

La selección del dataset se realizará en la rama `feat/dataset`. Antes de incorporarlo se documentarán su autor, fuente, licencia, idioma, versión o fecha de obtención, campos disponibles y compatibilidad con las tres etiquetas.

Se propone el siguiente esquema mínimo para los datos preparados:

| Campo | Contenido |
|---|---|
| `text` | Comentario original que se analizará. |
| `sentiment` | Una de las etiquetas `positivo`, `negativo` o `neutral`. |
| `source` | Procedencia del comentario, vinculada a la documentación de la fuente. |

El esquema definitivo, los identificadores y los metadatos de revisión se concretarán al conocer el dataset. Los ejemplos públicos, los escritos por personas para el proyecto y los generados con asistencia de IA mantendrán una procedencia diferenciada. Las etiquetas y los casos dudosos se revisarán y se registrarán las decisiones tomadas.

La preparación deberá separar entrenamiento, validación y prueba antes de ajustar el vectorizador o seleccionar modelos. Se controlarán duplicados y similitudes entre particiones; sus limitaciones deberán quedar documentadas. Si existen grupos relacionados, como variantes del mismo comentario, se evaluará una separación por grupo para evitar contaminación entre conjuntos.

La evaluación humana complementaria usará comentarios independientes. Si sus resultados sirven para modificar el modelo, se necesitarán otros comentarios reservados para la evaluación posterior. El conjunto de prueba final se mantendrá apartado de la selección de configuraciones.

## Pipeline NLP previsto

1. **Normalización:** definir el tratamiento de mayúsculas, Unicode, tildes, números, URLs, puntuación y emojis, justificando el efecto sobre el sentimiento.
2. **Tokenización:** separar el texto en unidades y revisar casos frecuentes del dominio.
3. **Stop words:** evaluar su retirada conservando negaciones relevantes, como `no`, `nunca` y `sin`.
4. **Stemming y lematización:** implementar ambas alternativas y comparar su efecto.
5. **N-gramas y TF-IDF:** comparar unigramas frente a unigramas y bigramas, y registrar los parámetros del vectorizador.
6. **Clasificación:** entrenar y comparar los modelos utilizando el mismo procedimiento de evaluación.

El pipeline seleccionado se serializará junto con el procesamiento y el clasificador. El backend reutilizará ese pipeline y sus trazas para evitar diferencias entre entrenamiento, predicción y visualización.

## Modelos y evaluación previstos

Se compararán **Multinomial Naive Bayes** y **Regresión Logística**. La selección considerará F1 macro, resultados por clase, matriz de confusión y análisis de errores. También se informará la exactitud, diferenciándola del F1 macro.

La comparación utilizará validación cruzada sobre entrenamiento, con semilla registrada. El número de particiones y los hiperparámetros se definirán cuando se conozcan el volumen y la distribución de los datos. TF-IDF se ajustará dentro de cada partición de entrenamiento de la validación cruzada.

Se documentarán la configuración elegida, las versiones utilizadas, la procedencia de los datos y la correspondencia entre el modelo guardado y sus informes. Las probabilidades se presentarán como estimaciones del clasificador; cualquier afirmación sobre su calibración requerirá una evaluación específica.

**No hay un modelo seleccionado ni métricas disponibles en esta etapa.** Los resultados del clasificador de tickets corresponden a otra tarea y no describen el rendimiento del analizador de opiniones.

## Arquitectura y estructura

Tecnologías previstas: **Python y scikit-learn** para modelos; **NLTK y spaCy** para procesamiento en español; **FastAPI** para la API; y **React con Vite** para la interfaz. Las dependencias base conservan las versiones del proyecto de tickets. Se fijan en `opiniones-nlp/requirements.txt` y `opiniones-frontend/package.json`; `requirements-eda.txt` añade herramientas de análisis y notebooks. El backend referencia las dependencias de NLP mediante su propio `requirements.txt`.

```text
analizador-opiniones/
├── opiniones-nlp/
│   ├── data/
│   │   ├── raw/          # Datos originales
│   │   ├── processed/    # Datos preparados y particiones
│   │   └── templates/    # Plantillas de captura y revisión
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
| `POST /predict` | Recibir un comentario y devolver sentimiento, probabilidades estimadas y trazas de procesamiento. |
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

El modelo `es_core_news_sm` permite el procesamiento lingüístico en español. El clasificador de sentimientos se entrenará posteriormente con los datos del nuevo proyecto.

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

Estas comprobaciones verifican dependencias y carga del modelo lingüístico. Las pruebas de preparación de datos, NLP, modelos y API se añadirán junto con esas implementaciones. El entrenamiento y el arranque completo de la aplicación se documentarán en sus respectivas etapas.

## Desarrollo por etapas

| Paso | Rama | Resultado esperado |
|---|---|---|
| 1 | `docs/alcance` | Objetivo, dominio, etiquetas y criterios para casos especiales documentados. |
| 2 | `chore/entorno` | Dependencias, `.gitignore` e instalación reproducible. |
| 3 | `feat/dataset` | Fuente documentada, datos validados y particiones preparadas. |
| 4 | `feat/eda` | Análisis exploratorio con resultados y conclusiones. |
| 5 | `feat/preprocesamiento` | Transformaciones NLP implementadas y verificadas. |
| 6 | `feat/modelos` | Modelos comparados y pipeline seleccionado con métricas. |
| 7 | `feat/api` | API de predicción, métricas y estado verificada. |
| 8 | `feat/interfaz` | Interfaz conectada a resultados y trazas reales. |
| 9 | `feat/evaluacion-final` | Evaluación independiente y análisis de errores del modelo fijado. |
| 10 | `chore/entrega` | Inicio integrado, documentación final y comprobación de reproducibilidad. |

Cada etapa se desarrollará en su rama con commits que describan avances reales y revisables. Se integrará en `main` mediante `git merge` o un Pull Request antes de crear la siguiente rama desde la versión actualizada.

## Limitaciones y pendientes

- Concretar el dominio según el dataset seleccionado y revisar la consistencia de sus etiquetas.
- Analizar desbalance, ruido, similitudes y representatividad de los comentarios.
- Medir las dificultades con negaciones, sarcasmo, opiniones mixtas y mensajes fuera del dominio.
- Implementar validaciones y comprobar los casos en que corresponde solicitar aclaración.
- Mantener una evaluación independiente y explicar el alcance real de las métricas obtenidas.
- Completar cada fase antes de presentar sus funciones como disponibles.
