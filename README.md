# Budget Forecasting Application with Machine Learning and Power BI Integration

**Prueba el modelo [aquí](https://aplicacion-inteligente-para-el-pronostico-de-presupuesto-empre.streamlit.app/):**

## Introducción
Este proyecto desarrolla una aplicación inteligente de pronóstico presupuestario para empresas, integrando técnicas avanzadas de machine learning, análisis exploratorio de datos y modelado predictivo de series temporales. Al limpiar, transformar y modelar dos grandes conjuntos de datos financieros (costos e ingresos), la solución estima con precisión los costos totales y la utilidad neta, identificando patrones, anomalías y variables clave que impactan el presupuesto.

El sistema incluye una API opcional y visualización automatizada a través de Power BI, permitiendo una integración perfecta en los flujos de trabajo empresariales. El objetivo final es proporcionar información basada en datos para la planificación financiera y la toma de decisiones.

## Descripción General del Proyecto

### Objetivos
Desarrollar un sistema inteligente y automatizado de pronóstico presupuestario que utiliza técnicas de machine learning y análisis de datos para:

- Predecir con precisión ingresos, costos y ganancias de la empresa
- Identificar impulsores financieros clave
- Detectar patrones y anomalías en datos históricos
- Construir modelos predictivos explicables para la toma de decisiones
- Integrar resultados con herramientas de visualización como Power BI
- Automatizar procesos de análisis y reportes financieros

### Características Principales
- Limpieza y fusión de conjuntos de datos financieros
- Ingeniería avanzada de características
- Múltiples modelos de machine learning (Random Forest, XGBoost, LightGBM, Prophet)
- Evaluación y validación de modelos
- API REST con Streamlit
- Integración con Power BI
- Generación automatizada de reportes

## Desarrollo del Modelo

### Pipeline de Procesamiento de Datos
**Carga de Datos y Evaluación Inicial:**
- Se cargaron y examinaron dos conjuntos de datos: costos (50,000 filas × 25 columnas) e ingresos (50,000 filas × 13 columnas)
- Identificación y conversión de columnas de fecha/hora
- Verificación de valores faltantes y duplicados

**Ingeniería de Características:**
- Creación de métricas financieras (márgenes de ganancia, porcentajes de desglose de costos)
- Extracción de características temporales (año, mes, día, día de la semana, hora)
- Generación de términos de interacción
- Aplicación de binning a variables continuas
- Codificación de variables categóricas

**Selección de Características:**
- Eliminación de características con baja varianza
- Eliminación de características altamente correlacionadas
- Reducción de dimensionalidad con PCA
- Selección de características principales mediante importancia de Random Forest

## Entrenamiento y Evaluación de Modelos

**Modelos Base:**
- SARIMA (series temporales univariadas)
- Prophet (pronóstico de series temporales)

**Modelos Avanzados:**
- Random Forest
- XGBoost
- LightGBM

**Métodos de Ensamblaje:**
- Stacking Regressor (XGBoost + LightGBM)
- Voting Regressor (XGBoost + LightGBM)

### Métricas de Rendimiento

| Modelo          | MAE (Costo Total) | RMSE (Costo Total) | R² (Costo Total) | MAE (Ganancia Neta) | RMSE (Ganancia Neta) | R² (Ganancia Neta) |
|-----------------|-------------------|--------------------|------------------|---------------------|----------------------|--------------------|
| Random Forest   | 107.26            | 225.47             | 0.996            | 100.28              | 156.85               | 0.987              |
| XGBoost         | 208.13            | 276.92             | 0.993            | 83.90               | 114.53               | 0.993              |
| LightGBM        | 219.30            | 299.68             | 0.992            | 85.14               | 121.98               | 0.992              |
| Stacking        | -                 | 92.52              | -                | -                   | -                    | -                  |
| Voting          | -                 | 88.03              | -                | -                   | -                    | -                  |

## Explorador de Datos

### Visualizaciones Clave
**Análisis Temporal:**
- Distribución de servicios por año/mes/día de la semana
- Patrones de retraso en pagos

**Métricas Financieras:**
- Porcentajes de desglose de costos
- Distribuciones de márgenes de ganancia
- Relaciones entre ingresos y ganancia neta

**Análisis de Correlación:**
- Mapas de calor que muestran relaciones entre variables clave
- Visualizaciones de valores SHAP

**Detección de Anomalías:**
- Visualización de valores atípicos usando diagramas de caja
- Resultados de Isolation Forest

## Herramienta de Pronóstico y Simulación

### Características
**Interfaz Interactiva (Streamlit):**
- Deslizadores para ajuste de parámetros
- Simulación de escenarios
- Actualizaciones de pronóstico en tiempo real

**Integración con Power BI:**
- Actualización automatizada de datos
- Paneles interactivos
- Capacidades de exploración detallada

**Endpoints de API:**
- Interfaz RESTful para integración
- Soporte para predicciones por lotes

## Resultados y Hallazgos

### Características Predictivas Principales
- profit_margin_percentage (más importante para ambos objetivos)
- transportation_cost_percent
- cost_anomaly flags
- Características temporales (mes, día de la semana)

### Conclusiones del Modelado
- Los modelos basados en árboles superaron a los enfoques de series temporales
- XGBoost mostró el mejor equilibrio entre rendimiento y velocidad
- Los métodos de ensamblaje proporcionaron mejoras marginales

### Implicaciones Empresariales
- Identificación de impulsores clave de costos y oportunidades de ganancia
- Habilitación de pronósticos precisos a 30-90 días
- Detección de eventos de servicio anómalos

## Conclusión
Este proyecto entregó con éxito una solución integral de pronóstico presupuestario que:

- Automatiza el análisis financiero realizado previamente de forma manual
- Mejora la precisión con modelos de machine learning (R² > 0.99)
- Proporciona información accionable mediante IA explicable
- Se integra perfectamente con herramientas de BI existentes

El sistema es particularmente valioso para:
- Equipos de planificación y análisis financiero
- Gerentes de operaciones
- Liderazgo ejecutivo

**Mejoras Futuras Podrían Incluir:**
- Pipelines de datos en tiempo real
- Modelado de escenarios "what-if"
- Alertas automatizadas para anomalías

**Prueba el modelo [aquí](https://aplicacion-inteligente-para-el-pronostico-de-presupuesto-empre.streamlit.app/):**
