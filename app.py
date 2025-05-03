import os
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuración de la página
st.set_page_config(
    page_title="Aplicación de Pronóstico Presupuestario",
    page_icon="💰",
    layout="wide"
)

# Cargar imagen
def load_image(image_path):
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        st.warning(f"Imagen no encontrada en la ruta: {image_path}")
        return None

# Cargar modelos con manejo de errores
@st.cache_resource
def load_models():
    try:
        lgbm_model = joblib.load('models/best_lgbm_model.pkl')
        xgb_model = joblib.load('models/best_xgb_model.pkl')
        stacking_model = joblib.load('models/best_stacking_model.pkl')
        return lgbm_model, xgb_model, stacking_model
    except FileNotFoundError as e:
        st.error(f"Archivo de modelo no encontrado: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"Error al cargar los modelos: {str(e)}")
        return None, None, None

# Cargar modelos al inicio
lgbm_model, xgb_model, stacking_model = load_models()

# Pestañas de navegación
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Sobre Mí", 
    "Sobre el Proyecto",
    "Información del Dataset",
    "¿Qué es esta herramienta?",
    "Cómo Funciona",
    "Características Principales",
    "Análisis del Modelo"
])

# Pestaña 1: Sobre Mí
with tab1:
    st.title("Sobre Mí")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        my_image = load_image("assets/Bo-Kolstrup.png")
        if my_image:
            st.image(my_image, width=200)
    
    with col2:
        st.markdown("""
        **Apasionado por aplicar ciencia de datos y aprendizaje automático para resolver desafíos empresariales reales.**
        
        Este proyecto forma parte de mi portafolio y demuestra mi capacidad para:
        - Desarrollar modelos predictivos robustos
        - Realizar análisis exploratorios de datos complejos
        - Crear aplicaciones interactivas para visualización de resultados
        - Implementar soluciones de machine learning de extremo a extremo
        """)

# Pestaña 2: Sobre el Proyecto
with tab2:
    st.title("Sobre el Proyecto")
    st.markdown("""
    ### Pronóstico de Presupuesto Empresarial con Machine Learning
    
    Este proyecto fue creado para mi portafolio y utiliza datos simulados de una empresa del sector de gestión de residuos.
    
    **Objetivos del Proyecto:**
    - Desarrollar un sistema inteligente de pronóstico presupuestario
    - Identificar los principales impulsores financieros
    - Detectar patrones y anomalías en datos históricos
    - Construir modelos predictivos explicables
    - Integrar resultados en herramientas de visualización
    
    **Tecnologías Utilizadas:**
    - Python (Pandas, NumPy, Scikit-learn)
    - Machine Learning (XGBoost, LightGBM, Random Forest)
    - Visualización (Matplotlib, Seaborn, Plotly)
    - Streamlit para la aplicación interactiva
    """)

# Pestaña 3: Información del Dataset
with tab3:
    st.title("Información del Dataset")
    st.markdown("""
    ### Datos Simulados para Gestión de Residuos
    
    **1. Dataset de Costos (costs_dataset.csv)**
    - 50,000 filas y 25 columnas
    - Incluye información sobre:
        - Tipos de servicio
        - Duración del servicio
        - Costos (mano de obra, equipo, transporte)
        - Factores operacionales (clima, disponibilidad de personal)
    
    **2. Dataset de Ingresos (earnings_dataset.csv)**
    - 50,000 filas y 13 columnas
    - Contiene datos sobre:
        - Facturación y márgenes de ganancia
        - Métodos de pago
        - Estado de pagos
        - Descuentos aplicados
    
    **Procesamiento Realizado:**
    - Limpieza de datos (valores faltantes, duplicados)
    - Ingeniería de características
    - Codificación de variables categóricas
    - Normalización de datos
    """)

# Pestaña 4: Explicación de la Herramienta
with tab4:
    st.title("¿Qué es esta herramienta?")
    st.markdown("""
    ### Sistema Inteligente de Pronóstico Presupuestario
    
    **Capacidades:**
    - Predice costos totales y utilidad neta
    - Identifica factores clave que impactan resultados financieros
    - Detecta patrones y anomalías en datos operacionales
    - Genera visualizaciones interactivas
    
    **Beneficios:**
    - Automatiza pronósticos financieros
    - Reduce dependencia de métodos manuales
    - Proporciona insights accionables
    - Mejora precisión en planificación presupuestaria
    """)

# Pestaña 5: Cómo Funciona
with tab5:
    st.title("Cómo Funciona")
    st.markdown("""
    ### Arquitectura de la Solución
    
    1. **Ingesta de Datos**
       - Carga y fusión de datasets
       - Limpieza y preprocesamiento
    
    2. **Análisis Exploratorio**
       - Visualización de patrones
       - Detección de valores atípicos
       - Ingeniería de características
    
    3. **Modelado Predictivo**
       - Entrenamiento de algoritmos (XGBoost, LightGBM, Random Forest)
       - Optimización de hiperparámetros
       - Validación cruzada
    
    4. **Implementación**
       - Dashboard interactivo
       - Exportación de resultados
    """)

# Pestaña 6: Características Principales
with tab6:
    st.title("Características Principales")
    st.markdown("""
    ### Funcionalidades Clave
    
    **1. Pronósticos Precisos**
    - MAE < 5% del valor real
    - R² > 0.99
    
    **2. Visualización Interactiva**
    - Tendencias temporales
    - Análisis de componentes principales
    - Mapas de calor de correlación
    
    **3. Explicabilidad del Modelo**
    - Gráficos SHAP
    - Importancia de características
    - Análisis de valores atípicos
    
    **4. Integración Empresarial**
    - Exportación a CSV/Excel
    - API REST
    - Compatibilidad con Power BI
    """)

# Pestaña 7: Análisis del Modelo
with tab7:
    st.title("Análisis del Modelo")
    
    if None in [lgbm_model, xgb_model, stacking_model]:
        st.error("Algunos modelos no se cargaron correctamente. Verifique los archivos en el directorio 'models'.")
    else:
        st.success("¡Modelos cargados exitosamente!")
        
        model_tab1, model_tab2, model_tab3 = st.tabs(["Rendimiento", "Importancia de Características", "Análisis SHAP"])
        
        with model_tab1:
            st.header("Comparación de Rendimiento del Modelo")
            
            # Métricas de ejemplo - reemplazar con métricas reales
            metrics_data = {
                'Modelo': ['XGBoost', 'LightGBM', 'Stacking'],
                'MAE': [63.56, 68.69, 60.12],
                'RMSE': [93.02, 97.51, 88.45],
                'R²': [0.996, 0.995, 0.997]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df.style.format({
                'MAE': '{:.2f}',
                'RMSE': '{:.2f}',
                'R²': '{:.3f}'
            }))
            
            # Visualización
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_df.set_index('Modelo').plot(kind='bar', ax=ax)
            plt.title('Comparación de Rendimiento del Modelo')
            plt.ylabel('Puntuación')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with model_tab2:
            st.header("Importancia de Características")
            
            # Importancia de características de ejemplo - reemplazar con datos reales
            if hasattr(xgb_model, 'feature_importances_'):
                # Usando nombres de características ficticios - reemplazar con características reales
                feature_names = [f'Característica_{i}' for i in range(1, xgb_model.n_features_in_ + 1)]
                importance_data = {
                    'Característica': feature_names,
                    'XGBoost': xgb_model.feature_importances_,
                    'LightGBM': lgbm_model.feature_importances_
                }
                
                importance_df = pd.DataFrame(importance_data)
                st.dataframe(importance_df)
                
                # Gráfico de comparación
                fig, ax = plt.subplots(figsize=(12, 6))
                importance_df.set_index('Característica').plot(kind='barh', ax=ax)
                plt.title('Comparación de Importancia de Características')
                st.pyplot(fig)
            else:
                st.warning("Importancia de características no disponible para estos modelos")
        
        with model_tab3:
            st.header("Análisis SHAP")
            
            try:
                # Obtener el número de características que espera el modelo
                n_features = xgb_model.n_features_in_
                feature_names = [f'Característica_{i}' for i in range(1, n_features + 1)]
                
                # Crear datos de fondo representativos (100 muestras)
                # Nota: En una aplicación real, deberías usar estadísticas de tus datos de entrenamiento reales
                background = np.random.randn(100, n_features) * 0.5 + 0.5  # Distribución similar a normal
                
                # Crear explicador
                explainer = shap.Explainer(xgb_model, background)
                
                # Calcular valores SHAP para una muestra (50 instancias)
                sample_data = background[:50]  # Usar primeras 50 muestras
                shap_values = explainer(sample_data)
                
                st.markdown("""
                ### Importancia de Características SHAP
                Muestra qué características son más importantes para las predicciones del modelo
                """)
                
                # Gráfico 1: Gráfico de barras
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                shap.plots.bar(shap_values, max_display=min(15, n_features))
                st.pyplot(fig1)
                
                st.markdown("""
                ### Gráfico Resumen SHAP
                Muestra la distribución de valores SHAP para cada característica
                """)
                
                # Gráfico 2: Gráfico resumen
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, sample_data, feature_names=feature_names)
                st.pyplot(fig2)
                
            except Exception as e:
                st.error(f"No se pudo generar el gráfico SHAP: {str(e)}")
                st.error("Para mejores resultados:")
                st.error("1. Reemplaza los datos de fondo generados con tus datos de entrenamiento reales")
                st.error("2. Asegúrate de que los nombres de las características coincidan con las expectativas del modelo")
                st.error(f"Detalles técnicos: {str(e)}")

# Pie de página
st.markdown("---")
st.markdown("""
**Portafolio de Ciencia de Datos** - Desarrollado por Bo Kolstrup  
[GitHub](https://github.com/bokolstrup) | [LinkedIn](https://linkedin.com/in/bokolstrup/)
""")