# 3_📈_Analyze.py
"""
Panel de Análisis Avanzado

Esta aplicación Streamlit proporciona herramientas analíticas interactivas para explorar
datos financieros, incluyendo:
- Análisis de impulsores de costos
- Detección de anomalías
- Capacidades de agrupamiento

Características:
- Visualizaciones interactivas con Plotly
- Controles dinámicos de parámetros
- Descomposición de series temporales
- Almacenamiento en caché de datos para rendimiento
"""

# Importar librerías requeridas con categorías agrupadas
# Procesamiento central de datos
import pandas as pd
import numpy as np

# Visualización
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Análisis estadístico
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Marco de la aplicación
import streamlit as st

# Configurar ajustes predeterminados de página
st.set_page_config(
    page_title="Panel de Análisis Avanzado",
    page_icon="📈",
    layout="wide",
    menu_items={
        'Get help': 'https://github.com/Bokols',
        'About': "Herramienta de análisis estadístico y machine learning avanzado"
    }
)

# Estilos CSS personalizados para elementos de UI consistentes
st.markdown("""
<style>
    /* Estilo para tarjetas de métricas */
    .stMetric {
        border-radius: 8px;
        padding: 15px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Estilo para contenedores de gráficos */
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Mejorar espaciado para pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CARGA DE DATOS ====================
@st.cache_data(show_spinner="Cargando conjunto de datos analíticos...")
def load_data():
    """
    Genera y almacena en caché datos sintéticos para fines de demostración.
    
    Returns:
        pd.DataFrame: Contiene datos generados con:
            - Datos temporales (fecha)
            - Características categóricas (tipo_servicio, región)
            - Métricas de costos numéricos
            - Métricas financieras derivadas (ingresos, ganancia)
    
    Nota:
        Usa semilla aleatoria de numpy para resultados reproducibles.
        Los datos incluyen relaciones realistas entre variables.
    """
    try:
        np.random.seed(42)  # Para resultados reproducibles
        size = 200  # Número de registros a generar
        
        # Crear dataframe con características principales
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=size),
            'service_type': np.random.choice(['Recolección', 'Disposición', 'Reciclaje', 'Peligroso'], size),
            'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], size),
            'labor_cost': np.random.uniform(200, 1500, size),
            'transport_cost': np.random.uniform(100, 800, size),
            'equipment_cost': np.random.uniform(100, 1200, size),
            'waste_volume': np.random.uniform(1, 20, size),
            'service_hours': np.random.uniform(1, 8, size),
            'hazardous': np.random.choice([True, False], size, p=[0.2, 0.8]),
            'fuel_price': np.random.uniform(2.5, 4.5, size)
        })
        
        # Calcular métricas financieras derivadas
        df['total_cost'] = (df['labor_cost'] + 
                          df['transport_cost'] + 
                          df['equipment_cost'] + 
                          (df['hazardous'] * 500))  # Prima por peligrosidad
        
        df['revenue'] = df['total_cost'] * np.random.uniform(1.4, 1.6, size)
        df['profit'] = df['revenue'] - df['total_cost']
        df['profit_margin'] = (df['profit'] / df['revenue']) * 100
        
        return df
    
    except Exception as e:
        st.error(f"Error al generar datos de muestra: {str(e)}")
        return pd.DataFrame()

# Cargar los datos
df = load_data()

# ==================== CONTROLES DE BARRA LATERAL ====================
with st.sidebar:
    st.header("⚙️ Parámetros de Análisis")
    
    with st.expander("Configuración Principal", expanded=True):
        # Selección de variable objetivo
        target_variable = st.selectbox(
            "Variable Objetivo",
            options=['profit', 'profit_margin', 'total_cost', 'revenue'],
            index=0,
            help="Selecciona la métrica financiera que deseas analizar"
        )
        
        # Control de ventana para detección de anomalías
        analysis_days = st.slider(
            "Ventana de Detección de Anomalías",
            min_value=7,
            max_value=30,
            value=14,
            help="Número de días a usar para calcular estadísticas móviles en detección de anomalías"
        )
    
    with st.expander("Opciones Avanzadas", expanded=False):
        # Alternar agrupamiento
        clustering_enabled = st.checkbox(
            "Habilitar Análisis de Agrupamiento",
            value=False,
            help="Habilita para realizar segmentación de clientes basada en características seleccionadas"
        )
        
        # Alternar descomposición de series temporales
        time_decomposition = st.checkbox(
            "Habilitar Descomposición de Series Temporales",
            value=False,
            help="Descompone datos de series temporales en componentes de tendencia, estacionalidad y residuos"
        )

# ==================== CONTENIDO PRINCIPAL ====================
st.title("📈 Análisis Financiero Avanzado")

# Sección "Acerca de" expandible
with st.expander("ℹ️ Acerca de esta herramienta", expanded=False):
    st.markdown("""
    **Bienvenido al Panel de Análisis Avanzado**  
    Esta herramienta interactiva proporciona información profunda sobre datos financieros.
    
    **Características Clave:**
    - **Análisis de Impulsores**: Identifica impulsores clave de costos y sus relaciones
    - **Detección de Anomalías**: Detecta patrones inusuales en métricas financieras
    - **Agrupamiento**: Descubre agrupaciones naturales en tus datos
    
    **Cómo usar:**
    1. Selecciona tu métrica objetivo en la barra lateral
    2. Ajusta los parámetros de análisis según sea necesario
    3. Navega entre pestañas para explorar diferentes análisis
    4. Pasa el cursor sobre visualizaciones para información detallada
    """)

# Crear pestañas principales de análisis
tab1, tab2, tab3 = st.tabs(["📊 Análisis de Impulsores", "📉 Detección de Anomalías", "🔄 Agrupamiento"])

# ==================== PESTAÑA DE ANÁLISIS DE IMPULSORES ====================
with tab1:
    st.subheader("Importancia de Impulsores de Costo")
    st.markdown("""
    Analiza cómo diferentes factores operacionales se correlacionan con tu métrica financiera seleccionada.
    El gráfico de barras muestra coeficientes de correlación de Pearson entre cada impulsor y tu variable objetivo.
    """)
    
    # Calcular correlaciones con variable objetivo
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_variable].drop(target_variable)
    
    # Crear gráfico de correlación interactivo
    fig = px.bar(
        correlations.sort_values(),
        orientation='h',
        title=f"Correlación con {target_variable.replace('_', ' ').title()}",
        labels={'value': 'Coeficiente de Correlación', 'index': 'Variable'},
        color=correlations.sort_values(),
        color_continuous_scale='RdYlGn',
        range_color=[-1, 1]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Análisis de Dependencia Parcial")
    st.markdown("""
    Explora cómo los cambios en un impulsor específico afectan tu métrica objetivo.
    La línea muestra valores promedio, mientras que los puntos opcionales muestran observaciones individuales.
    """)
    
    # Crear controles para análisis de dependencia parcial
    col1, col2 = st.columns(2)
    
    with col1:
        driver = st.selectbox(
            "Seleccionar Impulsor para Analizar",
            options=['labor_cost', 'transport_cost', 'equipment_cost', 
                    'waste_volume', 'service_hours', 'fuel_price'],
            index=0,
            help="Elige qué factor operacional analizar"
        )
    
    with col2:
        show_individual = st.checkbox(
            "Mostrar Observaciones Individuales",
            value=False,
            help="Muestra puntos de datos individuales detrás de la línea de tendencia promedio"
        )
    
    # Crear bins para análisis (basados en cuantiles para distribución uniforme)
    bins = pd.qcut(df[driver], q=10, duplicates='drop')
    grouped = df.groupby(bins)[target_variable].mean().reset_index()
    grouped[driver] = grouped[driver].apply(lambda x: x.mid)
    
    # Crear gráfico de dependencia parcial
    fig = px.line(
        grouped,
        x=driver,
        y=target_variable,
        title=f"Impacto de {driver.replace('_', ' ').title()} en {target_variable.replace('_', ' ').title()}",
        markers=True
    )
    
    # Añadir puntos individuales si se solicita
    if show_individual:
        fig.add_trace(go.Scatter(
            x=df[driver],
            y=df[target_variable],
            mode='markers',
            name='Individual',
            marker=dict(opacity=0.3)
        ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Descomposición de series temporales si está habilitada
    if time_decomposition:
        st.subheader("Descomposición de Series Temporales")
        st.markdown("""
        Descompone tus datos de series temporales en:
        - **Tendencia**: Dirección a largo plazo
        - **Estacionalidad**: Patrones repetitivos
        - **Residuos**: Variación no explicada
        """)
        
        try:
            # Preparar datos de series temporales
            ts_df = df.set_index('date').resample('D')[target_variable].mean().ffill()
            
            # Realizar descomposición
            decomposition = seasonal_decompose(ts_df, model='additive', period=7)
            
            # Crear visualización
            fig = go.Figure()
            
            # Añadir componente de tendencia
            fig.add_trace(go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend,
                name='Tendencia',
                line=dict(color='blue')
            ))
            
            # Añadir componente estacional
            fig.add_trace(go.Scatter(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal,
                name='Estacionalidad',
                line=dict(color='green')
            ))
            
            # Añadir componente residual
            fig.add_trace(go.Scatter(
                x=decomposition.resid.index,
                y=decomposition.resid,
                name='Residuos',
                line=dict(color='red')
            ))
            
            # Configurar diseño
            fig.update_layout(
                title=f"Descomposición de Series Temporales de {target_variable.replace('_', ' ').title()}",
                xaxis_title="Fecha",
                yaxis_title="Valor",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"No se pudo realizar la descomposición de series temporales: {str(e)}")

# ==================== PESTAÑA DE DETECCIÓN DE ANOMALÍAS ====================
with tab2:
    st.subheader("Detección de Anomalías")
    st.markdown("""
    Identifica patrones inusuales en tus datos financieros usando estadísticas móviles.
    Las anomalías se detectan cuando los valores exceden desviaciones estándar especificadas del promedio móvil.
    """)
    
    # Calcular estadísticas móviles
    rolling_df = df.set_index('date')[target_variable].rolling(f'{analysis_days}D')
    df['rolling_mean'] = rolling_df.mean().values
    df['rolling_std'] = rolling_df.std().values
    df['z_score'] = (df[target_variable] - df['rolling_mean']) / df['rolling_std']
    
    # Control de umbral para detección de anomalías
    threshold = st.slider(
        "Umbral de Anomalía (Z-Score)",
        min_value=1.0,
        max_value=5.0,
        value=2.5,
        step=0.5,
        help="Valores con esta cantidad de desviaciones estándar de la media serán marcados como anomalías"
    )
    
    # Marcar anomalías
    df['anomaly'] = abs(df['z_score']) > threshold
    
    # Crear visualización de anomalías
    fig = go.Figure()
    
    # Añadir línea principal de series temporales
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[target_variable],
        name=target_variable.replace('_', ' ').title(),
        mode='lines',
        line=dict(color='blue')
    ))
    
    # Añadir marcadores de anomalías
    fig.add_trace(go.Scatter(
        x=df[df['anomaly']]['date'],
        y=df[df['anomaly']][target_variable],
        name='Anomalía',
        mode='markers',
        marker=dict(color='red', size=8, line=dict(width=1, color='DarkSlateGrey'))
    ))
    
    # Configurar diseño
    fig.update_layout(
        title=f"Anomalías en {target_variable.replace('_', ' ').title()}",
        xaxis_title="Fecha",
        yaxis_title=target_variable.replace('_', ' ').title(),
        hovermode="x unified",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla opcional de detalles de anomalías
    if st.checkbox("Mostrar Detalles de Anomalías", help="Muestra información detallada sobre anomalías detectadas"):
        st.dataframe(
            df[df['anomaly']][['date', 'service_type', 'region', target_variable, 'z_score']]
            .sort_values('z_score', key=abs, ascending=False)
            .head(20)
            .style.format({
                target_variable: '{:,.2f}',
                'z_score': '{:.2f}'
            }),
            use_container_width=True
        )

# ==================== PESTAÑA DE AGRUPAMIENTO ====================
with tab3:
    if clustering_enabled:
        st.subheader("Análisis de Agrupamiento")
        st.markdown("""
        Agrupa registros similares para descubrir patrones en tus datos.
        Selecciona 2-3 características para analizar y ajusta el número de grupos.
        """)
        
        # Selección de características para agrupamiento
        clustering_features = st.multiselect(
            "Seleccionar características para agrupamiento",
            options=['labor_cost', 'transport_cost', 'equipment_cost', 
                    'waste_volume', 'service_hours', 'fuel_price'],
            default=['labor_cost', 'transport_cost', 'waste_volume'],
            help="Elige 2-3 características para usar en el análisis de agrupamiento"
        )
        
        # Control de cantidad de grupos
        n_clusters = st.slider(
            "Número de grupos",
            min_value=2,
            max_value=10,
            value=3,
            help="Ajusta para encontrar la agrupación más significativa de tus datos"
        )
        
        # Solo proceder si se seleccionan suficientes características
        if len(clustering_features) >= 2:
            # Preparar datos para agrupamiento
            X_cluster = df[clustering_features]
            
            # Estandarizar características (importante para agrupamiento basado en distancia)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            # Realizar agrupamiento K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            df['cluster'] = clusters
            
            # Crear visualización apropiada basada en cantidad de características
            if len(clustering_features) == 2:
                # Gráfico de dispersión 2D
                fig = px.scatter(
                    df,
                    x=clustering_features[0],
                    y=clustering_features[1],
                    color='cluster',
                    title="Visualización de Grupos en 2D",
                    hover_data=['service_type', 'region'],
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
            else:
                # Gráfico de dispersión 3D si se seleccionan 3 características
                fig = px.scatter_3d(
                    df,
                    x=clustering_features[0],
                    y=clustering_features[1],
                    z=clustering_features[2],
                    color='cluster',
                    title="Visualización de Grupos en 3D",
                    hover_data=['service_type', 'region'],
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
            
            # Mostrar la visualización
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar perfiles de grupos
            st.subheader("Perfiles de Grupos")
            st.markdown("""
            Compara valores promedio para cada característica entre grupos para entender sus características.
            """)
            
            cluster_profile = df.groupby('cluster')[clustering_features].mean()
            st.dataframe(
                cluster_profile.style.background_gradient(cmap='RdYlGn', axis=0),
                use_container_width=True
            )
    else:
        st.info("ℹ️ Habilita el análisis de agrupamiento en la barra lateral para usar esta función")

# ==================== PIE DE PÁGINA ====================
st.markdown("---")
st.markdown("""
**Portafolio de Ciencia de Datos** - Desarrollado por [Bo Kolstrup]  
[GitHub](https://github.com/Bokols) | [LinkedIn](https://www.linkedin.com/in/bo-kolstrup/)
""")