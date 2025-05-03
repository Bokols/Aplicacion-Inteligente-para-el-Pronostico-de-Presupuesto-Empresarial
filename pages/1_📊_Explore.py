# 1_📊_Explore.py
"""
Explorador de Datos Operacionales

Esta aplicación interactiva de Streamlit proporciona herramientas completas para explorar y analizar
datos operacionales con visualizaciones enriquecidas y capacidades de exportación.

Características Clave:
- Filtrado interactivo de datos operacionales
- Análisis de series temporales con promedios móviles
- Comparaciones dimensionales entre tipos de servicio y regiones
- Visualizaciones de distribuciones estadísticas
- Exportación de datos en múltiples formatos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO

# ==============================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================
st.set_page_config(
    page_title="Explorador de Datos",
    page_icon="📊",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/Bokols',
        'About': "Herramienta interactiva de exploración de datos operacionales"
    }
)

# ==============================================
# ESTILOS PERSONALIZADOS
# ==============================================
st.markdown("""
<style>
    /* Estilos personalizados para tarjetas de métricas */
    .metric-card {
        border-radius: 8px;
        padding: 15px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    /* Estilos personalizados para gráficos Plotly */
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Estilos personalizados para tooltips */
    .custom-tooltip {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
# SECCIÓN DE TÍTULO
# ==============================================
st.title("📊 Explorador de Datos Operacionales")

# Expansor "Acerca de" con documentación detallada
with st.expander("ℹ️ Acerca de esta herramienta", expanded=False):
    st.markdown("""
    **Bienvenido al Explorador de Datos Operacionales**  
    Esta herramienta interactiva te ayuda a explorar y analizar datos operacionales con estas características:
    
    - **Filtrado cruzado**: Selecciona puntos de datos en un gráfico para filtrar otros
    - **Exportación de datos**: Descarga conjuntos de datos filtrados y visualizaciones
    - **Múltiples pequeños**: Compara tendencias entre dimensiones
    - **Tooltips mejorados**: Más información contextual
    
    **Cómo usar:**
    1. Selecciona filtros para enfocarte en segmentos específicos de datos
    2. Elige tipos de visualización que respondan a tus preguntas
    3. Pasa el cursor sobre los gráficos para ver valores detallados
    4. Exporta insights para reportes
    
    **Diccionario de Datos:**
    - **fecha**: Fecha del servicio
    - **tipo_servicio**: Tipo de servicio (Recolección, Disposición, Reciclaje, Peligroso)
    - **región**: Región geográfica del servicio
    - **costo_mano_obra**: Costo de mano de obra para el servicio ($)
    - **costo_transporte**: Costo de transporte ($)
    - **costo_equipo**: Costo de uso de equipo ($)
    - **volumen_residuos**: Volumen procesado (toneladas)
    - **horas_servicio**: Horas dedicadas al servicio
    - **peligroso**: Si hubo materiales peligrosos involucrados
    - **precio_combustible**: Precio actual del combustible al momento del servicio ($/galón)
    - **costo_total**: Suma de todos los costos incluyendo prima por peligrosidad ($)
    - **ingresos**: Ingresos generados por el servicio ($)
    - **ganancia**: Ingresos menos costos totales ($)
    - **margen_ganancia**: Ganancia como porcentaje de ingresos (%)
    """)

# ==============================================
# CARGA DE DATOS
# ==============================================
@st.cache_data(show_spinner="Cargando datos operacionales...")
def load_data():
    """
    Genera y almacena en caché un conjunto de datos sintéticos para operaciones.
    
    Returns:
        pd.DataFrame: Un DataFrame que contiene datos operacionales sintéticos con:
            - Fechas abarcando el mismo rango que Forecast.py (200 días desde 2023-01-01)
            - Tipos de servicio, regiones y costos generados aleatoriamente
            - Métricas calculadas como costo_total, ingresos y ganancia
    """
    try:
        np.random.seed(42)  # Para datos aleatorios reproducibles
        size = 200  # Número de registros a generar
        
        # Crear fechas que coincidan con Forecast.py (200 días desde 2023-01-01)
        date_range = pd.date_range('2023-01-01', periods=size)
        
        # Crear DataFrame con datos sintéticos
        df = pd.DataFrame({
            'date': date_range,
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
        
        # Calcular métricas derivadas
        df['total_cost'] = (df['labor_cost'] + df['transport_cost'] + 
                           df['equipment_cost'] + (df['hazardous'] * 500))
        df['revenue'] = df['total_cost'] * np.random.uniform(1.4, 1.6, size)
        df['profit'] = df['revenue'] - df['total_cost']
        df['profit_margin'] = (df['profit'] / df['revenue']) * 100
        
        return df
    except Exception as e:
        st.error(f"Error al generar datos de muestra: {str(e)}")
        return pd.DataFrame()

df = load_data()  # Cargar o generar el conjunto de datos

# ==============================================
# FILTROS DE BARRA LATERAL
# ==============================================
with st.sidebar:
    st.header("🔍 Filtros de Datos")
    
    # Filtro de rango de fechas - usando las mismas fechas que el dataframe
    with st.expander("Rango de Fechas", expanded=True):
        date_range = st.date_input(
            "Seleccionar Rango de Fechas",
            value=[df['date'].min(), df['date'].max()],
            min_value=df['date'].min(),
            max_value=df['date'].max(),
            help="Selecciona el rango de fechas para el análisis. Los datos fuera de este rango serán excluidos."
        )
    
    # Filtros de opciones de servicio
    with st.expander("Opciones de Servicio", expanded=True):
        service_types = st.multiselect(
            "Tipos de Servicio",
            options=df['service_type'].unique(),
            default=df['service_type'].unique(),
            help="Selecciona uno o más tipos de servicio para incluir en el análisis."
        )
        
        regions = st.multiselect(
            "Regiones",
            options=df['region'].unique(),
            default=df['region'].unique(),
            help="Selecciona una o más regiones para incluir en el análisis."
        )
        
        hazardous_filter = st.radio(
            "Materiales Peligrosos",
            options=['Todos', 'Sí', 'No'],
            index=0,
            help="Filtra servicios basados en la participación de materiales peligrosos."
        )

# ==============================================
# FILTRADO DE DATOS
# ==============================================
# Aplicar filtro de rango de fechas
filtered_df = df[
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1])) &
    (df['service_type'].isin(service_types)) &
    (df['region'].isin(regions))
]

# Aplicar filtro de peligrosidad si no es 'Todos'
if hazardous_filter != 'Todos':
    filtered_df = filtered_df[filtered_df['hazardous'] == (hazardous_filter == 'Sí')]

# ==============================================
# MÉTRICAS RESUMEN
# ==============================================
with st.container():
    st.subheader("📌 Resumen Rápido")
    cols = st.columns(4)
    
    # Métrica de Servicios Totales
    with cols[0]:
        st.metric(
            "Servicios Totales", 
            len(filtered_df),
            help="Número total de servicios en el conjunto de datos filtrado"
        )
    
    # Métrica de Ingresos Totales
    with cols[1]:
        st.metric(
            "Ingresos Totales", 
            f"${filtered_df['revenue'].sum():,.0f}",
            help="Suma de ingresos para todos los servicios en el conjunto de datos filtrado"
        )
    
    # Métrica de Margen de Ganancia Promedio
    with cols[2]:
        st.metric(
            "Margen Gan. Prom.", 
            f"{filtered_df['profit_margin'].mean():.1f}%",
            help="Margen de ganancia promedio (ganancia como porcentaje de ingresos) en todos los servicios"
        )
    
    # Métrica de Costo Promedio por Servicio
    with cols[3]:
        st.metric(
            "Costo Prom. por Servicio", 
            f"${filtered_df['total_cost'].mean():,.0f}",
            help="Costo total promedio por servicio incluyendo mano de obra, transporte, equipo y primas por peligrosidad"
        )

# ==============================================
# PESTAÑAS DE ANÁLISIS PRINCIPAL
# ==============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tendencias", 
    "🔍 Comparaciones", 
    "📊 Distribuciones", 
    "💾 Exportar"
])

# ==============================================
# PESTAÑA TENDENCIAS - Análisis de Series Temporales
# ==============================================
with tab1:
    st.subheader("Análisis de Series Temporales")
    
    # Selección de métrica y opciones de visualización
    col1, col2 = st.columns([3, 1])
    with col1:
        trend_metric = st.selectbox(
            "Seleccionar Métrica",
            options=['total_cost', 'revenue', 'profit', 'waste_volume', 'service_hours'],
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Selecciona la métrica para analizar en el tiempo"
        )
    with col2:
        show_small_multiples = st.checkbox(
            "Mostrar Múltiples Pequeños", 
            value=False,
            help="Muestra gráficos separados para cada combinación de región y tipo de servicio"
        )
    
    # Generar gráfico de series temporales apropiado basado en selecciones
    if show_small_multiples:
        fig = px.line(
            filtered_df,
            x='date',
            y=trend_metric,
            facet_col='region',
            facet_row='service_type',
            title=f"{trend_metric.replace('_', ' ').title()} en el Tiempo por Región & Servicio",
            labels={'date': 'Fecha', trend_metric: trend_metric.replace('_', ' ').title()},
            hover_data=['hazardous', 'fuel_price'],
            height=800
        )
    else:
        fig = px.line(
            filtered_df,
            x='date',
            y=trend_metric,
            color='service_type',
            title=f"{trend_metric.replace('_', ' ').title()} en el Tiempo",
            hover_data=['region', 'hazardous', 'fuel_price'],
            labels={'date': 'Fecha', trend_metric: trend_metric.replace('_', ' ').title()}
        )
    
    # Mejorar apariencia del gráfico
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Fecha",
        yaxis_title=trend_metric.replace('_', ' ').title(),
        legend_title="Tipo de Servicio"
    )
    
    # Añadir plantilla personalizada para tooltip
    fig.update_traces(
        hovertemplate="<b>Fecha</b>: %{x}<br>" +
                     f"<b>{trend_metric.replace('_', ' ').title()}</b>: %{{y:,.0f}}<br>" +
                     "<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de promedios móviles
    st.subheader("Promedios Móviles")
    window = st.slider(
        "Ventana Móvil (días)", 
        7, 30, 7,
        help="Número de días a incluir en el cálculo del promedio móvil"
    )
    
    # Calcular promedios móviles
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    rolling_df = filtered_df.set_index('date')[numeric_cols].rolling(f'{window}D').mean().reset_index()
    
    # Crear gráfico de promedios móviles
    fig = px.line(
        rolling_df,
        x='date',
        y=['total_cost', 'revenue', 'profit'],
        title=f"Promedios Móviles de {window} Días",
        labels={'value': 'Monto ($)', 'variable': 'Métrica', 'date': 'Fecha'},
        hover_data={'date': True}
    )
    
    # Mejorar apariencia del gráfico
    fig.update_layout(
        hovermode="x unified",
        legend_title="Métrica Financiera"
    )
    
    # Añadir plantilla personalizada para tooltip
    fig.update_traces(
        hovertemplate="<b>Fecha</b>: %{x}<br>" +
                     "<b>Métrica</b>: %{fullData.name}<br>" +
                     "<b>Valor</b>: %{y:,.0f}<br>" +
                     "<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==============================================
# PESTAÑA COMPARACIONES - Análisis Dimensional
# ==============================================
with tab2:
    st.subheader("Comparaciones Dimensionales")
    
    # Selección de dimensión y métrica para comparación
    col1, col2 = st.columns(2)
    with col1:
        compare_dimension = st.selectbox(
            "Comparar Por",
            options=['service_type', 'region', 'hazardous'],
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Selecciona la dimensión para comparar métricas"
        )
    with col2:
        compare_metric = st.selectbox(
            "Métrica a Comparar",
            options=['total_cost', 'revenue', 'profit', 'waste_volume'],
            index=2,
            help="Selecciona la métrica para comparar entre dimensiones"
        )
    
    # Selección de tipo de gráfico
    chart_type = st.radio(
        "Tipo de Gráfico", 
        ['Barras', 'Violín', 'Caja'], 
        horizontal=True,
        help="Selecciona el tipo de visualización para la comparación"
    )
    
    # Generar gráfico de comparación apropiado
    if chart_type == 'Barras':
        fig = px.bar(
            filtered_df,
            x=compare_dimension,
            y=compare_metric,
            color=compare_dimension,
            barmode='group',
            title=f"{compare_metric.replace('_', ' ').title()} por {compare_dimension.replace('_', ' ').title()}",
            hover_data=['date', 'region', 'service_type'],
            labels={
                compare_dimension: compare_dimension.replace('_', ' ').title(),
                compare_metric: compare_metric.replace('_', ' ').title()
            }
        )
    elif chart_type == 'Violín':
        fig = px.violin(
            filtered_df,
            x=compare_dimension,
            y=compare_metric,
            color=compare_dimension,
            box=True,
            points="all",
            title=f"Distribución de {compare_metric.replace('_', ' ').title()} por {compare_dimension.replace('_', ' ').title()}",
            hover_data=['date', 'region', 'service_type']
        )
    else:  # Gráfico de caja
        fig = px.box(
            filtered_df,
            x=compare_dimension,
            y=compare_metric,
            color=compare_dimension,
            title=f"Distribución de {compare_metric.replace('_', ' ').title()} por {compare_dimension.replace('_', ' ').title()}",
            hover_data=['date', 'region', 'service_type']
        )
    
    # Mejorar apariencia del gráfico
    fig.update_layout(
        hovermode="closest",
        legend_title=compare_dimension.replace('_', ' ').title()
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de composición de costos
    st.subheader("Composición de Costos")
    cost_components = ['labor_cost', 'transport_cost', 'equipment_cost']
    cost_df = filtered_df.groupby(compare_dimension)[cost_components].mean().reset_index()
    cost_df = cost_df.melt(
        id_vars=compare_dimension, 
        var_name='cost_type', 
        value_name='amount'
    )
    
    # Crear gráfico de composición de costos
    fig = px.bar(
        cost_df,
        x=compare_dimension,
        y='amount',
        color='cost_type',
        title=f"Estructura de Costos por {compare_dimension.replace('_', ' ').title()}",
        labels={
            'amount': 'Costo ($)',
            compare_dimension: compare_dimension.replace('_', ' ').title(),
            'cost_type': 'Componente de Costo'
        },
        hover_data=['cost_type']
    )
    
    # Mejorar apariencia del gráfico
    fig.update_layout(
        hovermode="x unified",
        legend_title="Componente de Costo"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==============================================
# PESTAÑA DISTRIBUCIONES - Análisis Estadístico
# ==============================================
with tab3:
    st.subheader("Distribuciones de Datos")
    
    # Selección de métrica para distribución
    dist_metric = st.selectbox(
        "Seleccionar Distribución",
        options=['total_cost', 'revenue', 'profit', 'waste_volume', 'service_hours', 'fuel_price'],
        index=2,
        help="Selecciona la métrica para analizar su distribución"
    )
    
    # Mostrar visualizaciones de distribución en columnas
    col1, col2 = st.columns(2)
    
    # Histograma con gráfico de caja
    with col1:
        fig = px.histogram(
            filtered_df,
            x=dist_metric,
            nbins=20,
            title=f"Distribución de {dist_metric.replace('_', ' ').title()}",
            marginal="box",
            hover_data=['date', 'service_type', 'region'],
            labels={dist_metric: dist_metric.replace('_', ' ').title()}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de caja por tipo de servicio
    with col2:
        fig = px.box(
            filtered_df,
            y=dist_metric,
            x='service_type',
            color='service_type',
            title=f"Distribución por Tipo de Servicio",
            hover_data=['date', 'region'],
            labels={
                dist_metric: dist_metric.replace('_', ' ').title(),
                'service_type': 'Tipo de Servicio'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de correlación
    st.subheader("Análisis de Correlación")
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    corr_matrix = filtered_df[numeric_cols].corr()
    
    # Crear mapa de calor de correlación
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlación Entre Métricas",
        color_continuous_scale='RdBu',
        range_color=[-1, 1],
        labels=dict(x="Métrica", y="Métrica", color="Correlación")
    )
    
    # Mejorar apariencia del mapa de calor
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        hoverlabel=dict(bgcolor="white"),
        xaxis_title="Métrica",
        yaxis_title="Métrica"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==============================================
# PESTAÑA EXPORTAR - Exportación de Datos
# ==============================================
with tab4:
    st.subheader("Exportar Datos & Visualizaciones")
    
    # Sección de exportación de datos
    with st.expander("Exportar Datos", expanded=True):
        st.write(f"El conjunto de datos filtrado contiene {len(filtered_df)} registros")
        
        export_format = st.radio(
            "Formato de Exportación", 
            ['CSV', 'Excel', 'JSON'], 
            horizontal=True,
            help="Selecciona el formato de archivo para exportar los datos filtrados"
        )
        
        # Exportación CSV
        if export_format == 'CSV':
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Descargar CSV",
                data=csv,
                file_name=f"datos_operacionales_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
                help="Descarga los datos filtrados como archivo CSV"
            )
        # Exportación Excel
        elif export_format == 'Excel':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Datos')
                writer.close()
            st.download_button(
                "Descargar Excel",
                data=output.getvalue(),
                file_name=f"datos_operacionales_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime='application/vnd.ms-excel',
                help="Descarga los datos filtrados como archivo Excel"
            )
        # Exportación JSON
        else:
            json = filtered_df.to_json(orient='records')
            st.download_button(
                "Descargar JSON",
                data=json,
                file_name=f"datos_operacionales_{datetime.now().strftime('%Y%m%d')}.json",
                mime='application/json',
                help="Descarga los datos filtrados como archivo JSON"
            )
    
    # Sección de exportación de visualizaciones
    with st.expander("Exportar Visualizaciones", expanded=True):
        st.write("Próximamente: Exportar visualizaciones interactivas como PNG o PDF")
        st.info("""
        Por ahora, usa el ícono de cámara en la esquina superior derecha de cada gráfico para exportar.
        Nota: Esto exporta la vista actual del gráfico con todos los filtros aplicados.
        """, icon="ℹ️")

# ==============================================
# PIE DE PÁGINA
# ==============================================
st.markdown("---")
st.markdown("""
**Portafolio de Ciencia de Datos** - Desarrollado por [Bo Kolstrup]  
[GitHub](https://github.com/Bokols) | [LinkedIn](https://www.linkedin.com/in/bo-kolstrup/)
""")