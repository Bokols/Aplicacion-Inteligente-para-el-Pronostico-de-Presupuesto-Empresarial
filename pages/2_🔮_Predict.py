# 2_🔮_Predict.py
# Panel de Análisis de Impacto Financiero
# Características: Modelado de Escenarios, Simulación Monte Carlo, Análisis de Sensibilidad y Reportes

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
from io import BytesIO

# =================================================================
# CONFIGURACIÓN DE PÁGINA
# =================================================================
st.set_page_config(
    page_title="Analizador de Impacto Financiero",
    page_icon="💹",
    layout="wide",
    menu_items={
        'Get help': 'https://github.com/Bokols',
        'About': """
        Herramienta avanzada de análisis de impacto financiero.
        Versión 2.4.0 | Última actualización: 2023-10-15
        """
    }
)

# Estilos CSS personalizados para mejor UI
st.markdown("""
<style>
    /* Estilo para tarjetas de métricas */
    .stMetric {
        border-radius: 8px;
        padding: 15px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .stMetric:hover {
        transform: translateY(-2px);
    }
    
    /* Estilo para contenedores de gráficos */
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Organización de la barra lateral */
    .sidebar .st-expander {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =================================================================
# TÍTULO E INTRODUCCIÓN
# =================================================================
st.title("💹 Analizador de Impacto Financiero")
with st.expander("ℹ️ Acerca de esta herramienta", expanded=False):
    st.markdown("""
    **Bienvenido al Analizador de Impacto Financiero**  
    Esta herramienta ayuda a profesionales a modelar escenarios financieros con:
    
    ### Características Clave
    - **Modelado de Escenarios**: Compara situaciones hipotéticas
    - **Pronóstico Probabilístico**: Simulaciones Monte Carlo
    - **Análisis de Sensibilidad**: Identifica impulsores clave de costos
    - **Reportes Integrales**: Exporta resultados en múltiples formatos
    
    ### Cómo Usar
    1. Ajusta los impulsores de costos en la barra lateral ←
    2. Establece el horizonte de pronóstico (7-90 días)
    3. Haz clic en "Ejecutar Análisis"
    4. Explora los resultados abajo →
    5. Guarda/exporta escenarios
    
    *Consejo: Pasa el cursor sobre cualquier parámetro para explicaciones detalladas*
    """)

# =================================================================
# SISTEMA DE CARGA DE MODELOS
# =================================================================
@st.cache_resource(show_spinner="Cargando modelos financieros...")
def load_models():
    """
    Carga y almacena en caché modelos predictivos para pronósticos financieros.
    
    Returns:
        dict: Diccionario de modelos disponibles con estructura:
        {
            'XGBoost': Objeto de modelo,
            'LightGBM': Objeto de modelo,
            'Stacking': Objeto de modelo,
            'ARIMA': Objeto de modelo
        }
    
    Nota: En producción, reemplazar con código real de carga de modelos como:
    >>> joblib.load('models/xgboost_model.pkl')
    """
    try:
        # Simulación para fines de demostración - reemplazar con carga real de modelos
        models = {
            'XGBoost': {
                'model': "Modelo XGBoost Simulado",
                'metrics': {
                    'MAE': 1250.42,
                    'RMSE': 1850.75,
                    'R2': 0.92,
                    'Accuracy': 0.89
                },
                'description': """
                **XGBoost (Extreme Gradient Boosting)**
                - Mejor para datos tabulares estructurados con patrones claros
                - Maneja bien relaciones complejas
                - Requiere ajuste cuidadoso de parámetros
                - Bueno para escenarios con muchos impulsores de costo
                """
            },
            'LightGBM': {
                'model': "Modelo LightGBM Simulado",
                'metrics': {
                    'MAE': 1305.18,
                    'RMSE': 1920.33,
                    'R2': 0.91,
                    'Accuracy': 0.87
                },
                'description': """
                **LightGBM (Light Gradient Boosting Machine)**
                - Entrenamiento más rápido que XGBoost
                - Bueno para grandes conjuntos de datos
                - Maneja bien características categóricas
                - Mejor para iteraciones rápidas
                """
            },
            'Stacking': {
                'model': "Modelo Apilado Simulado",
                'metrics': {
                    'MAE': 1180.65,
                    'RMSE': 1750.28,
                    'R2': 0.93,
                    'Accuracy': 0.90
                },
                'description': """
                **Modelo de Ensamblado Apilado**
                - Combina múltiples modelos para mejor precisión
                - Más costoso computacionalmente
                - Mejor para predicciones finales después de exploración
                - Reduce sesgos de modelos individuales
                """
            },
            'ARIMA': {
                'model': "Modelo de Series de Tiempo Simulado",
                'metrics': {
                    'MAE': 1450.80,
                    'RMSE': 2100.45,
                    'R2': 0.88,
                    'Accuracy': 0.85
                },
                'description': """
                **ARIMA (AutoRegressive Integrated Moving Average)**
                - Específicamente para datos de series de tiempo
                - Captura tendencias y estacionalidad
                - Mejor para pronósticos con patrones históricos
                - Requiere datos estacionarios
                """
            }
        }
        return models
    except Exception as e:
        st.error(f"""
        ### Error al Cargar Modelo
        **Causa**: {str(e)}
        
        **Pasos para Solucionar**:
        1. Verificar que los archivos de modelo existan en el directorio `/models`
        2. Revisar permisos de archivos
        3. Asegurar que las dependencias estén instaladas:
           ```bash
           pip install xgboost lightgbm statsmodels
           ```
        """)
        return None

models = load_models()
if models:
    st.toast("¡Modelos cargados exitosamente!", icon="✅")

# =================================================================
# MOTOR DE KPIs FINANCIEROS
# =================================================================
def calculate_kpis(df):
    """
    Calcula métricas financieras integrales a partir de datos operacionales.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada que contiene:
            - componentes de costo (mano de obra, transporte, etc.)
            - métricas operacionales (volumen de residuos, horas de servicio)
    
    Returns:
        pd.DataFrame: Mejorado con 15+ métricas financieras incluyendo:
            - Márgenes de Ganancia (bruto, operativo, neto)
            - Métricas de Eficiencia (costo por tonelada, productividad laboral)
            - ROI y Análisis de Punto de Equilibrio
    
    Fórmulas Implementadas:
        Margen Bruto = (Ingresos - COGS) / Ingresos
        Productividad Laboral = Ingresos / Costo Laboral
        ROI = Ganancia Neta / Inversión Total
        Volumen de Punto de Equilibrio = Costos Fijos / (Precio por Unidad - Costo Variable por Unidad)
    """
    df = df.copy()
    
    # Simulación de ingresos (multiplicador 1.4x-1.6x para demo)
    df['revenue'] = df['total_cost'] * np.random.uniform(1.4, 1.6, len(df))
    
    # Cálculos de rentabilidad
    df['gross_profit'] = df['revenue'] - df['total_cost']
    df['gross_margin'] = (df['gross_profit'] / df['revenue']) * 100
    df['operating_expenses'] = df['total_cost'] * np.random.uniform(0.7, 0.9, len(df))
    df['operating_income'] = df['revenue'] - df['operating_expenses']
    df['operating_margin'] = (df['operating_income'] / df['revenue']) * 100
    df['net_profit'] = df['operating_income'] * np.random.uniform(0.8, 0.95, len(df))
    df['net_margin'] = (df['net_profit'] / df['revenue']) * 100
    
    # Métricas de eficiencia
    df['labor_productivity'] = df['revenue'] / df['labor_cost']
    df['transportation_efficiency'] = df['waste_volume_tons'] / df['transportation_cost']
    df['equipment_utilization'] = df['revenue'] / df['equipment_cost']
    
    # Métricas operacionales
    df['cost_per_ton'] = df['total_cost'] / df['waste_volume_tons']
    df['profit_per_ton'] = df['net_profit'] / df['waste_volume_tons']
    df['service_cost_per_hour'] = df['total_cost'] / df['service_duration_hours']
    
    # Métricas de salud financiera
    df['roi'] = (df['net_profit'] / df['total_cost']) * 100
    df['break_even_volume'] = df['total_cost'] / (df['revenue'] / df['waste_volume_tons'])
    
    return df

# =================================================================
# GENERADOR DE DATOS DE MUESTRA
# =================================================================
@st.cache_data(show_spinner="Generando datos de muestra...")
def sample_data():
    """
    Genera datos operacionales realistas para fines de demostración.
    
    Características:
    - Datos de series de tiempo con fechas de servicio
    - Distribución regional (Norte/Sur/Este/Oeste)
    - Componentes de costo con rangos realistas
    - Bandera de material peligroso (20% de probabilidad)
    
    Returns:
        pd.DataFrame: 30 días de registros operacionales de muestra con:
        - Metadatos de servicio (tipo, región, duración)
        - Componentes de costo (mano de obra, transporte, equipo)
        - Métricas operacionales (volumen, estado de peligrosidad)
    """
    np.random.seed(42)
    size = 30
    df = pd.DataFrame({
        'service_date': pd.date_range('2023-01-01', periods=size),
        'service_type': np.random.choice(
            ['Recolección', 'Disposición', 'Reciclaje', 'Peligroso'], 
            size,
            p=[0.4, 0.3, 0.2, 0.1]  # Distribución de probabilidad
        ),
        'service_region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], size),
        'labor_cost': np.random.uniform(200, 1500, size),
        'transportation_cost': np.random.uniform(100, 800, size),
        'regulatory_fees': np.random.uniform(50, 500, size),
        'equipment_cost': np.random.uniform(100, 1200, size),
        'waste_volume_tons': np.random.uniform(1, 20, size),
        'service_duration_hours': np.random.uniform(1, 8, size),
        'hazardous_material': np.random.choice([True, False], size, p=[0.2, 0.8]),
        'fuel_price': np.random.uniform(2.5, 4.5, size)
    })
    
    # Calcular costo total con recargo por peligrosidad
    df['total_cost'] = (
        df['labor_cost'] + 
        df['transportation_cost'] + 
        df['regulatory_fees'] + 
        df['equipment_cost'] + 
        (df['hazardous_material'] * 500)  # Recargo de $500
    )
    
    return calculate_kpis(df)

df = sample_data()

# =================================================================
# SISTEMA DE GESTIÓN DE ESCENARIOS
# =================================================================
class ScenarioManager:
    """
    Gestiona escenarios financieros con funcionalidad de guardar/cargar.
    
    Características:
    - Almacena escenarios en el estado de sesión
    - Registra marcas de tiempo para control de versiones
    - Maneja tanto parámetros como resultados
    
    Ejemplo de Uso:
        >>> manager = ScenarioManager()
        >>> manager.save_scenario("Optimista", params, results)
        >>> scenario = manager.get_scenario("Optimista")
    """
    
    def __init__(self):
        """Inicializa almacenamiento de escenarios en estado de sesión"""
        if 'scenarios' not in st.session_state:
            st.session_state.scenarios = {}
            st.session_state.current_scenario = None
    
    def save_scenario(self, name, params, results):
        """
        Guarda un escenario con metadatos.
        
        Args:
            name (str): Identificador único del escenario
            params (dict): Parámetros de entrada
            results (dict): Resultados calculados
        """
        st.session_state.scenarios[name] = {
            'params': params,
            'results': results,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.current_scenario = name
        st.toast(f"¡Escenario '{name}' guardado!", icon="✅")
    
    def get_scenario_names(self):
        """Devuelve lista de todos los nombres de escenarios guardados"""
        return list(st.session_state.scenarios.keys())
    
    def get_scenario(self, name):
        """
        Recupera datos del escenario por nombre.
        
        Returns:
            dict: {'params': dict, 'results': dict, 'timestamp': str}
            o None si el escenario no existe
        """
        return st.session_state.scenarios.get(name)

scenario_manager = ScenarioManager()

# =================================================================
# CONTROLES DE BARRA LATERAL - Mejorados con tooltips
# =================================================================
with st.sidebar:
    st.header("⚙️ Configuración de Escenario")
    
    # Selección de Modelo
    with st.expander("🔮 Selección de Modelo", expanded=True):
        selected_model = st.selectbox(
            "Modelo de Pronóstico",
            options=list(models.keys()) if models else [],
            index=0,
            help="""Selecciona enfoque de modelado:
            - **XGBoost**: Mejor para datos estructurados
            - **LightGBM**: Alternativa más rápida
            - **Stacking**: Método de ensamblado
            - **ARIMA**: Para patrones de series de tiempo"""
        )
        
        # Mostrar descripción del modelo como markdown en lugar de expansor anidado
        if selected_model:
            st.markdown("#### Detalles del Modelo")
            st.markdown(models[selected_model]['description'])
        
        monte_carlo = st.checkbox(
            "Habilitar Simulación Monte Carlo",
            value=False,
            help="""Ejecuta pronóstico probabilístico:
            - Realiza múltiples simulaciones (100-500 recomendado)
            - Calcula intervalos de confianza (95% por defecto)
            - Muestra rangos de incertidumbre en pronósticos
            - Más preciso pero más lento de calcular"""
        )
    
    # Configuración de Pronóstico
    with st.expander("📅 Configuración de Pronóstico", expanded=True):
        forecast_days = st.slider(
            "Horizonte de Pronóstico (días)",
            min_value=7,
            max_value=90,
            value=30,
            step=1,
            help="""Período de tiempo de proyección:
            - Corto plazo (7-14 días): Alta precisión
            - Mediano plazo (15-30 días): Vista equilibrada
            - Largo plazo (31-90 días): Tendencias generales"""
        )
        
        num_simulations = st.slider(
            "Cantidad de Simulaciones",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            disabled=not monte_carlo,
            help="""Número de iteraciones Monte Carlo:
            - 10-100: Resultados rápidos, menos precisos
            - 100-500: Balance óptimo
            - 500-1000: Más precisos, pero más lentos"""
        )
    
    # Impulsores de Costo
    with st.expander("💰 Ajustes de Costos", expanded=True):
        st.markdown("**Palancas de Costo Operacional**")
        
        labor_adjust = st.slider(
            "Ajuste de Costo Laboral (%)",
            min_value=-30,
            max_value=30,
            value=0,
            help="""Simula cambios en:
            - Aumentos/disminuciones salariales
            - Políticas de horas extras
            - Cambios en niveles de personal
            - Impactos de contratos sindicales"""
        ) / 100
        
        transport_adjust = st.slider(
            "Ajuste de Costo de Transporte (%)",
            min_value=-30,
            max_value=30,
            value=0,
            help="""Modela impactos de:
            - Fluctuaciones en precios de combustible
            - Optimizaciones de rutas
            - Actualizaciones/degradaciones de flota
            - Cambios en costos de mantenimiento"""
        ) / 100
        
        regulatory_adjust = st.slider(
            "Ajuste de Tarifas Regulatorias (%)",
            min_value=-30,
            max_value=30,
            value=0,
            help="""Anticipa cambios en:
            - Tarifas de disposición
            - Costos de cumplimiento ambiental
            - Ajustes en tarifas de permisos
            - Cambios en políticas gubernamentales"""
        ) / 100
        
        equipment_adjust = st.slider(
            "Ajuste de Costo de Equipo (%)",
            min_value=-30,
            max_value=30,
            value=0,
            help="""Considera:
            - Inversiones de capital
            - Programas de mantenimiento
            - Cambios en depreciación
            - Costos de arrendamiento de equipos"""
        ) / 100
        
        waste_adjust = st.slider(
            "Ajuste de Volumen de Residuos (%)",
            min_value=-30,
            max_value=30,
            value=0,
            help="""Proyecta cambios en:
            - Fluctuaciones en demanda de servicios
            - Variaciones estacionales
            - Adquisiciones de nuevos clientes
            - Expansiones de área de servicio"""
        ) / 100
        
        fuel_adjust = st.slider(
            "Ajuste de Precio de Combustible (%)",
            min_value=-30,
            max_value=30,
            value=0,
            help="""Modela precios de commodities:
            - Volatilidad del mercado
            - Impactos en cadena de suministro
            - Factores geopolíticos
            - Adopción de combustibles alternativos"""
        ) / 100
        
        hazardous_toggle = st.checkbox(
            "Incluir Recargo por Peligrosidad",
            value=True,
            help="""Tarifa adicional de $500 por carga de material peligroso:
            - Requisitos de manejo especial
            - Costos de cumplimiento regulatorio
            - Primas de seguros más altas
            - Necesidades de equipo especializado"""
        )
    
    # Gestión de Escenarios
    with st.expander("💾 Gestión de Escenarios", expanded=True):
        scenario_name = st.text_input(
            "Nombre del Escenario",
            value="Escenario 1",
            help="""Nombre descriptivo para guardar/cargar:
            - Usa nombres claros (ej. "Optimista_Aumento_Combustible")
            - Incluye fecha si haces seguimiento de versiones
            - Mantén bajo 30 caracteres"""
        )
        
        if st.button(
            "💾 Guardar Escenario Actual",
            help="Almacena todos los parámetros y resultados para comparación posterior"
        ):
            params = {
                'model': selected_model,
                'forecast_days': forecast_days,
                'labor_adjust': labor_adjust,
                'transport_adjust': transport_adjust,
                'regulatory_adjust': regulatory_adjust,
                'equipment_adjust': equipment_adjust,
                'waste_adjust': waste_adjust,
                'fuel_adjust': fuel_adjust,
                'hazardous_toggle': hazardous_toggle,
                'monte_carlo': monte_carlo,
                'num_simulations': num_simulations
            }
            scenario_manager.save_scenario(scenario_name, params, {})
        
        selected_scenario = st.selectbox(
            "📂 Cargar Escenario",
            options=scenario_manager.get_scenario_names(),
            index=0,
            disabled=len(scenario_manager.get_scenario_names()) == 0,
            help="Recupera escenarios guardados previamente para comparación"
        )

# =================================================================
# SECCIÓN PRINCIPAL DE ANÁLISIS
# =================================================================
st.subheader("📊 Análisis de Impacto Financiero")
st.markdown("""
Ajusta parámetros en la barra lateral y haz clic abajo para analizar impactos financieros.
*Consejo: Compara múltiples escenarios guardando diferentes configuraciones*
""")

if st.button("🚀 Ejecutar Análisis", type="primary", help="Ejecuta modelado financiero"):
    
    with st.spinner("Analizando impactos financieros..."):
        # Mostrar métricas de precisión del modelo
        st.subheader("📊 Métricas de Rendimiento del Modelo")
        model_metrics = models[selected_model]['metrics']
        
        cols = st.columns(4)
        cols[0].metric(
            "Error Absoluto Medio (MAE)", 
            f"${model_metrics['MAE']:,.2f}", 
            help="""Diferencia promedio en dólares entre predicciones y valores reales:
            - Menor es mejor
            - Mide magnitud promedio de error
            - Fácil de interpretar (cantidad en dólares)
            - Ejemplo: $1,250 significa que la predicción promedio está $1,250 desviada"""
        )
        cols[1].metric(
            "Error Cuadrático Medio (RMSE)", 
            f"${model_metrics['RMSE']:,.2f}", 
            help="""Raíz cuadrada del promedio de errores cuadrados:
            - Penaliza errores grandes más que MAE
            - Mismas unidades que MAE (dólares)
            - Útil cuando errores grandes son particularmente malos
            - Ejemplo: $1,850 significa que errores grandes son significativos"""
        )
        cols[2].metric(
            "R-cuadrado (R²)", 
            f"{model_metrics['R2']:.2f}", 
            help="""Proporción de varianza explicada (escala 0-1):
            - 1 = predicción perfecta
            - 0 = sin poder predictivo
            - 0.9+ = excelente
            - 0.7-0.9 = bueno
            - Menos de 0.5 = pobre"""
        )
        cols[3].metric(
            "Precisión", 
            f"{model_metrics['Accuracy']:.2%}", 
            help="""Corrección general de predicción:
            - Porcentaje de predicciones correctas
            - Considera dirección (sobre/subestimación)
            - 90% significa 9/10 predicciones son correctas
            - Mejor para decisiones binarias"""
        )
        
        # Generar datos de escenario futuro
        future_dates = pd.date_range(datetime.today(), periods=forecast_days, freq='D')
        size = len(future_dates)

        future_df = pd.DataFrame({
            'date': future_dates,
            'service_type': np.random.choice(['Recolección', 'Disposición', 'Reciclaje', 'Peligroso'], size),
            'service_region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], size),
            'labor_cost': np.random.uniform(200, 1500, size) * (1 + labor_adjust),
            'transportation_cost': np.random.uniform(100, 800, size) * (1 + transport_adjust),
            'regulatory_fees': np.random.uniform(50, 500, size) * (1 + regulatory_adjust),
            'equipment_cost': np.random.uniform(100, 1200, size) * (1 + equipment_adjust),
            'waste_volume_tons': np.random.uniform(1, 20, size) * (1 + waste_adjust),
            'service_duration_hours': np.random.uniform(1, 8, size),
            'hazardous_material': np.random.choice([True, False], size, p=[0.2, 0.8]) if hazardous_toggle else np.zeros(size, dtype=bool),
            'fuel_price': np.random.uniform(2.5, 4.5, size) * (1 + fuel_adjust)
        })

        future_df['total_cost'] = (
            future_df['labor_cost'] + 
            future_df['transportation_cost'] + 
            future_df['regulatory_fees'] + 
            future_df['equipment_cost'] +
            (future_df['hazardous_material'] * 500)
        )

        future_df = calculate_kpis(future_df)
        
        # Simulación Monte Carlo
        if monte_carlo:
            simulations = []
            for _ in range(num_simulations):
                sim_df = future_df.copy()
                noise = np.random.normal(1, 0.1, size)
                sim_df['revenue'] = sim_df['revenue'] * noise
                sim_df = calculate_kpis(sim_df)
                simulations.append(sim_df)
            
            all_simulations = pd.concat(simulations)
            grouped_sim = all_simulations.groupby('date').agg(['mean', 'std'])
            
            for col in ['revenue', 'net_profit', 'gross_margin']:
                future_df[f'{col}_lower'] = grouped_sim[col]['mean'] - 1.96 * grouped_sim[col]['std']
                future_df[f'{col}_upper'] = grouped_sim[col]['mean'] + 1.96 * grouped_sim[col]['std']

        # Calcular cambios en KPIs
        baseline = sample_data().select_dtypes(include=[np.number]).mean()
        scenario = future_df.select_dtypes(include=[np.number]).mean()
        
        kpi_changes = pd.DataFrame({
            'KPI': [
                'Margen Bruto', 'Margen Operativo', 'Margen Neto',
                'Productividad Laboral', 'Eficiencia de Transporte', 
                'Costo por Tonelada', 'Ganancia por Tonelada', 'ROI'
            ],
            'Línea Base': [
                baseline['gross_margin'], baseline['operating_margin'], baseline['net_margin'],
                baseline['labor_productivity'], baseline['transportation_efficiency'],
                baseline['cost_per_ton'], baseline['profit_per_ton'], baseline['roi']
            ],
            'Escenario': [
                scenario['gross_margin'], scenario['operating_margin'], scenario['net_margin'],
                scenario['labor_productivity'], scenario['transportation_efficiency'],
                scenario['cost_per_ton'], scenario['profit_per_ton'], scenario['roi']
            ]
        })
        kpi_changes['Cambio (%)'] = ((kpi_changes['Escenario'] - kpi_changes['Línea Base']) / 
                                    kpi_changes['Línea Base']) * 100

        # Almacenar resultados
        if scenario_name in st.session_state.scenarios:
            st.session_state.scenarios[scenario_name]['results'] = {
                'future_df': future_df,
                'kpi_changes': kpi_changes,
                'baseline': baseline,
                'scenario': scenario
            }

        # =========================================================
        # TABLERO DE VISUALIZACIÓN
        # =========================================================
        
        # 1. Tablero de Rendimiento de KPIs
        st.subheader("📈 Tablero de Rendimiento")
        
        cols = st.columns(4)
        kpi_info = {
            'Margen Bruto': {
                'format': '{:.1f}%', 
                'delta': f"{kpi_changes.loc[0, 'Cambio (%)']:+.1f}%",
                'help': """Porcentaje de ingresos restantes después de costos directos:
                - Indicador de salud para operaciones principales
                - 30%+ es generalmente bueno para gestión de residuos
                - Observa tendencias a lo largo del tiempo"""
            },
            'Margen Neto': {
                'format': '{:.1f}%', 
                'delta': f"{kpi_changes.loc[2, 'Cambio (%)']:+.1f}%",
                'help': """Porcentaje de ganancia final después de todos los gastos:
                - Incluye costos indirectos (administración, impuestos)
                - 10%+ es generalmente saludable
                - Clave para sostenibilidad a largo plazo"""
            },
            'Costo por Tonelada': {
                'format': '${:.2f}', 
                'delta': f"{kpi_changes.loc[5, 'Cambio (%)']:+.1f}%",
                'help': """Costo operacional total por tonelada procesada:
                - Métrica clave de eficiencia
                - Comparar con estándares de la industria
                - Menor es mejor (más eficiente)"""
            },
            'ROI': {
                'format': '{:.1f}%', 
                'delta': f"{kpi_changes.loc[7, 'Cambio (%)']:+.1f}%",
                'help': """Porcentaje de Retorno sobre Inversión:
                - Mide eficiencia de capital
                - 15%+ es generalmente bueno
                - Comparar con inversiones alternativas"""
            }
        }
        
        for i, (kpi, info) in enumerate(kpi_info.items()):
            row = kpi_changes[kpi_changes['KPI'] == kpi].iloc[0]
            cols[i].metric(
                label=kpi,
                value=info['format'].format(row['Escenario']),
                delta=info['delta'],
                help=info['help'] + f"\n\nLínea Base: {info['format'].format(row['Línea Base'])}"
            )

        # 2. Visualización de Impacto en KPIs
        st.subheader("📊 Impactos en Métricas")
        st.markdown("""
        **Entendiendo el Gráfico de Impacto**  
        Esta visualización muestra cómo cambia cada métrica clave desde tu escenario base.  
        - **Barras verdes**: Mejoras positivas (mayores márgenes, menores costos)  
        - **Barras rojas**: Impactos negativos (menores márgenes, mayores costos)  
        - **Pasar cursor**: Ver valores exactos de línea base y escenario  
        """)
        
        fig = px.bar(
            kpi_changes,
            x='KPI',
            y='Cambio (%)',
            color='Cambio (%)',
            color_continuous_scale='RdYlGn',
            title="Cambio Porcentual desde Línea Base",
            labels={'Cambio (%)': 'Impacto (%)'},
            hover_data=['Línea Base', 'Escenario'],
            height=500
        )
        fig.update_layout(
            showlegend=False,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3. Tendencias de Rentabilidad
        st.subheader("💵 Tendencias de Margen")
        st.markdown("""
        **Interpretando Tendencias de Margen**  
        Este gráfico muestra cómo evolucionan tus márgenes de ganancia durante el período de pronóstico:  
        - **Margen Bruto** (verde): Ingresos menos costos directos  
        - **Margen Operativo** (azul): Después de gastos operativos  
        - **Margen Neto** (naranja): Ganancia final después de todos los costos  
        - **Área sombreada** (si Monte Carlo): Intervalo de confianza del 95%  
        """)
        
        fig = go.Figure()
        
        if monte_carlo:
            fig.add_trace(go.Scatter(
                x=future_df['date'],
                y=future_df['gross_margin_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(44,160,44,0.2)',
                showlegend=False,
                name='CI Superior'
            ))
            fig.add_trace(go.Scatter(
                x=future_df['date'],
                y=future_df['gross_margin_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(44,160,44,0.2)',
                name='95% Confianza'
            ))
        
        fig.add_trace(go.Scatter(
            x=future_df['date'],
            y=future_df['gross_margin'],
            name='Margen Bruto',
            line=dict(color='#2ca02c', width=3),
            hovertemplate="%{y:.1f}%<extra>Margen Bruto</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=future_df['date'],
            y=future_df['operating_margin'],
            name='Margen Operativo',
            line=dict(color='#1f77b4', width=3),
            hovertemplate="%{y:.1f}%<extra>Margen Operativo</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=future_df['date'],
            y=future_df['net_margin'],
            name='Margen Neto',
            line=dict(color='#ff7f0e', width=3),
            hovertemplate="%{y:.1f}%<extra>Margen Neto</extra>"
        ))
        
        fig.update_layout(
            title="Tendencias de Margen de Ganancia en el Tiempo",
            xaxis_title="Fecha",
            yaxis_title="Margen (%)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. Análisis de Sensibilidad
        st.subheader("🧪 Sensibilidad de Impulsores")
        st.markdown("""
        **Entendiendo el Análisis de Sensibilidad**  
        Muestra cómo un cambio del 1% en cada parámetro afecta la ganancia neta:  
        - **Barras altas**: Impulsores de costo más impactantes (enfócate aquí)  
        - **Barras cortas**: Impacto mínimo (prioridad baja)  
        - **Color**: Verde = aumento de ganancia, Rojo = disminución  
        """)
        
        params = ['labor_adjust', 'transport_adjust', 'regulatory_adjust', 
                 'equipment_adjust', 'waste_adjust', 'fuel_adjust']
        param_names = [p.replace('_', ' ').title() for p in params]
        sensitivities = []
        
        for param in params:
            impact = (future_df['net_profit'].mean() * (eval(param) + 0.01)) - future_df['net_profit'].mean()
            sensitivities.append(impact)
        
        sensitivity_df = pd.DataFrame({
            'Parámetro': param_names,
            'Impacto': sensitivities,
            'Ajuste Actual': [f"{eval(p)*100:.1f}%" for p in params]
        }).sort_values('Impacto', key=abs, ascending=False)
        
        fig = px.bar(
            sensitivity_df,
            x='Parámetro',
            y='Impacto',
            color='Impacto',
            color_continuous_scale='RdYlGn',
            title="Sensibilidad de Ganancia Neta a Cambios del 1%",
            hover_data=['Ajuste Actual'],
            labels={'Impacto': 'Impacto en Ganancia Neta ($)'},
            height=500
        )
        fig.update_layout(
            yaxis_title="Impacto en Ganancia Neta ($)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # 5. Sección de Exportación
        with st.expander("📤 Exportar Resultados", expanded=True):
            st.markdown("""
            **Opciones de Exportación**  
            - **Excel**: Conjunto de datos completo con múltiples hojas  
            - **CSV**: Datos crudos para análisis externo  
            - **HTML**: Reporte formateado para compartir  
            """)
            
            export_format = st.radio(
                "Formato",
                ['Excel', 'CSV', 'Reporte HTML'],
                horizontal=True,
                help="""Elige según tus necesidades:
                - Excel: Mejor para análisis adicional en hojas de cálculo
                - CSV: Para importar a otros sistemas
                - HTML: Reporte profesional para partes interesadas"""
            )
            
            if export_format == 'Excel':
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    future_df.to_excel(writer, sheet_name='Datos de Escenario', index=False)
                    kpi_changes.to_excel(writer, sheet_name='Cambios en KPIs', index=False)
                    sensitivity_df.to_excel(writer, sheet_name='Sensibilidad', index=False)
                    writer.close()
                st.download_button(
                    "⬇️ Descargar Excel",
                    data=output.getvalue(),
                    file_name=f"analisis_escenario_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime='application/vnd.ms-excel',
                    help="Descargar análisis completo como libro de Excel"
                )
            elif export_format == 'CSV':
                csv = future_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar CSV",
                    data=csv,
                    file_name=f"datos_escenario_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv',
                    help="Descargar datos crudos de escenario como CSV"
                )
            else:
                html = f"""
                <html>
                    <head>
                        <title>Reporte de Análisis de Escenario</title>
                        <style>
                            body {{ font-family: Arial; margin: 20px; }}
                            h1 {{ color: #2c3e50; }}
                            .kpi-table {{ width: 100%; border-collapse: collapse; }}
                            .kpi-table th, .kpi-table td {{ border: 1px solid #ddd; padding: 8px; }}
                            .positive {{ color: green; }}
                            .negative {{ color: red; }}
                        </style>
                    </head>
                    <body>
                        <h1>Reporte de Análisis de Escenario</h1>
                        <p>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <h2>Rendimiento del Modelo</h2>
                        <ul>
                            <li>Modelo: {selected_model}</li>
                            <li>MAE: ${model_metrics['MAE']:,.2f}</li>
                            <li>RMSE: ${model_metrics['RMSE']:,.2f}</li>
                            <li>R²: {model_metrics['R2']:.2f}</li>
                            <li>Precisión: {model_metrics['Accuracy']:.2%}</li>
                        </ul>
                        <h2>Métricas Clave</h2>
                        {kpi_changes.to_html(classes='kpi-table', index=False)}
                        <h2>Parámetros del Escenario</h2>
                        <ul>
                            <li>Modelo: {selected_model}</li>
                            <li>Días de Pronóstico: {forecast_days}</li>
                            <li>Ajuste Laboral: {labor_adjust*100:.1f}%</li>
                            <li>Ajuste de Transporte: {transport_adjust*100:.1f}%</li>
                            <li>Ajuste Regulatorio: {regulatory_adjust*100:.1f}%</li>
                            <li>Ajuste de Equipo: {equipment_adjust*100:.1f}%</li>
                            <li>Ajuste de Volumen: {waste_adjust*100:.1f}%</li>
                            <li>Ajuste de Combustible: {fuel_adjust*100:.1f}%</li>
                            <li>Recargo por Peligrosidad: {'Incluido' if hazardous_toggle else 'Excluido'}</li>
                        </ul>
                    </body>
                </html>
                """
                st.download_button(
                    "⬇️ Descargar HTML",
                    data=html,
                    file_name=f"reporte_escenario_{datetime.now().strftime('%Y%m%d')}.html",
                    mime='text/html',
                    help="Descargar reporte formateado para compartir"
                )

# =================================================================
# COMPARACIÓN DE ESCENARIOS
# =================================================================
if len(scenario_manager.get_scenario_names()) > 1:
    st.subheader("🔄 Comparación de Escenarios")
    st.markdown("""
    **Cómo Comparar Escenarios**  
    Selecciona 2+ escenarios guardados para comparar sus métricas clave lado a lado:  
    - Ayuda a identificar estrategias óptimas  
    - Visualiza compensaciones entre opciones  
    - Apoya la toma de decisiones basada en datos  
    """)
    
    selected_scenarios = st.multiselect(
        "Seleccionar escenarios para comparar",
        options=scenario_manager.get_scenario_names(),
        default=scenario_manager.get_scenario_names()[:2],
        help="Elige múltiples escenarios guardados para comparación"
    )
    
    if len(selected_scenarios) >= 2:
        comparison_data = []
        for name in selected_scenarios:
            scenario = scenario_manager.get_scenario(name)
            if scenario and 'results' in scenario:
                row = scenario['params'].copy()
                row.update({
                    'name': name,
                    'net_profit': scenario['results']['scenario']['net_profit'],
                    'gross_margin': scenario['results']['scenario']['gross_margin'],
                    'roi': scenario['results']['scenario']['roi']
                })
                comparison_data.append(row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                comparison_df,
                x='name',
                y=['net_profit', 'gross_margin', 'roi'],
                barmode='group',
                title="Comparación de Escenarios",
                labels={'value': 'Valor de Métrica', 'variable': 'Métrica'},
                height=500
            )
            fig.update_layout(
                legend_title_text='Métrica',
                xaxis_title="Escenario",
                yaxis_title="Valor",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

# =================================================================
# PIE DE PÁGINA
# =================================================================
st.markdown("---")
st.markdown("""
**Portafolio de Ciencia de Datos** - Desarrollado por [Bo Kolstrup]  
[GitHub](https://github.com/Bokols) | [LinkedIn](https://www.linkedin.com/in/bo-kolstrup/)
""")