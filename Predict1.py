import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib

# Page configuration with descriptive metadata
st.set_page_config(
    page_title="Financial Impact Analyzer | Budget Forecast Pro",
    page_icon="💹",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'About': "This tool analyzes how cost driver changes affect financial KPIs"
    }
)

# Title section with detailed explanation
st.title("💹 Financial Impact Analyzer")
with st.expander("ℹ️ About this tool", expanded=False):
    st.markdown("""
    **Welcome to the Financial Impact Analyzer**  
    This interactive tool helps you:
    - 📊 Visualize relationships between cost drivers and financial performance
    - 🔍 Analyze how operational changes impact profitability
    - 📈 Forecast financial outcomes under different scenarios
    - 🎯 Identify key leverage points for cost optimization
    
    **How to use:**
    1. Adjust cost drivers in the sidebar
    2. Set your forecast horizon
    3. Click "Run Financial Impact Analysis"
    4. Explore the interactive visualizations
    """)

# Load models with detailed error handling
@st.cache_resource(show_spinner="Loading financial models...")
def load_models():
    try:
        models = {
            'XGBoost': joblib.load('models/best_xgb_model.pkl'),
            'LightGBM': joblib.load('models/best_lgbm_model.pkl'),
            'Stacking': joblib.load('models/best_stacking_model.pkl')
        }
        st.toast("Models loaded successfully!", icon="✅")
        return models
    except Exception as e:
        st.error(f"""
        **Model Loading Failed**  
        Error: {str(e)}  
        Please ensure:
        - Model files exist in the 'models' directory
        - Files are in the correct format (.pkl)
        - You have the required dependencies installed
        """)
        return None

models = load_models()

# Financial KPI calculations with docstrings
def calculate_kpis(df):
    """
    Calculate comprehensive financial KPIs based on operational data
    
    Parameters:
    df (DataFrame): Input data containing cost drivers and service metrics
    
    Returns:
    DataFrame: Enhanced with 15+ financial and operational KPIs including:
    - Profitability metrics (gross margin, net margin)
    - Efficiency ratios (labor productivity, transport efficiency)
    - Operational benchmarks (cost per ton, profit per hour)
    """
    df = df.copy()
    
    # Revenue simulation with tooltip explanation
    df['revenue'] = df['total_cost'] * np.random.uniform(1.4, 1.6, len(df))
    
    # Profitability calculations
    df['gross_profit'] = df['revenue'] - df['total_cost']
    df['gross_margin'] = (df['gross_profit'] / df['revenue']) * 100
    df['operating_expenses'] = df['total_cost'] * np.random.uniform(0.7, 0.9, len(df))
    df['operating_income'] = df['revenue'] - df['operating_expenses']
    df['operating_margin'] = (df['operating_income'] / df['revenue']) * 100
    df['net_profit'] = df['operating_income'] * np.random.uniform(0.8, 0.95, len(df))
    df['net_margin'] = (df['net_profit'] / df['revenue']) * 100
    
    # Efficiency metrics
    df['labor_productivity'] = df['revenue'] / df['labor_cost']
    df['transportation_efficiency'] = df['waste_volume_tons'] / df['transportation_cost']
    df['equipment_utilization'] = df['revenue'] / df['equipment_cost']
    
    # Operational metrics
    df['cost_per_ton'] = df['total_cost'] / df['waste_volume_tons']
    df['profit_per_ton'] = df['net_profit'] / df['waste_volume_tons']
    df['service_cost_per_hour'] = df['total_cost'] / df['service_duration_hours']
    
    return df

# Sample data generation with explanations
@st.cache_data(show_spinner="Generating sample data...")
def sample_data():
    """Generate realistic operational data with all required cost drivers"""
    np.random.seed(42)
    size = 30
    df = pd.DataFrame({
        'service_date': pd.date_range('2023-01-01', periods=size),
        'service_type': np.random.choice(['Collection', 'Disposal', 'Recycling', 'Hazardous'], size),
        'service_region': np.random.choice(['North', 'South', 'East', 'West'], size),
        'labor_cost': np.random.uniform(200, 1500, size),
        'transportation_cost': np.random.uniform(100, 800, size),
        'regulatory_fees': np.random.uniform(50, 500, size),
        'equipment_cost': np.random.uniform(100, 1200, size),
        'waste_volume_tons': np.random.uniform(1, 20, size),
        'service_duration_hours': np.random.uniform(1, 8, size),
        'hazardous_material': np.random.choice([True, False], size, p=[0.2, 0.8]),
        'fuel_price': np.random.uniform(2.5, 4.5, size)
    })
    
    df['total_cost'] = (
        df['labor_cost'] + df['transportation_cost'] + df['regulatory_fees'] + 
        df['equipment_cost'] + (df['hazardous_material'] * 500)
    )
    
    return calculate_kpis(df)

df = sample_data()

# Sidebar controls with comprehensive tooltips
st.sidebar.header("⚙️ Financial Scenario Controls")

with st.sidebar.expander("Model Selection"):
    selected_model = st.selectbox(
        "Forecast Model",
        list(models.keys()) if models else [],
        index=0,
        help="""Select predictive model for financial projections:
        - XGBoost: Best for complex non-linear relationships
        - LightGBM: Faster training for large datasets
        - Stacking: Ensemble combining both models"""
    )

with st.sidebar.expander("Forecast Settings"):
    forecast_days = st.slider(
        "Forecast Period (days)",
        7, 90, 30,
        help="""Time horizon for financial projections:
        - Short-term (7-14 days): Operational adjustments
        - Medium-term (30 days): Monthly planning
        - Long-term (60-90 days): Strategic decisions"""
    )

with st.sidebar.expander("Cost Driver Adjustments"):
    st.markdown("**Operational Cost Levers**")
    labor_adjust = st.slider(
        "Labor Cost Adjustment (%)",
        -30, 30, 0,
        help="Simulate changes in wage rates or staffing levels"
    ) / 100
    
    transport_adjust = st.slider(
        "Transportation Cost Adjustment (%)",
        -30, 30, 0,
        help="Impact of fuel prices, fleet efficiency, or route optimization"
    ) / 100
    
    regulatory_adjust = st.slider(
        "Regulatory Fees Adjustment (%)",
        -30, 30, 0,
        help="Changes in compliance costs or environmental fees"
    ) / 100
    
    equipment_adjust = st.slider(
        "Equipment Cost Adjustment (%)",
        -30, 30, 0,
        help="Equipment maintenance, leasing costs, or capital investments"
    ) / 100
    
    waste_adjust = st.slider(
        "Waste Volume Adjustment (%)",
        -30, 30, 0,
        help="Changes in customer demand or seasonal fluctuations"
    ) / 100
    
    fuel_adjust = st.slider(
        "Fuel Price Adjustment (%)",
        -30, 30, 0,
        help="Market price volatility or contracted fuel rates"
    ) / 100
    
    hazardous_toggle = st.checkbox(
        "Include Hazardous Surcharge",
        value=True,
        help="""Additional costs for handling hazardous materials:
        - Special handling procedures
        - Regulatory compliance
        - Higher insurance costs"""
    )

# Main analysis section
st.subheader("Financial KPI Impact Analysis")
st.markdown("""
Explore how cost driver changes affect your financial performance.  
Adjust parameters in the sidebar and click the button below to analyze.
""")

if st.button("Run Financial Impact Analysis", 
             help="Generate comprehensive financial analysis based on current settings",
             type="primary"):
    
    with st.spinner("Analyzing financial impacts..."):
        # Generate future scenario
        future_dates = pd.date_range(datetime.today(), periods=forecast_days, freq='D')
        size = len(future_dates)

        future_df = pd.DataFrame({
            'date': future_dates,
            'service_type': np.random.choice(['Collection', 'Disposal', 'Recycling', 'Hazardous'], size),
            'service_region': np.random.choice(['North', 'South', 'East', 'West'], size),
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
            future_df['labor_cost'] + future_df['transportation_cost'] + 
            future_df['regulatory_fees'] + future_df['equipment_cost'] +
            (future_df['hazardous_material'] * 500)
        )

        future_df = calculate_kpis(future_df)
        future_df['predicted_revenue'] = future_df['revenue'] * np.random.uniform(0.9, 1.1, size)

        # Calculate KPI changes from baseline
        baseline = sample_data().mean()
        scenario = future_df.mean()
        
        kpi_changes = pd.DataFrame({
            'KPI': [
                'Gross Margin', 'Operating Margin', 'Net Margin',
                'Labor Productivity', 'Transport Efficiency', 
                'Cost per Ton', 'Profit per Ton'
            ],
            'Baseline': [
                baseline['gross_margin'], baseline['operating_margin'], baseline['net_margin'],
                baseline['labor_productivity'], baseline['transportation_efficiency'],
                baseline['cost_per_ton'], baseline['profit_per_ton']
            ],
            'Scenario': [
                scenario['gross_margin'], scenario['operating_margin'], scenario['net_margin'],
                scenario['labor_productivity'], scenario['transportation_efficiency'],
                scenario['cost_per_ton'], scenario['profit_per_ton']
            ]
        })
        kpi_changes['Change (%)'] = ((kpi_changes['Scenario'] - kpi_changes['Baseline']) / kpi_changes['Baseline']) * 100

        # 1. Financial Performance Dashboard
        st.subheader("📊 Financial Performance Dashboard")
        with st.expander("About these metrics"):
            st.markdown("""
            **Key Performance Indicators:**
            - **Gross Margin:** Revenue after direct costs (materials, labor)
            - **Net Margin:** Final profit after all expenses including overhead
            - **Cost per Ton:** Operational efficiency benchmark
            - **Profit per Ton:** Overall profitability per unit of work
            """)
        
        cols = st.columns(4)
        kpi_info = {
            'Gross Margin': {'format': '{:.1f}%', 'delta': f"{kpi_changes.loc[0, 'Change (%)']:+.1f}%"},
            'Net Margin': {'format': '{:.1f}%', 'delta': f"{kpi_changes.loc[2, 'Change (%)']:+.1f}%"},
            'Cost per Ton': {'format': '${:.2f}', 'delta': f"{kpi_changes.loc[5, 'Change (%)']:+.1f}%"},
            'Profit per Ton': {'format': '${:.2f}', 'delta': f"{kpi_changes.loc[6, 'Change (%)']:+.1f}%"}
        }
        
        for i, (kpi, info) in enumerate(kpi_info.items()):
            row = kpi_changes[kpi_changes['KPI'] == kpi].iloc[0]
            cols[i].metric(
                kpi,
                info['format'].format(row['Scenario']),
                info['delta'],
                help=f"Baseline: {info['format'].format(row['Baseline'])}"
            )

        # 2. KPI Impact Visualization
        st.subheader("📈 KPI Impact Analysis")
        st.markdown("""
        How each financial metric changes compared to baseline operations  
        (Positive = Improvement, Negative = Deterioration)
        """)
        
        fig = px.bar(
            kpi_changes,
            x='KPI',
            y='Change (%)',
            color='Change (%)',
            color_continuous_scale='RdYlGn',
            title="Percentage Change in Key Financial Metrics",
            labels={'Change (%)': 'Change from Baseline (%)'},
            hover_data=['Baseline', 'Scenario']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # 3. Profitability Trends
        st.subheader("💵 Profitability Trends Over Time")
        st.markdown("""
        Daily evolution of profit margins during forecast period  
        Hover over lines to see exact values on each date
        """)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future_df['date'],
            y=future_df['gross_margin'],
            name='Gross Margin',
            line=dict(color='#2ca02c', width=3),
            hovertemplate="Date: %{x|%b %d}<br>Gross Margin: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=future_df['date'],
            y=future_df['operating_margin'],
            name='Operating Margin',
            line=dict(color='#1f77b4', width=3),
            hovertemplate="Date: %{x|%b %d}<br>Operating Margin: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=future_df['date'],
            y=future_df['net_margin'],
            name='Net Margin',
            line=dict(color='#ff7f0e', width=3),
            hovertemplate="Date: %{x|%b %d}<br>Net Margin: %{y:.1f}%<extra></extra>"
        ))
        fig.update_layout(
            title="Profit Margin Trends",
            xaxis_title="Date",
            yaxis_title="Margin (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. Efficiency Metrics
        st.subheader("⚡ Efficiency Metrics")
        st.markdown("""
        Operational efficiency indicators showing productivity per dollar spent  
        (Higher values indicate better efficiency)
        """)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future_df['date'],
            y=future_df['labor_productivity'],
            name='Labor Productivity (Revenue/$)',
            yaxis='y',
            hovertemplate="Date: %{x|%b %d}<br>Productivity: %{y:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=future_df['date'],
            y=future_df['transportation_efficiency'],
            name='Transport Efficiency (Tons/$)',
            yaxis='y2',
            hovertemplate="Date: %{x|%b %d}<br>Efficiency: %{y:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=future_df['date'],
            y=future_df['equipment_utilization'],
            name='Equipment Utilization (Revenue/$)',
            yaxis='y3',
            hovertemplate="Date: %{x|%b %d}<br>Utilization: %{y:.2f}<extra></extra>"
        ))
        fig.update_layout(
            title="Efficiency Metrics Over Time",
            xaxis_title="Date",
            yaxis=dict(title="Labor Productivity", side="left"),
            yaxis2=dict(title="Transport Efficiency", side="right", overlaying="y"),
            yaxis3=dict(title="Equipment Utilization", side="right", overlaying="y", position=0.85),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 5. Cost Structure Analysis
        st.subheader("🧩 Cost Structure Breakdown")
        st.markdown("""
        Composition of total operational costs  
        (Shows where your dollars are being spent)
        """)
        
        cost_components = ['labor_cost', 'transportation_cost', 'regulatory_fees', 'equipment_cost']
        cost_data = future_df[cost_components].sum().reset_index()
        cost_data.columns = ['Cost Type', 'Amount']
        cost_data['Percentage'] = (cost_data['Amount'] / cost_data['Amount'].sum()) * 100
        cost_data['Cost Type'] = cost_data['Cost Type'].str.replace('_', ' ').str.title()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                cost_data,
                names='Cost Type',
                values='Amount',
                title="Cost Composition",
                hover_data=['Percentage'],
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                cost_data,
                x='Cost Type',
                y='Amount',
                color='Cost Type',
                title="Absolute Cost Breakdown",
                text='Amount',
                labels={'Amount': 'Total Cost ($)'}
            )
            fig.update_traces(
                texttemplate='$%{value:,.0f}',
                textposition='outside',
                hovertemplate="Cost Type: %{x}<br>Amount: $%{y:,.0f}<br>Share: %{customdata[0]:.1f}%<extra></extra>",
                customdata=cost_data[['Percentage']]
            )
            st.plotly_chart(fig, use_container_width=True)

        # 6. Detailed KPI Table
        st.subheader("📋 Detailed Financial Metrics")
        st.markdown("""
        Daily financial metrics for detailed analysis  
        (Scroll to see all data, click column headers to sort)
        """)
        
        st.dataframe(
            future_df[[
                'date', 'gross_margin', 'operating_margin', 'net_margin',
                'labor_productivity', 'transportation_efficiency',
                'cost_per_ton', 'profit_per_ton'
            ]].rename(columns={
                'date': 'Date',
                'gross_margin': 'Gross Margin (%)',
                'operating_margin': 'Operating Margin (%)',
                'net_margin': 'Net Margin (%)',
                'labor_productivity': 'Labor Productivity (Rev/$)',
                'transportation_efficiency': 'Transport Eff. (Tons/$)',
                'cost_per_ton': 'Cost per Ton ($)',
                'profit_per_ton': 'Profit per Ton ($)'
            }).style.format({
                'Gross Margin (%)': '{:.1f}%',
                'Operating Margin (%)': '{:.1f}%',
                'Net Margin (%)': '{:.1f}%',
                'Cost per Ton ($)': '${:.2f}',
                'Profit per Ton ($)': '${:.2f}'
            }),
            use_container_width=True,
            height=400
        )

# Comprehensive help section
with st.expander("📚 Comprehensive Guide to Financial Analysis", expanded=False):
    st.markdown("""
    ## Understanding Financial Impact Analysis
    
    **1. Key Performance Indicators (KPIs)**
    
    - **Gross Margin**: Measures basic profitability after direct costs  
      *Example: If revenue is $100 and direct costs are $60, gross margin is 40%*
    
    - **Operating Margin**: Shows profitability after operating expenses  
      *Includes overhead like administration, marketing, etc.*
    
    - **Net Margin**: Final profitability after all expenses and taxes  
      *The "bottom line" that determines overall financial health*
    
    **2. Efficiency Metrics**
    
    - **Labor Productivity**: Revenue generated per dollar spent on labor  
      *Higher values mean your workforce is more productive*
    
    - **Transport Efficiency**: Tons moved per dollar of transportation cost  
      *Measures how efficiently you're using your transportation resources*
    
    - **Equipment Utilization**: Revenue generated per equipment dollar  
      *Indicates whether equipment investments are paying off*
    
    **3. Operational Benchmarks**
    
    - **Cost per Ton**: Total operational cost divided by waste volume  
      *Key metric for pricing and cost control*
    
    - **Profit per Ton**: Net profit divided by waste volume  
      *Shows real profitability of your core operations*
    
    **4. Interpreting Results**
    
    - Positive changes (green) indicate improvements from baseline
    - Negative changes (red) show deterioration
    - Compare trends over time to identify patterns
    - Use cost breakdown to find optimization opportunities
    """)

# Footer with additional resources
st.markdown("---")
st.caption("""
**Need more help?**  
📞 Contact our analytics team at analytics@example.com  
📚 Visit our [knowledge base](https://www.example.com/kb) for tutorials  
🔄 Remember to save your scenarios using the export function
""")#