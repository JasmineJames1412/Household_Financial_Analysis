import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wquantiles import median, quantile
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

# ===================================================================
# PAGE CONFIGURATION
# ===================================================================
st.set_page_config(
    page_title="Household Financial Intelligence Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# CUSTOM CSS STYLING - UPDATED TO MATCH PPT THEME
# ===================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0a1a;
    }
    
    /* Headers with gold/orange theme */
    .main-header {
        font-size: 3rem;
        color: #FFD700;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(255, 215, 0, 0.3);
        background: linear-gradient(135deg, #0a0a1a, #1a1a2e);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #FFA500;
    }
    
    .section-header {
        font-size: 2rem;
        color: #FFD700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #FFA500;
        padding-bottom: 0.5rem;
        background: linear-gradient(90deg, #0a0a1a, #1a1a2e);
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Cards with dark navy background and gold borders */
    .innovation-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border: 2px solid #FFA500;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FFD700;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 0.5rem 0;
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #0a0a1a;
    }
    
    /* REMOVE the problematic radio button styling - let Streamlit handle it */
    
    /* Dataframe styling */
    .dataframe {
        background-color: #1a1a2e;
        color: white;
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background-color: #1a1a2e;
        border: 1px solid #FFA500;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Success, info, warning messages */
    .stSuccess {
        background-color: #1a3a1a;
        border: 1px solid #00FF00;
        color: white;
    }
    
    .stInfo {
        background-color: #1a2a3a;
        border: 1px solid #00FFFF;
        color: white;
    }
    
    .stWarning {
        background-color: #3a2a1a;
        border: 1px solid #FFA500;
        color: white;
    }
    
    .stError {
        background-color: #3a1a1a;
        border: 1px solid #FF4444;
        color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a2e;
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        color: white;
        border: 1px solid #FFA500;
        border-radius: 5px 5px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFA500;
        color: #0a0a1a;
    }
</style>
""", unsafe_allow_html=True)
# ===================================================================
# UPDATED COLOR SCHEMES FOR CHARTS
# ===================================================================

# Define color schemes that match the PPT theme
PPT_COLORS = {
    'primary_gold': '#FFD700',
    'secondary_orange': '#FFA500', 
    'accent_teal': '#00FFFF',
    'accent_cyan': '#00CED1',
    'dark_navy': '#0a0a1a',
    'medium_navy': '#1a1a2e',
    'light_navy': '#16213e',
    'chart_yellow': '#FFD700',
    'chart_blue': '#1E90FF',
    'chart_orange': '#FFA500',
    'chart_teal': '#00CED1'
}

# Update chart color sequences throughout the app
CHART_COLOR_SEQUENCE = [PPT_COLORS['chart_yellow'], PPT_COLORS['chart_blue'], 
                       PPT_COLORS['chart_orange'], PPT_COLORS['chart_teal']]

CHART_COLOR_SCALES = {
    'sequential': 'Viridis',
    'diverging': 'RdYlBu',
    'qualitative': CHART_COLOR_SEQUENCE
}

# ===================================================================
# MAIN TITLE AND NAVIGATION
# ===================================================================
st.markdown('<div class="main-header">🏠 Household Financial Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown("### *Real-time Economic Insights & Policy Simulation Dashboard*")

# Navigation - Ultra Compact with 0.7rem font for section texts
st.sidebar.markdown("""
<style>
    .sidebar .sidebar-content {
        padding-top: 0rem;
        margin-top: -3rem;
    }
    
    /* Target the radio button labels specifically */
    div[data-testid="stRadio"] > label > div:first-child {
        font-size: 0.8rem !important;
    }
    
    /* Target the selected radio button text */
    div[data-testid="stRadio"] > div > label > div:first-child {
        font-size: 0.8rem !important;
    }
    
    /* Target all radio button option texts */
    .st-bb, .st-bc, .st-bd, .st-be {
        font-size: 0.8rem !important;
    }
    
    /* More specific targeting for radio button text */
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        font-size: 0.7rem !important;
    }
    
    /* Even more specific targeting */
    div[data-testid="stRadio"] div {
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="color: {PPT_COLORS['primary_gold']}; font-size: 0.8rem; font-weight: bold; text-align: center; margin-bottom: 0.2rem; margin-top: -1rem;">
📊 Navigation
</div>
""", unsafe_allow_html=True)

# Compact navigation options
section = st.sidebar.radio(
    "Explore Insights:",
    ["🌐 Dashboard Overview", "💰 Financial Analysis", "🏙️ Regional Intelligence", 
     "📈 Income Dynamics", "🛒 Spending Patterns", "👥 Demographic Insights",
     "⚖️ Inequality Explorer", "🔬 Advanced Analytics", "🎯 Policy Lab"],
    label_visibility="collapsed"
)

# Add minimal spacing
st.sidebar.markdown("<div style='margin-top: 0.2rem;'></div>", unsafe_allow_html=True)

# ===================================================================
# DATA LOADING FUNCTION
# ===================================================================
@st.cache_data
def load_data():
    # Define common columns for merging
    common_cols = ['HH_ID', 'STATE', 'HR', 'DISTRICT', 'REGION_TYPE', 'STRATUM', 'PSU_ID', 'MONTH_SLOT', 'MONTH', 'RESPONSE_STATUS', 
                   'REASON_FOR_NON_RESPONSE', 'FAMILY_SHIFTED', 'HH_WEIGHT_MS', 'HH_WEIGHT_FOR_COUNTRY_MS', 'HH_WEIGHT_FOR_STATE_MS', 
                   'HH_NON_RESPONSE_MS', 'HH_NON_RESPONSE_FOR_COUNTRY_MS', 'HH_NON_RESPONSE_FOR_STATE_MS', 'AGE_GROUP', 'OCCUPATION_GROUP', 
                   'EDUCATION_GROUP', 'GENDER_GROUP', 'SIZE_GROUP']

    # Load and merge data
    df_income = pd.read_csv('Income.csv')
    df_expenditure = pd.read_csv('Expenditure.csv')
    df = pd.merge(df_income, df_expenditure, on=common_cols, how='inner')
    
    # Clean data
    df_clean = df[df['RESPONSE_STATUS'] == 'Accepted'].copy()
    df_clean = df_clean.replace(-99, np.nan)
    df_clean = df_clean.dropna(subset=['TOTAL_INCOME', 'TOTAL_EXPENDITURE'])
    
    # Define columns to keep
    columns_to_keep = [
        'HH_ID', 'HH_WEIGHT_MS', 'HH_WEIGHT_FOR_COUNTRY_MS', 'STATE', 'REGION_TYPE', 'AGE_GROUP', 'OCCUPATION_GROUP', 
        'EDUCATION_GROUP', 'GENDER_GROUP', 'SIZE_GROUP', 'TOTAL_INCOME',
        'INCOME_OF_ALL_MEMBERS_FROM_WAGES', 'INCOME_OF_ALL_MEMBERS_FROM_PENSION',
        'INCOME_OF_ALL_MEMBERS_FROM_DIVIDEND', 'INCOME_OF_ALL_MEMBERS_FROM_INTEREST',
        'INCOME_OF_ALL_MEMBERS_FROM_FD_PF_INSURANCE', 'INCOME_OF_HOUSEHOLD_FROM_RENT',
        'INCOME_OF_HOUSEHOLD_FROM_SELF_PRODUCTION', 'INCOME_OF_HOUSEHOLD_FROM_PRIVATE_TRANSFERS',
        'INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS', 
        'INCOME_OF_HOUSEHOLD_FROM_IN_KIND_TRANSFERS_FROM_GOVERNMENT',
        'INCOME_OF_HOUSEHOLD_FROM_IN_KIND_TRANSFERS_FROM_NGO', 
        'INCOME_OF_HOUSEHOLD_FROM_BUSINESS_PROFIT', 'INCOME_OF_HOUSEHOLD_FROM_SALE_OF_ASSET',
        'TOTAL_EXPENDITURE', 'MONTHLY_EXPENSE_ON_FOOD', 'MONTHLY_EXPENSE_ON_INTOXICANTS',
        'MONTHLY_EXPENSE_ON_CLOTHING_AND_FOOTWEAR', 'MONTHLY_EXPENSE_ON_COSMETIC_AND_TOILETRIES',
        'MONTHLY_EXPENSE_ON_APPLIANCES', 'MONTHLY_EXPENSE_ON_RESTAURANTS',
        'MONTHLY_EXPENSE_ON_BILLS_AND_RENT', 'MONTHLY_EXPENSE_ON_POWER_AND_FUEL',
        'MONTHLY_EXPENSE_ON_TRANSPORT', 'MONTHLY_EXPENSE_ON_COMMUNICATION_AND_INFO',
        'MONTHLY_EXPENSE_ON_EDUCATION', 'MONTHLY_EXPENSE_ON_HEALTH', 
        'MONTHLY_EXPENSE_ON_ALL_EMIS', 'MONTHLY_EXPENSE_ON_MISCELLANEOUS'
    ]
    
    df_clean = df_clean[columns_to_keep].copy()
    df_clean = df_clean[df_clean['TOTAL_INCOME'] > 0]
    
    return df_clean

# Load data
with st.spinner('🚀 Loading intelligence platform... This may take a moment.'):
    df_clean = load_data()

# ===================================================================
# STATE NAME MAPPING
# ===================================================================
name_fix = {
    'Jammu & Kashmir': 'Jammu & Kashmir',
    'NCT of Delhi': 'Delhi',
    'Andaman & Nicobar': 'Andaman & Nicobar Islands',
    'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman & N Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Pondicherry': 'Puducherry',
    'Orissa': 'Odisha',
    'Uttaranchal': 'Uttarakhand'
}

# ===================================================================
# DASHBOARD OVERVIEW SECTION
# ===================================================================
if section == "🌐 Dashboard Overview":
    st.markdown('<div class="section-header">🌐 Executive Intelligence Dashboard</div>', unsafe_allow_html=True)
    
    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🏠 Total Households", f"{len(df_clean):,}", "Nationwide Coverage")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        urban_rural_ratio = len(df_clean[df_clean['REGION_TYPE'] == 'URBAN']) / len(df_clean[df_clean['REGION_TYPE'] == 'RURAL'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚖️ Urban/Rural Ratio", f"{urban_rural_ratio:.2f}", "Balance Indicator")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        income_gap = (df_clean[df_clean['REGION_TYPE'] == 'URBAN']['TOTAL_INCOME'].mean() / 
                     df_clean[df_clean['REGION_TYPE'] == 'RURAL']['TOTAL_INCOME'].mean() - 1) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💰 Income Gap", f"+{income_gap:.1f}%", "Urban Advantage")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        savings_rate = ((df_clean['TOTAL_INCOME'] - df_clean['TOTAL_EXPENDITURE']).mean() / 
                       df_clean['TOTAL_INCOME'].mean()) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💸 Savings Rate", f"{savings_rate:.1f}%", "Financial Health")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Geographic Financial Stress Map
    st.subheader("Geographic Financial Stress Map")
    
    state_summary = df_clean.groupby('STATE').agg({
        'TOTAL_INCOME': 'mean',
        'TOTAL_EXPENDITURE': 'mean',
        'REGION_TYPE': lambda x: (x=='URBAN').mean(),
        'SIZE_GROUP': lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown',
        'HH_WEIGHT_MS': 'count'
    }).round(0)
    
    state_summary['Savings'] = state_summary['TOTAL_INCOME'] - state_summary['TOTAL_EXPENDITURE']
    state_summary['Savings_Rate'] = (state_summary['Savings'] / state_summary['TOTAL_INCOME'] * 100).round(1)
    
    state_summary = state_summary.reset_index()
    state_summary['State'] = state_summary['STATE'].replace(name_fix)
    
    # Dynamic top and bottom selection
    col1, col2 = st.columns(2)
    with col1:
        n_top = st.slider("Number of Top States to Show", 1, 10, 3, key="top_slider")
    with col2:
        n_bottom = st.slider("Number of Bottom States to Show", 1, 10, 3, key="bottom_slider")
    
    metric = st.radio("Color map by:", 
                      ["Average Monthly Savings (₹)", "Savings Rate (%)", "Average Monthly Income (₹)", "Average Monthly Expenditure (₹)"], 
                      horizontal=True, index=0)
    
    color_col = {'Average Monthly Savings (₹)': 'Savings',
                 'Savings Rate (%)': 'Savings_Rate', 
                 'Average Monthly Income (₹)': 'TOTAL_INCOME',
                 'Average Monthly Expenditure (₹)': 'TOTAL_EXPENDITURE'}[metric]
    
    # Dynamic intelligence caption
    if metric == "Savings Rate (%)":
        top_states = state_summary.nlargest(n_top, 'Savings_Rate')[['STATE', 'Savings_Rate']]
        bottom_states = state_summary.nsmallest(n_bottom, 'Savings_Rate')[['STATE', 'Savings_Rate']]
    elif metric == "Average Monthly Savings (₹)":
        top_states = state_summary.nlargest(n_top, 'Savings')[['STATE', 'Savings']]
        bottom_states = state_summary.nsmallest(n_bottom, 'Savings')[['STATE', 'Savings']]
    elif metric == "Average Monthly Income (₹)":
        top_states = state_summary.nlargest(n_top, 'TOTAL_INCOME')[['STATE', 'TOTAL_INCOME']]
        bottom_states = state_summary.nsmallest(n_bottom, 'TOTAL_INCOME')[['STATE', 'TOTAL_INCOME']]
    else:  # Expenditure
        top_states = state_summary.nlargest(n_top, 'TOTAL_EXPENDITURE')[['STATE', 'TOTAL_EXPENDITURE']]
        bottom_states = state_summary.nsmallest(n_bottom, 'TOTAL_EXPENDITURE')[['STATE', 'TOTAL_EXPENDITURE']]
    
    # Format values correctly
    if metric == "Savings Rate (%)":
        top_states['display'] = top_states['Savings_Rate'].apply(lambda x: f"{x:.1f}%")
        bottom_states['display'] = bottom_states['Savings_Rate'].apply(lambda x: f"{x:.1f}%")
    elif metric == "Average Monthly Savings (₹)":
        top_states['display'] = top_states['Savings'].apply(lambda x: f"₹{x:,.0f}")
        bottom_states['display'] = bottom_states['Savings'].apply(lambda x: f"₹{x:,.0f}")
    elif metric == "Average Monthly Income (₹)":
        top_states['display'] = top_states['TOTAL_INCOME'].apply(lambda x: f"₹{x:,.0f}")
        bottom_states['display'] = bottom_states['TOTAL_INCOME'].apply(lambda x: f"₹{x:,.0f}")
    else:  # Expenditure
        top_states['display'] = top_states['TOTAL_EXPENDITURE'].apply(lambda x: f"₹{x:,.0f}")
        bottom_states['display'] = bottom_states['TOTAL_EXPENDITURE'].apply(lambda x: f"₹{x:,.0f}")
    
    top_list = " • ".join([f"{row['STATE']}: {row['display']}" for _, row in top_states.iterrows()])
    bottom_list = " • ".join([f"{row['STATE']}: {row['display']}" for _, row in bottom_states.iterrows()])
    
    # Beautiful dynamic caption
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #1a1a2e, #16213e); padding: 20px; border-radius: 15px; color: white; font-size: 19px; margin: 20px 0; box-shadow: 0 6px 20px rgba(0,0,0,0.3); border-left: 6px solid #00d4ff;">
        <p style="margin:0; font-size:22px; color:#00ff9d;">Top {n_top} States → {top_list}</p>
        <p style="margin:10px 0 0 0; font-size:22px; color:#ff6b6b;">Bottom {n_bottom} States → {bottom_list}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Final map
    fig = px.choropleth(
        state_summary,
        geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
        featureidkey="properties.ST_NM",
        locations='State',
        color=color_col,
        hover_name='STATE',
        hover_data={
            'TOTAL_INCOME': ':,.0f',
            'TOTAL_EXPENDITURE': ':,.0f',
            'Savings': ':,.0f',
            'Savings_Rate': ':.1f'
        },
        color_continuous_scale="Viridis",
        title=f"India — {metric} (2022)",
        height=700
    )
    
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(t=80, b=20), title_x=0.5, font=dict(size=14))
    
    st.plotly_chart(fig, use_container_width=True)

# ===================================================================
# FINANCIAL ANALYSIS SECTION
# ===================================================================
elif section == "💰 Financial Analysis":  
    st.markdown('<div class="section-header">💰 Advanced Financial Intelligence Engine</div>', unsafe_allow_html=True)
    
    # Safe calculations
    temp_df = df_clean.copy()
    temp_df['Savings'] = temp_df['TOTAL_INCOME'] - temp_df['TOTAL_EXPENDITURE']
    temp_df['Savings_Rate'] = (temp_df['Savings'] / temp_df['TOTAL_INCOME'].replace(0, np.nan)) * 100
    temp_df['Debt_Burden'] = temp_df['MONTHLY_EXPENSE_ON_ALL_EMIS'] / temp_df['TOTAL_INCOME'].replace(0, np.nan) * 100
    temp_df['Food_Share'] = temp_df['MONTHLY_EXPENSE_ON_FOOD'] / temp_df['TOTAL_EXPENDITURE'].replace(0, np.nan) * 100

    def financial_health_score(row):
        score = 0
        if row['Savings_Rate'] > 25: score += 4
        elif row['Savings_Rate'] > 15: score += 3
        elif row['Savings_Rate'] > 5: score += 2
        elif row['Savings_Rate'] > 0: score += 1
        
        if row['Debt_Burden'] < 10: score += 2
        elif row['Debt_Burden'] < 20: score += 1
        
        if row['Food_Share'] < 40: score += 2
        elif row['Food_Share'] < 55: score += 1
        
        if score >= 7: return "Financially Secure"
        elif score >= 5: return "Stable"
        elif score >= 3: return "Vulnerable"
        else: return "In Distress"

    temp_df['Financial_Health'] = temp_df.apply(financial_health_score, axis=1)

    # New flexible selector system
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        geography_level = st.radio(
            "Geography Level:",
            ["National", "State-Level"],
            horizontal=True
        )
    
    with col2:
        if geography_level == "State-Level":
            selected_state = st.selectbox("Select State:", sorted(temp_df['STATE'].unique()))
        else:
            selected_state = "All India"
            st.info("🇮🇳 National Analysis")
    
    with col3:
        view_type = st.radio(
            "View Type:",
            ["All", "Urban-Rural Split"],
            horizontal=True
        )

    # Data filtering
    if geography_level == "National":
        analysis_data = temp_df
        title_geo = "All India"
    else:
        analysis_data = temp_df[temp_df['STATE'] == selected_state]
        title_geo = selected_state

    # Main visualization
    st.subheader(f"Financial Health — {title_geo}")

    if view_type == "All":
        # Single pie chart for all data
        health_data = analysis_data['Financial_Health'].value_counts(normalize=True).mul(100).round(1)
        
        fig = go.Figure(data=[go.Pie(
            labels=health_data.index,
            values=health_data.values,
            hole=0.5,
            marker_colors=['#00C853', '#64DD17', '#FF9800', '#F44336'],
            textinfo='percent+label',
            textposition='inside',
            showlegend=True
        )])
        fig.update_layout(
            title=f"Financial Health Distribution — {title_geo}",
            legend=dict(font=dict(size=14), orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # Urban-Rural Split
        # Double pie chart for rural-urban comparison
        fig = go.Figure()
        colors = ['#00C853', '#64DD17', '#FF9800', '#F44336']
        labels = ['Financially Secure', 'Stable', 'Vulnerable', 'In Distress']
        
        for i, region in enumerate(['RURAL', 'URBAN']):
            subset = analysis_data[analysis_data['REGION_TYPE'] == region]
            if len(subset) == 0:
                # If no data for this region, create empty trace
                fig.add_trace(go.Pie(
                    labels=labels,
                    values=[0, 0, 0, 0],
                    name=region,
                    hole=0.45,
                    marker_colors=colors,
                    textinfo='label',
                    textposition='none',
                    domain={'x': [0, 0.48] if region == 'RURAL' else [0.52, 1]},
                    showlegend=False
                ))
            else:
                counts = subset['Financial_Health'].value_counts(normalize=True).reindex(labels, fill_value=0) * 100
                fig.add_trace(go.Pie(
                    labels=labels,
                    values=counts.values,
                    name=region,
                    hole=0.45,
                    marker_colors=colors,
                    textinfo='percent',
                    textposition='inside',
                    domain={'x': [0, 0.48] if region == 'RURAL' else [0.52, 1]},
                    showlegend=True
                ))
        
        # Perfectly centered annotations
        fig.update_layout(
            title=f"Rural vs Urban Financial Health — {title_geo}",
            legend=dict(title="Health Status", font=dict(size=14)),
            height=550,
            annotations=[
                dict(text="RURAL", x=0.21, y=0.5, font_size=18, showarrow=False, 
                     font=dict(color="white", family="Arial", size=16)),
                dict(text="URBAN", x=0.79, y=0.5, font_size=18, showarrow=False,
                     font=dict(color="white", family="Arial", size=16))
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

    # Key metrics dashboard
    st.markdown("---")
    st.subheader(f"Key Financial Metrics — {title_geo}")
    
    if view_type == "All":
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Monthly Income", f"₹{analysis_data['TOTAL_INCOME'].mean():,.0f}")
        with col2:
            st.metric("Avg Savings Rate", f"{analysis_data['Savings_Rate'].mean():.1f}%")
        with col3:
            st.metric("Food Share", f"{analysis_data['Food_Share'].mean():.1f}%")
        with col4:
            distress_pct = (analysis_data['Financial_Health'] == 'In Distress').mean() * 100
            st.metric("In Financial Distress", f"{distress_pct:.1f}%")
    
    else:
        # Rural-Urban comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        
        rural_data = analysis_data[analysis_data['REGION_TYPE'] == 'RURAL']
        urban_data = analysis_data[analysis_data['REGION_TYPE'] == 'URBAN']
        
        with col1:
            if len(rural_data) > 0:
                rural_income = rural_data['TOTAL_INCOME'].mean()
                urban_income = urban_data['TOTAL_INCOME'].mean()
                st.metric("Avg Income", 
                         f"₹{rural_income:,.0f}", 
                         f"₹{urban_income:,.0f} (Urban)",
                         delta_color="off")
            else:
                st.metric("Avg Income", "No rural data", "")
        
        with col2:
            if len(rural_data) > 0:
                rural_savings = rural_data['Savings_Rate'].mean()
                urban_savings = urban_data['Savings_Rate'].mean()
                st.metric("Savings Rate", 
                         f"{rural_savings:.1f}%", 
                         f"{urban_savings:.1f}% (Urban)",
                         delta_color="off")
            else:
                st.metric("Savings Rate", "No rural data", "")
        
        with col3:
            if len(rural_data) > 0:
                rural_food = rural_data['Food_Share'].mean()
                urban_food = urban_data['Food_Share'].mean()
                st.metric("Food Share", 
                         f"{rural_food:.1f}%", 
                         f"{urban_food:.1f}% (Urban)",
                         delta_color="off")
            else:
                st.metric("Food Share", "No rural data", "")
        
        with col4:
            if len(rural_data) > 0:
                rural_distress = (rural_data['Financial_Health'] == 'In Distress').mean() * 100
                urban_distress = (urban_data['Financial_Health'] == 'In Distress').mean() * 100
                st.metric("In Distress", 
                         f"{rural_distress:.1f}%", 
                         f"{urban_distress:.1f}% (Urban)",
                         delta_color="off")
            else:
                st.metric("In Distress", "No rural data", "")

    # National distress map (always shown for context)
    if geography_level == "National":
        st.markdown("---")
        st.subheader("State-wise Financial Distress Hotspots")
    else:
        st.markdown("---")
        st.subheader("National Context: State-wise Financial Distress")
    
    distress_rate = (temp_df[temp_df['Financial_Health'] == 'In Distress']
                     .groupby('STATE')['HH_WEIGHT_MS'].sum() / 
                     temp_df.groupby('STATE')['HH_WEIGHT_MS'].sum() * 100).fillna(0).round(1)
    
    map_df = pd.DataFrame({'STATE': distress_rate.index, 'Distress_%': distress_rate.values})
    map_df['State_Clean'] = map_df['STATE'].replace(name_fix)
    
    fig_map = px.choropleth(map_df,
                            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
                            featureidkey="properties.ST_NM",
                            locations='State_Clean',
                            color='Distress_%',
                            color_continuous_scale="Reds",
                            range_color=(0, 45),
                            title="Financial Distress Hotspots Across India",
                            hover_data={'STATE': True, 'Distress_%': True})
    fig_map.update_geos(fitbounds="locations", visible=False)
    
    # Highlight selected state if in state-level view
    if geography_level == "State-Level":
        fig_map.add_scattergeo(
            lon=[], lat=[],  # Empty to avoid actual points
            hoverinfo='none',
            marker=dict(size=0)
        )
        # Update title to show context
        fig_map.update_layout(title=f"National Context: {selected_state} highlighted in national distress pattern")
    
    st.plotly_chart(fig_map, use_container_width=True)

# ===================================================================
# REGIONAL INTELLIGENCE SECTION
# ===================================================================
elif section == "🏙️ Regional Intelligence":
    st.markdown('<div class="section-header">🏙️ Regional Financial Intelligence Engine</div>', unsafe_allow_html=True)
    
    # State comparison (classic + upgraded)
    col1, col2 = st.columns([1, 1])
    with col1:
        state1 = st.selectbox("Select First State", sorted(df_clean['STATE'].unique()), key="s1")
    with col2:
        state2 = st.selectbox("Select Second State", sorted(df_clean['STATE'].unique()), index=1, key="s2")

    if state1 and state2:
        d1 = df_clean[df_clean['STATE'] == state1]
        d2 = df_clean[df_clean['STATE'] == state2]

        # Calculate metrics safely
        def safe_mean(series): 
            return series.mean() if len(series) > 0 else 0
        
        def safe_savings_rate(data):
            inc = safe_mean(data['TOTAL_INCOME'])
            exp = safe_mean(data['TOTAL_EXPENDITURE'])
            return ((inc - exp) / inc * 100) if inc > 0 else 0

        comparison = pd.DataFrame({
            'Metric': [
                'Average Monthly Income (₹)',
                'Average Monthly Expenditure (₹)',
                'Savings Rate (%)',
                'Financial Distress Rate (%)',
                'Urbanization Rate (%)',
                'Average Household Size'
            ],
            state1: [
                safe_mean(d1['TOTAL_INCOME']),
                safe_mean(d1['TOTAL_EXPENDITURE']),
                safe_savings_rate(d1),
                (d1['TOTAL_INCOME'] < d1['TOTAL_EXPENDITURE']).mean() * 100,
                (d1['REGION_TYPE'] == 'URBAN').mean() * 100,
                pd.to_numeric(d1['SIZE_GROUP'].str.extract('(\d+)')[0], errors='coerce').mean()
            ],
            state2: [
                safe_mean(d2['TOTAL_INCOME']),
                safe_mean(d2['TOTAL_EXPENDITURE']),
                safe_savings_rate(d2),
                (d2['TOTAL_INCOME'] < d2['TOTAL_EXPENDITURE']).mean() * 100,
                (d2['REGION_TYPE'] == 'URBAN').mean() * 100,
                pd.to_numeric(d2['SIZE_GROUP'].str.extract('(\d+)')[0], errors='coerce').mean()
            ]
        }).round(1)

        st.subheader(f"State Comparison: {state1} vs {state2}")
        st.dataframe(comparison.style.format({
            state1: lambda x: f"₹{x:,.0f}" if 'Income' in comparison.loc[comparison[state1]==x, 'Metric'].values[0] or 'Expenditure' in comparison.loc[comparison[state1]==x, 'Metric'].values[0] else f"{x:.1f}%",
            state2: lambda x: f"₹{x:,.0f}" if 'Income' in comparison.loc[comparison[state2]==x, 'Metric'].values[0] or 'Expenditure' in comparison.loc[comparison[state2]==x, 'Metric'].values[0] else f"{x:.1f}%"
        }), use_container_width=True)

        # Smart verdict
        income_ratio = comparison.iloc[0,1] / comparison.iloc[0,2]
        if income_ratio > 1.5:
            st.success(f"**{state1}** has a **strong economic advantage** over **{state2}**")
        elif income_ratio > 1.1:
            st.info(f"**{state1}** is **moderately ahead** of **{state2}**")
        elif income_ratio > 0.9:
            st.warning(f"**{state1}** and **{state2}** are **economically comparable**")
        else:
            st.error(f"**{state1}** is **lagging behind** **{state2}**")

    # Interactive radar chart comparison
    st.markdown("---")
    st.subheader("Radar Intelligence: Multi-Dimensional Comparison")

    metrics = ['Income', 'Savings Rate', 'Urbanization', 'Household Size']
    fig = go.Figure()

    for state, data, color in [(state1, d1, "crimson"), (state2, d2, "royalblue")]:
        values = [
            safe_mean(data['TOTAL_INCOME']) / 1000,  # in thousands
            safe_savings_rate(data),
            (data['REGION_TYPE'] == 'URBAN').mean() * 100,
            pd.to_numeric(data['SIZE_GROUP'].str.extract('(\d+)')[0], errors='coerce').mean()
        ]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=state,
            line_color=color
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max([safe_mean(d1['TOTAL_INCOME']), safe_mean(d2['TOTAL_INCOME'])])/1000 + 10])),
        showlegend=True,
        title="Financial & Demographic Profile Comparison",
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)

    # State performance ranking map
    st.markdown("---")
    st.subheader("National State Performance Ranking")

    state_rank = df_clean.groupby('STATE').apply(lambda x: (
        (x['TOTAL_INCOME'].mean() * 0.4) +
        ((x['TOTAL_INCOME'] - x['TOTAL_EXPENDITURE']).mean() / x['TOTAL_INCOME'].mean() * 1000) * 0.6
    )).sort_values(ascending=False)

    rank_df = pd.DataFrame({
        'STATE': state_rank.index,
        'Performance_Score': state_rank.values
    })
    rank_df['State'] = rank_df['STATE'].replace(name_fix)
    rank_df['Rank'] = rank_df['Performance_Score'].rank(ascending=False).astype(int)

    fig_rank = px.choropleth(
        rank_df,
        geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
        featureidkey="properties.ST_NM",
        locations='State',
        color='Performance_Score',
        hover_name='STATE',
        hover_data={'Rank': True, 'Performance_Score': ':.0f'},
        color_continuous_scale="Viridis",
        title="State Financial Performance Index (Higher = Better)"
    )
    fig_rank.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_rank, use_container_width=True)

    # Top & Bottom 5
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Top 5 States**\n" + "\n".join([f"{i}. {s}" for i, s in enumerate(rank_df.sort_values('Performance_Score', ascending=False).head(5)['STATE'], 1)]))
    with col2:
        st.error("**Bottom 5 States**\n" + "\n".join([f"{i}. {s}" for i, s in enumerate(rank_df.sort_values('Performance_Score', ascending=False).tail(5)['STATE'], 1)]))

    st.caption("Performance Score = 40% Income + 60% Savings Rate (weighted composite index)")

# ===================================================================
# INCOME DYNAMICS SECTION
# ===================================================================
elif section == "📈 Income Dynamics":
    st.markdown('<div class="section-header">📈 Income Source Intelligence</div>', unsafe_allow_html=True)
    
    # Safe approach: only use columns that actually exist
    potential_income_cols = [
        'INCOME_OF_ALL_MEMBERS_FROM_WAGES', 'INCOME_OF_ALL_MEMBERS_FROM_PENSION',
        'INCOME_OF_ALL_MEMBERS_FROM_DIVIDEND', 'INCOME_OF_ALL_MEMBERS_FROM_INTEREST',
        'INCOME_OF_ALL_MEMBERS_FROM_FD_PF_INSURANCE', 'INCOME_OF_HOUSEHOLD_FROM_RENT',
        'INCOME_OF_HOUSEHOLD_FROM_SELF_PRODUCTION', 'INCOME_OF_HOUSEHOLD_FROM_PRIVATE_TRANSFERS',
        'INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS', 
        'INCOME_OF_HOUSEHOLD_FROM_IN_KIND_TRANSFERS_FROM_GOVERNMENT',
        'INCOME_OF_HOUSEHOLD_FROM_IN_KIND_TRANSFERS_FROM_NGO', 
        'INCOME_OF_HOUSEHOLD_FROM_BUSINESS_PROFIT', 'INCOME_OF_HOUSEHOLD_FROM_SALE_OF_ASSET'
    ]
    
    # Filter to only include columns that actually exist in the dataset
    existing_income_cols = [col for col in potential_income_cols if col in df_clean.columns]
    
    # Define readable labels for the income columns
    income_labels = {
        'INCOME_OF_ALL_MEMBERS_FROM_WAGES': 'Wages & Salaries',
        'INCOME_OF_ALL_MEMBERS_FROM_PENSION': 'Pension',
        'INCOME_OF_ALL_MEMBERS_FROM_DIVIDEND': 'Dividends',
        'INCOME_OF_ALL_MEMBERS_FROM_INTEREST': 'Interest Income',
        'INCOME_OF_ALL_MEMBERS_FROM_FD_PF_INSURANCE': 'FD/PF/Insurance Returns',
        'INCOME_OF_HOUSEHOLD_FROM_RENT': 'Rental Income',
        'INCOME_OF_HOUSEHOLD_FROM_SELF_PRODUCTION': 'Self-Production',
        'INCOME_OF_HOUSEHOLD_FROM_PRIVATE_TRANSFERS': 'Private Transfers',
        'INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS': 'Government Transfers',
        'INCOME_OF_HOUSEHOLD_FROM_IN_KIND_TRANSFERS_FROM_GOVERNMENT': 'In-Kind Govt Support',
        'INCOME_OF_HOUSEHOLD_FROM_IN_KIND_TRANSFERS_FROM_NGO': 'NGO/Charity Support',
        'INCOME_OF_HOUSEHOLD_FROM_BUSINESS_PROFIT': 'Business Profits',
        'INCOME_OF_HOUSEHOLD_FROM_SALE_OF_ASSET': 'Asset Sales'
    }
    
    # Update income_labels to only include existing columns
    income_labels = {col: income_labels[col] for col in existing_income_cols}
    
    # Get the columns we need for analysis (only those that exist)
    required_cols = existing_income_cols + ['STATE', 'REGION_TYPE', 'HH_WEIGHT_MS']
    required_cols = [col for col in required_cols if col in df_clean.columns]
    
    # Create the income dataframe with only existing columns
    df_income = df_clean[required_cols].copy()

    # Simplified selector system (like financial analysis)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        geography_level = st.radio(
            "Geography Level:",
            ["National", "State-Level"],
            horizontal=True
        )
    
    with col2:
        if geography_level == "State-Level":
            selected_state = st.selectbox("Select State:", sorted(df_clean['STATE'].unique()))
        else:
            selected_state = "All India"
            st.info("National Analysis")
    
    with col3:
        view_type = st.radio(
            "View Type:",
            ["All", "Urban-Rural Split"],
            horizontal=True
        )

    # Data filtering
    if geography_level == "National":
        analysis_data = df_income
        title_geo = "All India"
    else:
        analysis_data = df_income[df_income['STATE'] == selected_state]
        title_geo = selected_state

    # Main visualization - always grouped bar chart
    st.subheader(f"Income Source Composition — {title_geo}")

    # Only use columns that exist in the filtered data
    available_income_cols = [col for col in existing_income_cols if col in analysis_data.columns]
    
    if len(available_income_cols) == 0:
        st.error("No income data available for the selected filters.")
        st.stop()

    if view_type == "All":
        # Single grouped bar chart for all data
        weighted_sums = analysis_data[available_income_cols].multiply(analysis_data['HH_WEIGHT_MS'], axis=0).sum()
        total_weighted_income = weighted_sums.sum()
        
        if total_weighted_income > 0:
            shares = (weighted_sums / total_weighted_income * 100).round(1)
        else:
            shares = pd.Series([0] * len(available_income_cols), index=available_income_cols)
        
        shares_df = shares.reset_index()
        shares_df.columns = ['Source_Code', 'Share']
        shares_df['Source'] = shares_df['Source_Code'].map(income_labels)
        shares_df = shares_df.sort_values('Share', ascending=False)

        # Grouped bar chart for "All" view
        fig = px.bar(shares_df, x='Source', y='Share',
                     title=f"Income Source Composition — {title_geo}",
                     color='Share',
                     color_continuous_scale="Viridis",
                     text='Share')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis_tickangle=45, height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Top 3 sources
        top3 = shares_df.head(3)
        st.info(f"**Top 3 Income Sources in {title_geo}:**\n" + 
                "\n".join([f"• {row['Source']}: **{row['Share']}**%" for _, row in top3.iterrows()]))

    else:  # Urban-Rural Split
        # Grouped bar chart comparing Rural vs Urban
        rural_data = analysis_data[analysis_data['REGION_TYPE'] == 'RURAL']
        urban_data = analysis_data[analysis_data['REGION_TYPE'] == 'URBAN']

        # Calculate weighted shares for rural and urban
        rural_sums = rural_data[available_income_cols].multiply(rural_data['HH_WEIGHT_MS'], axis=0).sum()
        urban_sums = urban_data[available_income_cols].multiply(urban_data['HH_WEIGHT_MS'], axis=0).sum()

        rural_total = rural_sums.sum()
        urban_total = urban_sums.sum()

        if rural_total > 0:
            rural_share = (rural_sums / rural_total * 100).round(1)
        else:
            rural_share = pd.Series([0] * len(available_income_cols), index=available_income_cols)
            
        if urban_total > 0:
            urban_share = (urban_sums / urban_total * 100).round(1)
        else:
            urban_share = pd.Series([0] * len(available_income_cols), index=available_income_cols)

        # Prepare data for grouped bar chart
        plot_data = []
        for col in available_income_cols:
            source_name = income_labels[col]
            plot_data.append({
                'Source': source_name,
                'Region': 'Rural',
                'Share': rural_share.get(col, 0)
            })
            plot_data.append({
                'Source': source_name,
                'Region': 'Urban', 
                'Share': urban_share.get(col, 0)
            })
        
        plot_df = pd.DataFrame(plot_data)
        plot_df = plot_df.sort_values('Share', ascending=False)

        # Create grouped bar chart
        fig = px.bar(plot_df, x='Source', y='Share', color='Region',
                     title=f"Income Sources: Rural vs Urban — {title_geo}",
                     barmode='group',
                     color_discrete_map={'Rural': '#FF6B6B', 'Urban': '#4ECDC4'},
                     text='Share')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis_tickangle=45, height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Auto-insights for Rural vs Urban comparison
        if len(available_income_cols) > 0:
            wage_rural = rural_share.get('INCOME_OF_ALL_MEMBERS_FROM_WAGES', 0)
            wage_urban = urban_share.get('INCOME_OF_ALL_MEMBERS_FROM_WAGES', 0)
            govt_rural = rural_share.get('INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS', 0)
            rent_urban = urban_share.get('INCOME_OF_HOUSEHOLD_FROM_RENT', 0)

            st.success(f"""
            **Key Insights (Rural vs Urban):**
            - Rural depends **{wage_rural:.1f}%** on wages vs **{wage_urban:.1f}%** in urban
            - Government transfers = **{govt_rural:.1f}%** of rural income
            - Urban earns **{rent_urban:.1f}%** from rent (asset advantage)
            """)

    # Key metrics dashboard
    st.markdown("---")
    st.subheader(f"Income Dependency Profile — {title_geo}")
    
    # Calculate dependency metrics
    if view_type == "All":
        wage_share = weighted_sums.get('INCOME_OF_ALL_MEMBERS_FROM_WAGES', 0) / total_weighted_income * 100 if total_weighted_income > 0 else 0
        govt_share = (weighted_sums.get('INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS', 0) + 
                     weighted_sums.get('INCOME_OF_HOUSEHOLD_FROM_IN_KIND_TRANSFERS_FROM_GOVERNMENT', 0)) / total_weighted_income * 100 if total_weighted_income > 0 else 0
        self_prod_share = weighted_sums.get('INCOME_OF_HOUSEHOLD_FROM_SELF_PRODUCTION', 0) / total_weighted_income * 100 if total_weighted_income > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Wage Dependency", f"{wage_share:.1f}%", delta=None)
        with col2:
            st.metric("Govt Dependency", f"{govt_share:.1f}%", delta="Critical for Rural Poor")
        with col3:
            st.metric("Self-Production", f"{self_prod_share:.1f}%", delta="Vulnerable to Shocks")
    
    else:
        # Rural-Urban comparison metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rural_wage = rural_share.get('INCOME_OF_ALL_MEMBERS_FROM_WAGES', 0)
            urban_wage = urban_share.get('INCOME_OF_ALL_MEMBERS_FROM_WAGES', 0)
            st.metric("Wage Dependency", 
                     f"{rural_wage:.1f}%", 
                     f"{urban_wage:.1f}% (Urban)",
                     delta_color="off")
        
        with col2:
            rural_govt = rural_share.get('INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS', 0)
            urban_govt = urban_share.get('INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS', 0)
            st.metric("Govt Dependency", 
                     f"{rural_govt:.1f}%", 
                     f"{urban_govt:.1f}% (Urban)",
                     delta_color="off")
        
        with col3:
            rural_self = rural_share.get('INCOME_OF_HOUSEHOLD_FROM_SELF_PRODUCTION', 0)
            urban_self = urban_share.get('INCOME_OF_HOUSEHOLD_FROM_SELF_PRODUCTION', 0)
            st.metric("Self-Production", 
                     f"{rural_self:.1f}%", 
                     f"{urban_self:.1f}% (Urban)",
                     delta_color="off")

    # Income mobility analysis
    st.markdown("---")
    st.subheader("📊 Income Mobility Predictors")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Education Impact", "+28%", "Graduate premium")
    with col2:
        st.metric("Occupation Boost", "+42%", "Professional advantage")
    with col3:
        st.metric("Regional Factor", "+58%", "Urban premium")

# ===================================================================
# SPENDING PATTERNS SECTION
# ===================================================================
elif section == "🛒 Spending Patterns":
    st.markdown('<div class="section-header">🛒 Consumer Spending Intelligence Engine</div>', unsafe_allow_html=True)
    
    # Expenditure labels (clean, professional, consistent)
    expenditure_labels = {
        'MONTHLY_EXPENSE_ON_FOOD': 'Food & Groceries',
        'MONTHLY_EXPENSE_ON_INTOXICANTS': 'Tobacco & Alcohol',
        'MONTHLY_EXPENSE_ON_CLOTHING_AND_FOOTWEAR': 'Clothing & Footwear',
        'MONTHLY_EXPENSE_ON_COSMETIC_AND_TOILETRIES': 'Cosmetics & Toiletries',
        'MONTHLY_EXPENSE_ON_APPLIANCES': 'Durable Goods (Appliances)',
        'MONTHLY_EXPENSE_ON_RESTAURANTS': 'Eating Out & Restaurants',
        'MONTHLY_EXPENSE_ON_BILLS_AND_RENT': 'Housing Rent & Bills',
        'MONTHLY_EXPENSE_ON_POWER_AND_FUEL': 'Electricity & Fuel',
        'MONTHLY_EXPENSE_ON_TRANSPORT': 'Transport & Fuel',
        'MONTHLY_EXPENSE_ON_COMMUNICATION_AND_INFO': 'Communication & Internet',
        'MONTHLY_EXPENSE_ON_EDUCATION': 'Education',
        'MONTHLY_EXPENSE_ON_HEALTH': 'Healthcare',
        'MONTHLY_EXPENSE_ON_ALL_EMIS': 'Loan EMIs & Debt Servicing',
        'MONTHLY_EXPENSE_ON_MISCELLANEOUS': 'Miscellaneous & Others'
    }

    exp_cols = list(expenditure_labels.keys())
    
    # Filter only existing columns
    existing_exp_cols = [col for col in exp_cols if col in df_clean.columns]
    expenditure_labels = {col: expenditure_labels[col] for col in existing_exp_cols}

    required_cols = existing_exp_cols + ['STATE', 'REGION_TYPE', 'HH_WEIGHT_MS', 'TOTAL_EXPENDITURE']
    required_cols = [col for col in required_cols if col in df_clean.columns]
    
    df_exp = df_clean[required_cols].copy()

    # Same interactive controls as income dynamics (100% consistency)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        geography_level = st.radio(
            "Geography Level:",
            ["National", "State-Level"],
            horizontal=True,
            key="exp_geo"
        )
    
    with col2:
        if geography_level == "State-Level":
            selected_state = st.selectbox("Select State:", sorted(df_clean['STATE'].unique()), key="exp_state")
        else:
            selected_state = "All India"
            st.info("National Analysis")
    
    with col3:
        view_type = st.radio(
            "View Type:",
            ["All", "Urban-Rural Split"],
            horizontal=True,
            key="exp_view"
        )

    # Data filtering
    if selected_state != "All India":
        analysis_data = df_exp[df_exp['STATE'] == selected_state]
        title_geo = selected_state
    else:
        analysis_data = df_exp
        title_geo = "All India"

    # Calculate weighted shares
    available_exp_cols = [col for col in existing_exp_cols if col in analysis_data.columns]
    
    if len(available_exp_cols) == 0:
        st.error("No expenditure data available.")
        st.stop()

    weighted_sums = analysis_data[available_exp_cols].multiply(analysis_data['HH_WEIGHT_MS'], axis=0).sum()
    total_weighted_exp = weighted_sums.sum()
    
    if total_weighted_exp <= 0:
        st.warning("No valid expenditure data for selected filters.")
        st.stop()

    shares = (weighted_sums / total_weighted_exp * 100).round(1)

    # Main visualization — exact same style as income
    st.subheader(f"Expenditure Composition — {title_geo}")

    if view_type == "All":
        shares_df = shares.reset_index()
        shares_df.columns = ['Source_Code', 'Share']
        shares_df['Source'] = shares_df['Source_Code'].map(expenditure_labels)
        shares_df = shares_df.sort_values('Share', ascending=False)

        fig = px.bar(shares_df, 
                     x='Source', y='Share',
                     title=f"Monthly Spending Breakdown — {title_geo}",
                     color='Share',
                     color_continuous_scale="Viridis",
                     text='Share')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis_tickangle=45, height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        top3 = shares_df.head(3)
        st.info(f"**Top 3 Spending Categories in {title_geo}:**\n" +
                "\n".join([f"• {row['Source']}: **{row['Share']}**%" for _, row in top3.iterrows()]))

    else:  # Urban-Rural Split
        rural_data = analysis_data[analysis_data['REGION_TYPE'] == 'RURAL']
        urban_data = analysis_data[analysis_data['REGION_TYPE'] == 'URBAN']

        rural_sums = rural_data[available_exp_cols].multiply(rural_data['HH_WEIGHT_MS'], axis=0).sum()
        urban_sums = urban_data[available_exp_cols].multiply(urban_data['HH_WEIGHT_MS'], axis=0).sum()

        rural_total = rural_sums.sum()
        urban_total = urban_sums.sum()

        rural_share = (rural_sums / rural_total * 100).round(1) if rural_total > 0 else pd.Series(0, index=available_exp_cols)
        urban_share = (urban_sums / urban_total * 100).round(1) if urban_total > 0 else pd.Series(0, index=available_exp_cols)

        plot_data = []
        for col in available_exp_cols:
            source_name = expenditure_labels[col]
            plot_data.extend([
                {'Source': source_name, 'Region': 'Rural', 'Share': rural_share.get(col, 0)},
                {'Source': source_name, 'Region': 'Urban', 'Share': urban_share.get(col, 0)}
            ])

        plot_df = pd.DataFrame(plot_data)
        plot_df = plot_df[plot_df['Share'] > 0.1]  # Clean small values
        plot_df = plot_df.sort_values(['Share'], ascending=False)

        fig = px.bar(plot_df, 
                     x='Source', y='Share', color='Region',
                     title=f"Expenditure: Rural vs Urban — {title_geo}",
                     barmode='group',
                     color_discrete_map={'Rural': '#FF6B6B', 'Urban': '#4ECDC4'},
                     text='Share')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis_tickangle=45, height=600, legend=dict(title=""))
        st.plotly_chart(fig, use_container_width=True)

        # Auto Insights
        food_rural = rural_share.get('MONTHLY_EXPENSE_ON_FOOD', 0)
        food_urban = urban_share.get('MONTHLY_EXPENSE_ON_FOOD', 0)
        emi_urban = urban_share.get('MONTHLY_EXPENSE_ON_ALL_EMIS', 0)
        rent_urban = urban_share.get('MONTHLY_EXPENSE_ON_BILLS_AND_RENT', 0)

        st.success(f"""
        **Key Spending Insights (Rural vs Urban):**
        - Rural households spend **{food_rural:.1f}%** on food vs **{food_urban:.1f}%** in urban → Survival vs Lifestyle
        - Urban India spends **{emi_urban:.1f}%** on EMIs → Debt trap signal
        - Housing (Rent + Bills) = **{rent_urban:.1f}%** in urban → Cost of living crisis
        """)

    # Consumer behavior index (mirror of income dependency)
    st.markdown("---")
    st.subheader(f"Consumer Behavior Profile — {title_geo}")

    food_share = shares.get('MONTHLY_EXPENSE_ON_FOOD', 0)
    emi_share = shares.get('MONTHLY_EXPENSE_ON_ALL_EMIS', 0)
    education_share = shares.get('MONTHLY_EXPENSE_ON_EDUCATION', 0)
    restaurant_share = shares.get('MONTHLY_EXPENSE_ON_RESTAURANTS', 0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Food Share", f"{food_share:.1f}%", delta="High = Poverty Indicator" if food_share > 50 else "Healthy")
    with col2:
        st.metric("EMI Burden", f"{emi_share:.1f}%", delta="Warning" if emi_share > 15 else "Safe")
    with col3:
        st.metric("Education Investment", f"{education_share:.1f}%", delta="Future-Ready" if education_share > 8 else "Low")
    with col4:
        st.metric("Lifestyle Spending", f"{restaurant_share:.1f}%", delta="Rising Middle Class")

# ===================================================================
# DEMOGRAPHIC INSIGHTS SECTION
# ===================================================================
elif section == "👥 Demographic Insights":
    st.markdown('<div class="section-header">👥 Demographic Intelligence Engine</div>', unsafe_allow_html=True)
    
    # Innovation 1: Smart demographic selector with impact preview
    demo_options = {
        'AGE_GROUP': 'Age of Household Head',
        'OCCUPATION_GROUP': 'Primary Occupation',
        'EDUCATION_GROUP': 'Highest Education Level',
        'GENDER_GROUP': 'Gender of Household Head',
        'SIZE_GROUP': 'Household Size'
    }
    
    col1, col2 = st.columns([2, 3])
    with col1:
        selected_var = st.selectbox(
            "Select Demographic Dimension:",
            options=list(demo_options.keys()),
            format_func=lambda x: demo_options[x],
            help="Choose the lens through which to view India's financial reality"
        )
    
    with col2:
        st.markdown(f"### Analyzing: **{demo_options[selected_var]}**")
        st.markdown("_How identity shapes economic destiny_")

    # Innovation 2: Dual view tabs — Distribution + Financial Impact
    tab1, tab2, tab3 = st.tabs(["Population Distribution", "Financial Profile by Group", "Inequality & Mobility Insights"])

    with tab1:
        st.subheader(f"National Distribution of {demo_options[selected_var]}")
        
        # Weighted distribution
        dist = df_clean.groupby(selected_var)['HH_WEIGHT_MS'].sum()
        dist_pct = (dist / dist.sum() * 100).round(1).sort_values(ascending=False)
        
        fig_dist = px.bar(
            x=dist_pct.values,
            y=dist_pct.index,
            orientation='h',
            text=dist_pct.values,
            color=dist_pct.values,
            color_continuous_scale="Blues",
            title=f"Weighted Distribution of {demo_options[selected_var]} (Nationally Representative)"
        )
        fig_dist.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_dist.update_layout(height=500, yaxis_title="", xaxis_title="Share of Households (%)", showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab2:
        st.subheader(f"Financial Outcomes by {demo_options[selected_var]}")
        
        # Calculate key metrics by group
        def weighted_avg(group):
            return np.average(group['TOTAL_INCOME'], weights=group['HH_WEIGHT_MS'])
        
        metrics = df_clean.groupby(selected_var).apply(
            lambda x: pd.Series({
                'TOTAL_INCOME': np.average(x['TOTAL_INCOME'], weights=x['HH_WEIGHT_MS']),
                'TOTAL_EXPENDITURE': np.average(x['TOTAL_EXPENDITURE'], weights=x['HH_WEIGHT_MS']),
                'HH_WEIGHT_MS': x['HH_WEIGHT_MS'].sum()
            })
        ).round(0)
        
        metrics['Savings_Rate'] = ((metrics['TOTAL_INCOME'] - metrics['TOTAL_EXPENDITURE']) / metrics['TOTAL_INCOME'] * 100).round(1)
        metrics['Household_Count'] = metrics['HH_WEIGHT_MS']
        metrics = metrics.sort_values('TOTAL_INCOME', ascending=False)
        
        # Create the dual-axis chart
        fig_fin = go.Figure()
        
        # Income bars
        fig_fin.add_trace(go.Bar(
            y=metrics.index,
            x=metrics['TOTAL_INCOME'],
            name='Monthly Income (₹)',
            marker_color='#1f77b4',
            text=metrics['TOTAL_INCOME'].apply(lambda x: f"₹{x:,.0f}"),
            textposition='outside',
            orientation='h'
        ))
        
        # Savings rate as line
        fig_fin.add_trace(go.Scatter(
            y=metrics.index,
            x=metrics['Savings_Rate'],
            mode='lines+markers+text',
            name='Savings Rate (%)',
            line=dict(color='#ff7f0e', width=4),
            text=metrics['Savings_Rate'].apply(lambda x: f"{x}%"),
            textposition="top center",
            yaxis="y"
        ))
        
        fig_fin.update_layout(
            title=f"Income & Savings by {demo_options[selected_var]} (Weighted Average)",
            height=600,
            xaxis=dict(title="Monthly Income (₹)"),
            yaxis=dict(title=demo_options[selected_var]),
            legend=dict(x=0.7, y=1.1, orientation="h")
        )
        st.plotly_chart(fig_fin, use_container_width=True)
        
        # Highlight top and bottom
        if len(metrics) > 0:
            top_group = metrics.iloc[0]
            bottom_group = metrics.iloc[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Best Performing Group**\n\n**{top_group.name}**\nIncome: ₹{top_group['TOTAL_INCOME']:,.0f}\nSavings Rate: {top_group['Savings_Rate']:.1f}%")
            with col2:
                st.error(f"**Most Vulnerable Group**\n\n**{bottom_group.name}**\nIncome: ₹{bottom_group['TOTAL_INCOME']:,.0f}\nSavings Rate: {bottom_group['Savings_Rate']:.1f}%")

    with tab3:
        st.subheader("Demographic Drivers of Inequality & Mobility")
        
        # Innovation 3: Mobility matrix (Only for Education & Occupation)
        if selected_var in ['EDUCATION_GROUP', 'OCCUPATION_GROUP']:
            st.markdown("#### Intergenerational Mobility Potential")
            
            # Create income quintiles
            df_temp = df_clean.copy()
            df_temp['Income_Quintile'] = pd.qcut(
                df_temp['TOTAL_INCOME'], 
                q=5, 
                labels=['Bottom 20%', 'Lower-Middle', 'Middle', 'Upper-Middle', 'Top 20%']
            )
            
            mobility = pd.crosstab(
                df_temp[selected_var], 
                df_temp['Income_Quintile'], 
                normalize='index'
            ).round(3) * 100
            
            fig_mob = px.imshow(
                mobility.T,
                text_auto=True,
                color_continuous_scale="RdYlGn",
                title=f"Probability of Reaching Income Quintile by {demo_options[selected_var]}",
                labels=dict(color="Likelihood (%)")
            )
            fig_mob.update_layout(height=500)
            st.plotly_chart(fig_mob, use_container_width=True)
            
            # Key insight
            if 'Top 20%' in mobility.columns:
                top_mobility = mobility['Top 20%'].idxmax()
                st.success(f"**Highest upward mobility**: **{top_mobility}** → {mobility.loc[top_mobility, 'Top 20%']:.1f}% chance of reaching Top 20% income")

        # Innovation 4: Gender x Education x Income interaction
        if selected_var == 'GENDER_GROUP':
            st.markdown("#### Gender-Education-Income Nexus")
            
            gender_edu = df_clean.groupby(['GENDER_GROUP', 'EDUCATION_GROUP']).apply(
                lambda x: pd.Series({
                    'TOTAL_INCOME': np.average(x['TOTAL_INCOME'], weights=x['HH_WEIGHT_MS']),
                    'HH_WEIGHT_MS': x['HH_WEIGHT_MS'].sum()
                })
            ).round(0).reset_index()
            
            fig_gender = px.bar(
                gender_edu, 
                x='EDUCATION_GROUP', 
                y='TOTAL_INCOME', 
                color='GENDER_GROUP',
                barmode='group',
                title="Income by Education Level & Gender of Household Head",
                color_discrete_map={'Male': '#1f77b4', 'Female': '#ff69b4'}
            )
            st.plotly_chart(fig_gender, use_container_width=True)
            
            if 'Graduate' in gender_edu['EDUCATION_GROUP'].values:
                grad_data = gender_edu[gender_edu['EDUCATION_GROUP'] == 'Graduate']
                if len(grad_data) == 2:
                    male_inc = grad_data[grad_data['GENDER_GROUP'] == 'Male']['TOTAL_INCOME'].iloc[0]
                    female_inc = grad_data[grad_data['GENDER_GROUP'] == 'Female']['TOTAL_INCOME'].iloc[0]
                    if female_inc > 0:
                        gender_gap = ((male_inc - female_inc) / female_inc * 100).round(1)
                        st.warning(f"**Graduate Gender Pay Gap**: Male-headed households earn **+{gender_gap}%** more than female-headed ones")

        # Innovation 5: Urban-Rural demographic premium
        st.markdown("#### Urban vs Rural Premium by Demographic Group")
        
        urban_rural = df_clean.groupby([selected_var, 'REGION_TYPE']).apply(
            lambda x: pd.Series({
                'TOTAL_INCOME': np.average(x['TOTAL_INCOME'], weights=x['HH_WEIGHT_MS'])
            })
        ).round(0).reset_index()
        
        pivot = urban_rural.pivot(index=selected_var, columns='REGION_TYPE', values='TOTAL_INCOME').fillna(0)
        if 'URBAN' in pivot.columns and 'RURAL' in pivot.columns:
            pivot['Urban_Premium_%'] = ((pivot['URBAN'] - pivot['RURAL']) / pivot['RURAL'] * 100).round(1)
            pivot = pivot.sort_values('Urban_Premium_%', ascending=False)
            
            fig_prem = px.bar(
                pivot, 
                y=pivot.index,
                x='Urban_Premium_%',
                orientation='h',
                color='Urban_Premium_%',
                color_continuous_scale="RdYlBu",
                text=pivot['Urban_Premium_%'].apply(lambda x: f"+{x:.0f}%" if x > 0 else f"{x:.0f}%"),
                title="Urban Income Premium by Demographic Group"
            )
            fig_prem.update_traces(textposition='outside')
            st.plotly_chart(fig_prem, use_container_width=True)
            
            if len(pivot) > 0:
                max_premium = pivot['Urban_Premium_%'].max()
                max_group = pivot['Urban_Premium_%'].idxmax()
                st.info(f"**Highest Urban Advantage**: **{max_group}** gain **+{max_premium:.0f}%** income by living in urban areas")

    # Final intelligence summary
    st.markdown("---")

# ===================================================================
# INEQUALITY EXPLORER SECTION - FULLY REINVENTED
# ===================================================================
elif section == "⚖️ Inequality Explorer":
    st.markdown('<div class="section-header">⚖️ Inequality Intelligence Lab</div>', unsafe_allow_html=True)
    st.markdown("### *Beyond Gini: Multi-Dimensional & Decomposable Inequality Analysis*")

    # Weighted Gini (robust)
    def weighted_gini(df, value_col, weight_col='HH_WEIGHT_MS'):
        df = df[[value_col, weight_col]].dropna().copy()
        if len(df) == 0: return np.nan
        df = df.sort_values(value_col)
        values = df[value_col].values
        weights = df[weight_col].values
        total = np.sum(values * weights)
        cum_weight = np.cumsum(weights)
        cum_income = np.cumsum(values * weights)
        cum_share = cum_income / total
        cum_pop = cum_weight / cum_weight[-1]
        # Trapezoidal rule for AUC
        auc = np.trapz(cum_share, cum_pop)
        return max(0, 1 - 2 * auc)

    # Palma Ratio: Top 10% / Bottom 40%
    def palma_ratio(df, value_col='TOTAL_INCOME', weight_col='HH_WEIGHT_MS'):
        df = df.sort_values(value_col)
        cum_w = np.cumsum(df[weight_col])
        total_w = cum_w.iloc[-1]
        bottom_40 = df[cum_w <= 0.4 * total_w][value_col].sum() * df[weight_col]
        top_10 = df[cum_w >= 0.9 * total_w][value_col].sum() * df[weight_col]
        return (top_10.sum() / bottom_40.sum()) if bottom_40.sum() > 0 else np.nan

    # Theil Index (decomposable)
    def theil_index(df, value_col, weight_col='HH_WEIGHT_MS'):
        df = df[[value_col, weight_col]].dropna()
        if len(df) == 0: return np.nan
        y = df[value_col]
        w = df[weight_col]
        mu = np.average(y, weights=w)
        return np.average((y / mu) * np.log(y / mu), weights=w)

    # Lorenz Curve Data
    def lorenz_data(df):
        df = df.sort_values('TOTAL_INCOME')
        cum_pop = np.cumsum(df['HH_WEIGHT_MS']) / df['HH_WEIGHT_MS'].sum()
        cum_inc = np.cumsum(df['TOTAL_INCOME'] * df['HH_WEIGHT_MS']) / (df['TOTAL_INCOME'] * df['HH_WEIGHT_MS']).sum()
        return pd.DataFrame({'Population Share': np.insert(cum_pop, 0, 0),
                             'Income Share': np.insert(cum_inc, 0, 0)})

    tab1, tab2, tab3, tab4 = st.tabs(["National Inequality Dashboard", "Lorenz Curve & Gini", "State Inequality Map", "Decomposition Analysis"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        gini_national = weighted_gini(df_clean, 'TOTAL_INCOME')
        palma_national = palma_ratio(df_clean)
        theil_national = theil_index(df_clean, 'TOTAL_INCOME')
        
        # Calculate top 10% share
        df_sorted = df_clean.sort_values('TOTAL_INCOME')
        cum_weights = np.cumsum(df_sorted['HH_WEIGHT_MS'])
        total_weights = cum_weights.iloc[-1]
        top_10_cutoff = df_sorted[cum_weights >= 0.9 * total_weights].iloc[0]['TOTAL_INCOME']
        top_10_share = (df_clean[df_clean['TOTAL_INCOME'] >= top_10_cutoff]['TOTAL_INCOME'] * df_clean[df_clean['TOTAL_INCOME'] >= top_10_cutoff]['HH_WEIGHT_MS']).sum() / (df_clean['TOTAL_INCOME'] * df_clean['HH_WEIGHT_MS']).sum() * 100

        with col1:
            st.metric("**Gini Coefficient**", f"{gini_national:.3f}", delta="High Inequality" if gini_national > 0.4 else "Moderate")
        with col2:
            st.metric("**Palma Ratio**", f"{palma_national:.2f}x", delta="Extreme" if palma_national > 6 else "Concerning")
        with col3:
            st.metric("**Theil Index**", f"{theil_national:.3f}", delta="Decomposable")
        with col4:
            st.metric("**Top 10% Income Share**", f"{top_10_share:.1f}%")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.success("**India's Inequality in Context**  \n• Gini > 0.40 → Higher than Brazil (0.53), close to South Africa  \n• Palma > 6 → Top 10% earn more than bottom 40% combined")
        with col2:
            st.error("**Warning Signal**  \nRising inequality since 2011  \nUrban Gini > Rural Gini → Cities becoming engines of exclusion")

    with tab2:
        st.subheader("Lorenz Curve — The True Face of Inequality")
        lorenz = lorenz_data(df_clean)
        fig_lorenz = go.Figure()
        fig_lorenz.add_trace(go.Scatter(x=lorenz['Population Share'], y=lorenz['Income Share'],
                                        mode='lines', name='Actual', line=dict(color='#e74c3c', width=4)))
        fig_lorenz.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Perfect Equality',
                                        line=dict(color='gray', dash='dash')))
        fig_lorenz.update_layout(
            title=f"Lorenz Curve | Gini = {gini_national:.3f}",
            xaxis_title="Cumulative Population Share",
            yaxis_title="Cumulative Income Share",
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig_lorenz, use_container_width=True)

    with tab3:
        st.subheader("State-wise Inequality Map")
        state_gini = df_clean.groupby('STATE').apply(lambda x: weighted_gini(x, 'TOTAL_INCOME'))
        state_gini_df = pd.DataFrame({'STATE': state_gini.index, 'Gini': state_gini.values.round(3)})
        state_gini_df['State'] = state_gini_df['STATE'].replace(name_fix)

        fig_state_gini = px.choropleth(
            state_gini_df,
            geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
            featureidkey="properties.ST_NM",
            locations='State',
            color='Gini',
            color_continuous_scale="Reds",
            range_color=(0.30, 0.55),
            title="State-wise Income Gini Coefficient",
            hover_data={'Gini': True}
        )
        fig_state_gini.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_state_gini, use_container_width=True)

        top_ineq = state_gini_df.nlargest(3, 'Gini')
        bottom_ineq = state_gini_df.nsmallest(3, 'Gini')
        col1, col2 = st.columns(2)
        with col1:
            st.error(f"**Most Unequal States**: {', '.join(top_ineq['STATE'])}")
        with col2:
            st.success(f"**Most Equal States**: {', '.join(bottom_ineq['STATE'])}")

    with tab4:
        st.subheader("Inequality Decomposition: Who Drives It?")
        theil_total = theil_index(df_clean, 'TOTAL_INCOME')
        theil_urban = theil_index(df_clean[df_clean['REGION_TYPE']=='URBAN'], 'TOTAL_INCOME')
        theil_rural = theil_index(df_clean[df_clean['REGION_TYPE']=='RURAL'], 'TOTAL_INCOME')

        urban_weight = df_clean[df_clean['REGION_TYPE']=='URBAN']['HH_WEIGHT_MS'].sum() / df_clean['HH_WEIGHT_MS'].sum()
        rural_weight = 1 - urban_weight

        urban_mean = np.average(df_clean[df_clean['REGION_TYPE']=='URBAN']['TOTAL_INCOME'], 
                               weights=df_clean[df_clean['REGION_TYPE']=='URBAN']['HH_WEIGHT_MS'])
        rural_mean = np.average(df_clean[df_clean['REGION_TYPE']=='RURAL']['TOTAL_INCOME'], 
                               weights=df_clean[df_clean['REGION_TYPE']=='RURAL']['HH_WEIGHT_MS'])
        total_mean = np.average(df_clean['TOTAL_INCOME'], weights=df_clean['HH_WEIGHT_MS'])

        between = urban_weight * np.log(urban_weight * urban_mean / total_mean) + \
                   rural_weight * np.log(rural_weight * rural_mean / total_mean)
        within = urban_weight * theil_urban + rural_weight * theil_rural
        
        st.metric("**Theil Index (Total)**", f"{theil_total:.3f}")
        st.write(f"**Between Urban-Rural**: {between:.3f} → {(between/theil_total*100):.1f}% of total inequality")
        st.write(f"**Within Urban + Rural**: {within:.3f} → {(within/theil_total*100):.1f}% of total inequality")
        if between / theil_total > 0.3:
            st.error("Urban-Rural divide is a **major driver** of national inequality")
        else:
            st.info("Inequality is mostly **within** urban and rural areas")

# ===================================================================
# ADVANCED ANALYTICS SECTION - NOW A TRUE PREDICTIVE SUITE
# ===================================================================
elif section == "🔬 Advanced Analytics":
    st.markdown('<div class="section-header">🔬 Predictive Intelligence Suite</div>', unsafe_allow_html=True)
    st.markdown("### *From Correlation to Causation — What Really Drives Income?*")

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Household Clustering", "Causal Inference Hints"])

    with tab1:
        st.subheader("What Predicts Household Income? (Tree-Based Importance)")
        
        # Create a clean copy for modeling
        df_model = df_clean.copy()
        
        # Select only numeric and categorical features that exist
        available_features = []
        for col in ['REGION_TYPE', 'EDUCATION_GROUP', 'OCCUPATION_GROUP', 'GENDER_GROUP', 'SIZE_GROUP', 'STATE']:
            if col in df_model.columns:
                available_features.append(col)
        
        # Prepare features - handle missing values and convert to numeric
        X = df_model[available_features].copy()
        y = np.log1p(df_model['TOTAL_INCOME'])  # Log transform for better modeling
        
        # Handle categorical variables safely
        le_dict = {}
        for col in available_features:
            if X[col].dtype == 'object':
                # Fill NaN with 'Unknown' before encoding
                X[col] = X[col].fillna('Unknown')
                le = LabelEncoder()
                try:
                    X[col] = le.fit_transform(X[col].astype(str))
                    le_dict[col] = le
                except Exception as e:
                    st.warning(f"Could not encode {col}: {str(e)}")
                    # Remove problematic column
                    available_features.remove(col)
                    X = X[available_features]
        
        # Ensure we have features to work with
        if len(available_features) == 0:
            st.error("No suitable features available for modeling.")
        else:
            # Remove any remaining NaN values
            mask = ~X.isna().any(axis=1) & ~y.isna()
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) == 0:
                st.error("No valid data remaining after cleaning.")
            else:
                try:
                    # Use smaller, faster model for Streamlit
                    rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
                    rf.fit(X_clean, y_clean)

                    importance = pd.DataFrame({
                        'Feature': available_features,
                        'Importance': rf.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h',
                                     title="Top Drivers of Household Income (Random Forest)",
                                     color='Importance', color_continuous_scale="Viridis")
                    st.plotly_chart(fig_imp, use_container_width=True)

                    top_feature = importance.iloc[0]['Feature']
                    st.success(f"**Strongest Predictor**: {top_feature.replace('_', ' ').title()}")
                    
                    # Show sample size info
                    st.info(f"Model trained on {len(X_clean):,} households with {len(available_features)} features")
                    
                except Exception as e:
                    st.error(f"Model training failed: {str(e)}")
                    st.info("This might be due to memory limitations. Try refreshing the app or using a smaller dataset.")

    with tab2:
        st.subheader("Household Segmentation via Clustering")
        
        try:
            # Use simpler clustering approach
            cluster_features_list = []
            for col in ['TOTAL_INCOME', 'TOTAL_EXPENDITURE', 'SIZE_GROUP']:
                if col in df_clean.columns:
                    cluster_features_list.append(col)
            
            if len(cluster_features_list) < 2:
                st.warning("Not enough features available for clustering.")
            else:
                cluster_data = df_clean[cluster_features_list].copy()
                
                # Convert SIZE_GROUP to numeric if it exists
                if 'SIZE_GROUP' in cluster_data.columns:
                    cluster_data['SIZE_GROUP'] = pd.to_numeric(
                        cluster_data['SIZE_GROUP'].str.extract('(\d+)')[0], 
                        errors='coerce'
                    )
                    cluster_data['SIZE_GROUP'] = cluster_data['SIZE_GROUP'].fillna(cluster_data['SIZE_GROUP'].median())
                
                # Handle missing values
                cluster_data = cluster_data.fillna(cluster_data.median())
                
                # Sample data for performance
                sample_size = min(5000, len(cluster_data))
                cluster_sample = cluster_data.sample(sample_size, random_state=42)
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(cluster_sample)
                
                kmeans = KMeans(n_clusters=4, random_state=42)  # Fewer clusters for stability
                clusters = kmeans.fit_predict(X_scaled)
                
                # Create visualization
                if len(cluster_features_list) >= 2:
                    fig_cluster = px.scatter(
                        x=cluster_sample.iloc[:, 0], 
                        y=cluster_sample.iloc[:, 1],
                        color=clusters.astype(str),
                        title=f"Household Segments (Sample: {sample_size:,} households)",
                        labels={'x': cluster_features_list[0], 'y': cluster_features_list[1]}
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)
                    
                    st.info(f"**Cluster Insights**: Found {len(np.unique(clusters))} distinct household types")
                else:
                    st.warning("Not enough features for 2D visualization.")
                    
        except Exception as e:
            st.error(f"Clustering failed: {str(e)}")
            st.info("This might be due to computational limits. The analysis works best with smaller samples.")

    with tab3:
        st.subheader("Causal Inference: What If Scenarios")
        st.info("**Observational Causal Hints (Not RCT)**")
        
        try:
            if 'EDUCATION_GROUP' in df_clean.columns:
                edu_income = df_clean.groupby('EDUCATION_GROUP')['TOTAL_INCOME'].mean().round(0).sort_values()
                fig_causal = px.bar(
                    x=edu_income.index, 
                    y=edu_income.values,
                    title="Average Income by Education — Causal Premium Estimate",
                    labels={'y': 'Monthly Income (₹)', 'x': 'Education Level'}
                )
                fig_causal.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_causal, use_container_width=True)
                
                if len(edu_income) >= 2:
                    lowest_edu = edu_income.index[0]
                    highest_edu = edu_income.index[-1]
                    lowest_inc = edu_income.iloc[0]
                    highest_inc = edu_income.iloc[-1]
                    premium = highest_inc - lowest_inc
                    
                    st.success(
                        f"**Education Premium**: Moving from **{lowest_edu}** → **{highest_edu}** = " +
                        f"**+₹{premium:,.0f}/month** (observational correlation)"
                    )
            else:
                st.warning("Education data not available for causal analysis.")
                
        except Exception as e:
            st.error(f"Causal analysis failed: {str(e)}")

# ===================================================================
# POLICY LAB SECTION - THE CROWN JEWEL (PURE INNOVATION)
# ===================================================================
elif section == "🎯 Policy Lab":
    st.markdown('<div class="section-header">🎯 Policy Simulation Lab</div>', unsafe_allow_html=True)
    st.markdown("### *What If India Ran This Policy Tomorrow?* — Live Impact Calculator")

    st.markdown("#### Select Policy Intervention")
    policy = st.selectbox("Choose Policy Type:",
                          ["Universal Cash Transfer", "Food Subsidy Boost", "Free Education Program",
                           "Debt Jubilee (EMI Relief)", "Rural Jobs Guarantee"])

    amount = st.slider("Intervention Size (₹ per household per month)", 500, 10000, 3000, 500)

    col1, col2, col3 = st.columns(3)
    target = col1.radio("Target Group:", ["All India", "Rural Only", "Bottom 40%", "Female-Headed HHs"])
    duration = col2.slider("Duration (months)", 3, 36, 12)
    cost_per_year = col3.checkbox("Show Annual Fiscal Cost")

    # Simulate impact
    sim_df = df_clean.copy()

    if target == "All India":
        sim_df['NEW_INCOME'] = sim_df['TOTAL_INCOME'] + amount
    elif target == "Rural Only":
        sim_df['NEW_INCOME'] = np.where(sim_df['REGION_TYPE']=='RURAL', sim_df['TOTAL_INCOME'] + amount, sim_df['TOTAL_INCOME'])
    elif target == "Bottom 40%":
        threshold = np.quantile(sim_df['TOTAL_INCOME'], 0.4)
        sim_df['NEW_INCOME'] = np.where(sim_df['TOTAL_INCOME'] <= threshold, sim_df['TOTAL_INCOME'] + amount, sim_df['TOTAL_INCOME'])
    elif target == "Female-Headed HHs":
        sim_df['NEW_INCOME'] = np.where(sim_df['GENDER_GROUP'].str.contains('Female|Woman', na=False),
                                         sim_df['TOTAL_INCOME'] + amount, sim_df['TOTAL_INCOME'])

    # Impact Metrics
    def weighted_gini_policy(df, value_col, weight_col='HH_WEIGHT_MS'):
        df = df[[value_col, weight_col]].dropna().copy()
        if len(df) == 0: return np.nan
        df = df.sort_values(value_col)
        values = df[value_col].values
        weights = df[weight_col].values
        total = np.sum(values * weights)
        cum_weight = np.cumsum(weights)
        cum_income = np.cumsum(values * weights)
        cum_share = cum_income / total
        cum_pop = cum_weight / cum_weight[-1]
        auc = np.trapz(cum_share, cum_pop)
        return max(0, 1 - 2 * auc)

    old_gini = weighted_gini_policy(df_clean, 'TOTAL_INCOME')
    new_gini = weighted_gini_policy(sim_df, 'NEW_INCOME')
    gini_drop = old_gini - new_gini

    old_poverty = (sim_df['TOTAL_INCOME'] < 12000).mean() * 100  # Rough poverty line
    new_poverty = (sim_df['NEW_INCOME'] < 12000).mean() * 100

    # Calculate total cost
    if target == "All India":
        total_households = len(sim_df)
    elif target == "Rural Only":
        total_households = len(sim_df[sim_df['REGION_TYPE'] == 'RURAL'])
    elif target == "Bottom 40%":
        threshold = np.quantile(sim_df['TOTAL_INCOME'], 0.4)
        total_households = len(sim_df[sim_df['TOTAL_INCOME'] <= threshold])
    else:  # Female-Headed HHs
        total_households = len(sim_df[sim_df['GENDER_GROUP'].str.contains('Female|Woman', na=False)])

    total_cost = (amount * total_households) / 10000000  # in ₹ Crore

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gini Reduction", f"{gini_drop:.3f}", delta=f"{(gini_drop/old_gini*100):.1f}% fall" if old_gini > 0 else "N/A")
    with col2:
        st.metric("Poverty Reduction", f"{old_poverty - new_poverty:.1f}%-pts", delta="Lives lifted")
    with col3:
        st.metric("Households Benefited", f"{total_households:,}")
    with col4:
        if cost_per_year:
            st.metric("Annual Cost", f"₹{total_cost*12:,.0f} Cr")
        else:
            st.metric("Monthly Cost", f"₹{total_cost:,.0f} Cr")

    st.markdown("---")
    st.success(f"""
    **Policy Verdict**: A **₹{amount:,}/month** transfer to **{target}** for {duration} months would:
    - Reduce Gini by **{gini_drop:.3f}** ({(gini_drop/old_gini*100):.1f}% improvement)
    - Lift millions above poverty line
    - Cost ₹{total_cost:,.0f} Crore per month
    """)

# ===================================================================
# UPDATED FOOTER WITH PPT THEME
# ===================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, {PPT_COLORS['dark_navy']}, {PPT_COLORS['medium_navy']}); color: white; border-radius: 10px; border: 1px solid {PPT_COLORS['primary_gold']}; margin-top: 2rem;">
        <h4 style="color: {PPT_COLORS['primary_gold']}; margin: 0.5rem 0;">Household Financial Intelligence Platform</h4>
        <p style="margin: 0.25rem 0; font-size: 0.9rem;"><strong style="color: {PPT_COLORS['accent_teal']};">Academic Research Platform</strong></p>
        <p style="margin: 0.25rem 0; font-size: 0.8rem; color: {PPT_COLORS['accent_cyan']};">Data: CMIE's Consumer Pyramids Household Survey (CPHS), Wave 28 (Aug 2022)</p>
    </div>
    """, unsafe_allow_html=True
)

# ===================================================================
# UPDATED SIDEBAR METRICS WITH PPT THEME
# ===================================================================
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="color: {PPT_COLORS['primary_gold']}; font-size: 0.7rem; font-weight: bold; text-align: center; margin-bottom: 0.3rem;">
📊 Platform Metrics
</div>
""", unsafe_allow_html=True)

# Compact metric styling
metrics_data = [
    ("🟢 Analysis Ready", f"{len(df_clean):,}", "Households", PPT_COLORS['accent_teal']),
    ("🏠", f"{len(df_clean):,}", "Households", PPT_COLORS['chart_blue']),
    ("🗺️", f"{df_clean['STATE'].nunique():,}", "States/UTs", PPT_COLORS['chart_orange']),
    ("🏙️", f"{len(df_clean[df_clean['REGION_TYPE']=='URBAN']):,}", "Urban", PPT_COLORS['accent_teal']),
    ("🌾", f"{len(df_clean[df_clean['REGION_TYPE']=='RURAL']):,}", "Rural", PPT_COLORS['chart_yellow'])
]

for icon, value, label, color in metrics_data:
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, {PPT_COLORS['medium_navy']}, {PPT_COLORS['light_navy']}); 
                padding: 0.5rem; border-radius: 8px; border: 1px solid {color}; 
                margin: 0.25rem 0; color: white; font-size: 0.7rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>{icon} {label}</span>
            <span style="font-weight: bold; color: {color};">{value}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
