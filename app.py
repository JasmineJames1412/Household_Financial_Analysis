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
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Household Financial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">🏠 Household Financial Analysis: Income & Expenditure Trends in India</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["Data Overview", "Income & Expenditure Analysis", "Regional Comparison", 
     "Income Sources Breakdown", "Expenditure Categories", "Demographic Analysis",
     "Inequality Measures", "Statistical Analysis"]
)

# Load data function with caching
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
with st.spinner('Loading and processing data... This may take a moment.'):
    df_clean = load_data()

# Data Overview Section
if section == "Data Overview":
    st.markdown('<div class="section-header">📈 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Households", f"{len(df_clean):,}")
    with col2:
        st.metric("Rural Households", f"{len(df_clean[df_clean['REGION_TYPE'] == 'RURAL']):,}")
    with col3:
        st.metric("Urban Households", f"{len(df_clean[df_clean['REGION_TYPE'] == 'URBAN']):,}")
    
    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df_clean.head(10))
    
    # Basic statistics
    st.subheader("Basic Statistics")
    st.dataframe(df_clean[['TOTAL_INCOME', 'TOTAL_EXPENDITURE']].describe())
    
    # Column information
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Columns:** {len(df_clean.columns)}")
        st.write(f"**Numeric Columns:** {len(df_clean.select_dtypes(include=[np.number]).columns)}")
        st.write(f"**Categorical Columns:** {len(df_clean.select_dtypes(include=['object']).columns)}")
    with col2:
        st.write(f"**Memory Usage:** {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Income & Expenditure Analysis
elif section == "Income & Expenditure Analysis":
    st.markdown('<div class="section-header">💰 Income & Expenditure Analysis</div>', unsafe_allow_html=True)
    
    # Weighted statistics function
    def weighted_stats(df, value_col, weight_col):
        weighted_mean = np.average(df[value_col], weights=df[weight_col])
        weighted_median = median(df[value_col], df[weight_col])
        weighted_std = np.sqrt(np.average((df[value_col] - weighted_mean) ** 2, weights=df[weight_col]))
        return weighted_mean, weighted_median, weighted_std
    
    # Overall statistics
    income_mean, income_median, income_std = weighted_stats(df_clean, 'TOTAL_INCOME', 'HH_WEIGHT_MS')
    exp_mean, exp_median, exp_std = weighted_stats(df_clean, 'TOTAL_EXPENDITURE', 'HH_WEIGHT_MS')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income Statistics")
        st.metric("Mean Income", f"₹{income_mean:,.0f}")
        st.metric("Median Income", f"₹{income_median:,.0f}")
        st.metric("Standard Deviation", f"₹{income_std:,.0f}")
    
    with col2:
        st.subheader("Expenditure Statistics")
        st.metric("Mean Expenditure", f"₹{exp_mean:,.0f}")
        st.metric("Median Expenditure", f"₹{exp_median:,.0f}")
        st.metric("Standard Deviation", f"₹{exp_std:,.0f}")
    
    # Distribution plots
    st.subheader("Distribution of Income and Expenditure")
    
    # Calculate key stats
    mean_income = df_clean['TOTAL_INCOME'].mean()
    median_income = df_clean['TOTAL_INCOME'].median()
    mean_exp = df_clean['TOTAL_EXPENDITURE'].mean()
    median_exp = df_clean['TOTAL_EXPENDITURE'].median()
    
    # Set visualization range (98th percentile)
    income_x_max = df_clean['TOTAL_INCOME'].quantile(0.98)
    exp_x_max = df_clean['TOTAL_EXPENDITURE'].quantile(0.98)
    
    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Income Distribution', 'Expenditure Distribution'))
    
    # Income histogram
    fig.add_trace(go.Histogram(x=df_clean['TOTAL_INCOME'], name='Income', 
                              marker_color='dodgerblue', opacity=0.7), row=1, col=1)
    # Expenditure histogram
    fig.add_trace(go.Histogram(x=df_clean['TOTAL_EXPENDITURE'], name='Expenditure', 
                              marker_color='gold', opacity=0.7), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        width=1000,
        height=500,
        showlegend=False,
        plot_bgcolor='white'
    )
    fig.update_xaxes(range=[0, income_x_max], row=1, col=1, title_text="Income (₹)")
    fig.update_xaxes(range=[0, exp_x_max], row=1, col=2, title_text="Expenditure (₹)")
    
    st.plotly_chart(fig, use_container_width=True)

# Regional Comparison
elif section == "Regional Comparison":
    st.markdown('<div class="section-header">🏙️ Regional Comparison</div>', unsafe_allow_html=True)
    
    # Rural vs Urban comparison
    rural_income = df_clean[df_clean['REGION_TYPE'] == 'RURAL']['TOTAL_INCOME']
    urban_income = df_clean[df_clean['REGION_TYPE'] == 'URBAN']['TOTAL_INCOME']
    rural_exp = df_clean[df_clean['REGION_TYPE'] == 'RURAL']['TOTAL_EXPENDITURE']
    urban_exp = df_clean[df_clean['REGION_TYPE'] == 'URBAN']['TOTAL_EXPENDITURE']
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income Comparison")
        st.metric("Rural Mean Income", f"₹{rural_income.mean():,.0f}")
        st.metric("Urban Mean Income", f"₹{urban_income.mean():,.0f}")
        st.metric("Difference", f"₹{urban_income.mean() - rural_income.mean():,.0f}")
    
    with col2:
        st.subheader("Expenditure Comparison")
        st.metric("Rural Mean Expenditure", f"₹{rural_exp.mean():,.0f}")
        st.metric("Urban Mean Expenditure", f"₹{urban_exp.mean():,.0f}")
        st.metric("Difference", f"₹{urban_exp.mean() - rural_exp.mean():,.0f}")
    
    # Interactive bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Rural Income', x=['Rural'], y=[rural_income.mean()], marker_color='blue'))
    fig.add_trace(go.Bar(name='Urban Income', x=['Urban'], y=[urban_income.mean()], marker_color='lightblue'))
    fig.add_trace(go.Bar(name='Rural Expenditure', x=['Rural'], y=[rural_exp.mean()], marker_color='darkorange'))
    fig.add_trace(go.Bar(name='Urban Expenditure', x=['Urban'], y=[urban_exp.mean()], marker_color='orange'))
    
    fig.update_layout(
        title="Average Income and Expenditure by Region",
        xaxis_title="Region",
        yaxis_title="Amount (₹)",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Income Sources Breakdown
elif section == "Income Sources Breakdown":
    st.markdown('<div class="section-header">💼 Income Sources Breakdown</div>', unsafe_allow_html=True)
    
    # Define income columns
    income_cols = [col for col in df_clean.columns if col.startswith('INCOME_OF_')]
    specific_cols = [col for col in income_cols if 'ALL_SOURCES' not in col]
    
    # Calculate total income by source
    income_sums = df_clean[specific_cols].sum()
    total_income = income_sums.sum()
    percentages = (income_sums / total_income) * 100
    
    # Short labels for display
    short_labels = {
        'INCOME_OF_ALL_MEMBERS_FROM_WAGES': 'Wages',
        'INCOME_OF_ALL_MEMBERS_FROM_PENSION': 'Pension',
        'INCOME_OF_ALL_MEMBERS_FROM_DIVIDEND': 'Dividend',
        'INCOME_OF_ALL_MEMBERS_FROM_INTEREST': 'Interest',
        'INCOME_OF_ALL_MEMBERS_FROM_FD_PF_INSURANCE': 'FD/PF/Insurance',
        'INCOME_OF_HOUSEHOLD_FROM_RENT': 'Rent',
        'INCOME_OF_HOUSEHOLD_FROM_SELF_PRODUCTION': 'Self-Production',
        'INCOME_OF_HOUSEHOLD_FROM_PRIVATE_TRANSFERS': 'Private Transfers',
        'INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS': 'Govt Transfers',
        'INCOME_OF_HOUSEHOLD_FROM_BUSINESS_PROFIT': 'Business Profit',
        'INCOME_OF_HOUSEHOLD_FROM_SALE_OF_ASSET': 'Asset Sale'
    }
    
    labels = [short_labels.get(col, col) for col in specific_cols]
    
    # Create pie chart
    fig = px.pie(
        values=percentages.values, 
        names=labels,
        title="Income Sources Distribution",
        hover_data=[income_sums.values],
        labels={'value': 'Percentage', 'hover_data_0': 'Total Income'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display table
    income_table = pd.DataFrame({
        'Income Source': labels,
        'Total Income (₹)': income_sums.values,
        'Percentage': percentages.values
    }).sort_values('Percentage', ascending=False)
    
    st.subheader("Income Sources Details")
    st.dataframe(income_table)

# Expenditure Categories
elif section == "Expenditure Categories":
    st.markdown('<div class="section-header">🛒 Expenditure Categories</div>', unsafe_allow_html=True)
    
    # Define expenditure columns
    exp_cols = [
        'MONTHLY_EXPENSE_ON_FOOD', 'MONTHLY_EXPENSE_ON_INTOXICANTS', 'MONTHLY_EXPENSE_ON_CLOTHING_AND_FOOTWEAR',
        'MONTHLY_EXPENSE_ON_COSMETIC_AND_TOILETRIES', 'MONTHLY_EXPENSE_ON_APPLIANCES', 'MONTHLY_EXPENSE_ON_RESTAURANTS',
        'MONTHLY_EXPENSE_ON_BILLS_AND_RENT', 'MONTHLY_EXPENSE_ON_POWER_AND_FUEL', 'MONTHLY_EXPENSE_ON_TRANSPORT',
        'MONTHLY_EXPENSE_ON_COMMUNICATION_AND_INFO', 'MONTHLY_EXPENSE_ON_EDUCATION', 'MONTHLY_EXPENSE_ON_HEALTH',
        'MONTHLY_EXPENSE_ON_ALL_EMIS', 'MONTHLY_EXPENSE_ON_MISCELLANEOUS'
    ]
    
    exp_sums = df_clean[exp_cols].sum()
    total_expense = exp_sums.sum()
    percentages = (exp_sums / total_expense) * 100
    
    # Short labels
    short_labels = {
        'MONTHLY_EXPENSE_ON_FOOD': 'Food',
        'MONTHLY_EXPENSE_ON_INTOXICANTS': 'Intoxicants',
        'MONTHLY_EXPENSE_ON_CLOTHING_AND_FOOTWEAR': 'Clothing/Footwear',
        'MONTHLY_EXPENSE_ON_COSMETIC_AND_TOILETRIES': 'Cosmetics/Toiletries',
        'MONTHLY_EXPENSE_ON_APPLIANCES': 'Appliances',
        'MONTHLY_EXPENSE_ON_RESTAURANTS': 'Restaurants',
        'MONTHLY_EXPENSE_ON_BILLS_AND_RENT': 'Bills/Rent',
        'MONTHLY_EXPENSE_ON_POWER_AND_FUEL': 'Power/Fuel',
        'MONTHLY_EXPENSE_ON_TRANSPORT': 'Transport',
        'MONTHLY_EXPENSE_ON_COMMUNICATION_AND_INFO': 'Communication',
        'MONTHLY_EXPENSE_ON_EDUCATION': 'Education',
        'MONTHLY_EXPENSE_ON_HEALTH': 'Health',
        'MONTHLY_EXPENSE_ON_ALL_EMIS': 'EMIs',
        'MONTHLY_EXPENSE_ON_MISCELLANEOUS': 'Miscellaneous'
    }
    
    labels = [short_labels.get(col, col) for col in exp_cols]
    
    # Create bar chart
    fig = px.bar(
        x=labels, y=percentages.values,
        title="Expenditure Distribution by Category",
        labels={'x': 'Expenditure Category', 'y': 'Percentage (%)'},
        color=percentages.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(xaxis_tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display table
    exp_table = pd.DataFrame({
        'Expenditure Category': labels,
        'Total Expenditure (₹)': exp_sums.values,
        'Percentage': percentages.values
    }).sort_values('Percentage', ascending=False)
    
    st.subheader("Expenditure Categories Details")
    st.dataframe(exp_table)

# Demographic Analysis
elif section == "Demographic Analysis":
    st.markdown('<div class="section-header">👥 Demographic Analysis</div>', unsafe_allow_html=True)
    
    demographic_var = st.selectbox(
        "Select Demographic Variable:",
        ['AGE_GROUP', 'OCCUPATION_GROUP', 'EDUCATION_GROUP', 'GENDER_GROUP', 'SIZE_GROUP']
    )
    
    counts = df_clean[demographic_var].value_counts(normalize=True) * 100
    
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        title=f'Distribution of {demographic_var.replace("_", " ").title()}',
        labels={'x': 'Percentage (%)', 'y': demographic_var.replace('_', ' ').title()}
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Inequality Measures
elif section == "Inequality Measures":
    st.markdown('<div class="section-header">⚖️ Inequality Measures</div>', unsafe_allow_html=True)
    
    # Gini coefficient function
    def gini_coefficient(df, value_col, weight_col):
        df2 = df[[value_col, weight_col]].dropna()
        if df2.shape[0] == 0:
            return np.nan
        df_sorted = df2.sort_values(value_col)
        values = df_sorted[value_col].astype(float).values
        weights = df_sorted[weight_col].astype(float).values
        total_weight = weights.sum()
        if total_weight == 0 or values.sum() == 0:
            return np.nan
        w = weights / total_weight
        cumw = np.concatenate(([0.0], np.cumsum(w)))
        cumv = np.concatenate(([0.0], np.cumsum(values * w) / np.sum(values * w)))
        auc = np.trapz(cumv, cumw)
        gini = 1 - 2 * auc
        return gini
    
    # Calculate Gini coefficients
    gini_income_rural = gini_coefficient(df_clean[df_clean['REGION_TYPE'] == 'RURAL'], 'TOTAL_INCOME', 'HH_WEIGHT_MS')
    gini_income_urban = gini_coefficient(df_clean[df_clean['REGION_TYPE'] == 'URBAN'], 'TOTAL_INCOME', 'HH_WEIGHT_MS')
    gini_exp_rural = gini_coefficient(df_clean[df_clean['REGION_TYPE'] == 'RURAL'], 'TOTAL_EXPENDITURE', 'HH_WEIGHT_MS')
    gini_exp_urban = gini_coefficient(df_clean[df_clean['REGION_TYPE'] == 'URBAN'], 'TOTAL_EXPENDITURE', 'HH_WEIGHT_MS')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income Inequality (Gini Coefficient)")
        st.metric("Rural", f"{gini_income_rural:.3f}")
        st.metric("Urban", f"{gini_income_urban:.3f}")
    
    with col2:
        st.subheader("Expenditure Inequality (Gini Coefficient)")
        st.metric("Rural", f"{gini_exp_rural:.3f}")
        st.metric("Urban", f"{gini_exp_urban:.3f}")
    
    # Interpretation
    st.info("""
    **Gini Coefficient Interpretation:**
    - 0 = Perfect equality
    - 0.2-0.3 = Relatively equal
    - 0.3-0.4 = Moderate inequality
    - 0.4+ = High inequality
    """)

# Statistical Analysis
elif section == "Statistical Analysis":
    st.markdown('<div class="section-header">📊 Statistical Analysis</div>', unsafe_allow_html=True)
    
    st.subheader("Correlation Analysis")
    
    # Select variables for correlation
    numeric_cols = ['TOTAL_INCOME', 'TOTAL_EXPENDITURE'] + [
        col for col in df_clean.columns if 'MONTHLY_EXPENSE_ON_' in col or 'INCOME_OF_' in col
    ]
    numeric_cols = [col for col in numeric_cols if col in df_clean.columns][:10]  # Limit for performance
    
    if st.checkbox("Show Correlation Heatmap"):
        corr_matrix = df_clean[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "**Project:** Exploring Household Financial Landscapes | "
    "**Data Source:** Income & Expenditure Survey Data"
)