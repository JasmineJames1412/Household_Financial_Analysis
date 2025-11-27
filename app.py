import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wquantiles import median
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Household Financial Analysis - Research Findings",
    page_icon="🏠",
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
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title - Consistent with your report
st.markdown('<div class="main-header">🏠 Household Financial Analysis: Income & Expenditure Trends in India</div>', unsafe_allow_html=True)
st.markdown("**Research Findings Dashboard - Consistent with Chapter 4 Results**")

# Sidebar for navigation
st.sidebar.title("📊 Research Sections")
section = st.sidebar.radio(
    "Navigate to:",
    ["Executive Summary", "Descriptive Statistics", "Income Analysis", "Expenditure Analysis", 
     "Regional Comparison", "Inequality Measures", "Demographic Analysis", "Econometric Findings"]
)

# Load data function
@st.cache_data
def load_data():
    # Your existing data loading code
    common_cols = ['HH_ID', 'STATE', 'HR', 'DISTRICT', 'REGION_TYPE', 'STRATUM', 'PSU_ID', 'MONTH_SLOT', 'MONTH', 'RESPONSE_STATUS', 
                   'REASON_FOR_NON_RESPONSE', 'FAMILY_SHIFTED', 'HH_WEIGHT_MS', 'HH_WEIGHT_FOR_COUNTRY_MS', 'HH_WEIGHT_FOR_STATE_MS', 
                   'HH_NON_RESPONSE_MS', 'HH_NON_RESPONSE_FOR_COUNTRY_MS', 'HH_NON_RESPONSE_FOR_STATE_MS', 'AGE_GROUP', 'OCCUPATION_GROUP', 
                   'EDUCATION_GROUP', 'GENDER_GROUP', 'SIZE_GROUP']

    df_income = pd.read_csv('Income.csv')
    df_expenditure = pd.read_csv('Expenditure.csv')
    df = pd.merge(df_income, df_expenditure, on=common_cols, how='inner')
    
    # Clean data
    df_clean = df[df['RESPONSE_STATUS'] == 'Accepted'].copy()
    df_clean = df_clean.replace(-99, np.nan)
    df_clean = df_clean.dropna(subset=['TOTAL_INCOME', 'TOTAL_EXPENDITURE'])
    
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
with st.spinner('Loading research data...'):
    df_clean = load_data()

# EXECUTIVE SUMMARY SECTION
if section == "Executive Summary":
    st.markdown('<div class="section-header">📊 Executive Summary</div>', unsafe_allow_html=True)
    
    # Key Findings from your report
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Urban Income", "₹26,290", "+58% vs Rural")
    with col2:
        st.metric("Rural Income", "₹17,158", "Baseline")
    with col3:
        st.metric("Income Gap", "₹9,132", "Monthly difference")
    with col4:
        st.metric("Urban Premium", "₹3,314", "PSM Analysis")
    
    st.markdown("---")
    
    # Key Insights from your report
    st.subheader("🎯 Key Research Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **💰 Income Patterns:**
        - Wages contribute **71%** to total household income
        - Business profits are second largest source (**17.6%**)
        - Urban households earn **58% more** than rural
        - City bonus exists at **all income levels**
        """)
        
        st.markdown("""
        **🏙️ Regional Differences:**
        - Urban spending varies more (SD: ₹15,088 vs ₹5,620)
        - Rural households spend **47.1%** on food vs urban **42.4%**
        - Urban families save more (-0.409 vs -0.280 log ratio)
        """)
    
    with col2:
        st.markdown("""
        **📈 Savings Drivers:**
        - **Education**: Graduates save dramatically more (-0.847 vs -0.270)
        - **Occupation**: White-collar professionals are biggest savers
        - **Family Structure**: Large joint families save most
        - **Location**: State matters significantly for savings capacity
        """)
        
        st.markdown("""
        **⚖️ Inequality Insights:**
        - Rural income inequality higher (Gini: 0.416 vs 0.345)
        - Urban spending inequality higher (Gini: 0.266 vs 0.226)
        - Income varies more than spending across all households
        """)

# DESCRIPTIVE STATISTICS SECTION
elif section == "Descriptive Statistics":
    st.markdown('<div class="section-header">📈 Descriptive Statistics</div>', unsafe_allow_html=True)
    
    # Overall Statistics Table 4.1
    st.subheader("Table 4.1: Overall Income and Expenditure Statistics")
    
    overall_stats = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Standard Deviation'],
        'Income (₹)': ['20,678', '16,000', '19,953'],
        'Expenditure (₹)': ['13,110', '11,536', '11,030']
    })
    st.dataframe(overall_stats, use_container_width=True)
    
    st.info("""
    **Interpretation:** Average monthly household income (₹20,678) is higher than expenditure (₹13,110), 
    indicating most families spend less than they earn. Medians are lower than means, showing right-skewed 
    distributions with some high-earning households pulling averages up.
    """)
    
    # Regional Statistics Table 4.2
    st.subheader("Table 4.2: Regional Income and Expenditure Patterns")
    
    regional_stats = pd.DataFrame({
        'Metric': ['Mean Income', 'Median Income', 'Std Dev Income', 'Mean Expenditure', 'Median Expenditure', 'Std Dev Expenditure'],
        'Rural': ['17,158', '13,348', '18,448', '11,578', '10,455', '5,620'],
        'Urban': ['26,290', '20,900', '20,952', '15,552', '13,810', '15,088']
    })
    st.dataframe(regional_stats, use_container_width=True)
    
    st.info("""
    **Interpretation:** Urban families earn more, spend more, and show greater variation in both income and expenditure 
    compared to rural families who are more clustered around middle values.
    """)

# INCOME ANALYSIS SECTION
elif section == "Income Analysis":
    st.markdown('<div class="section-header">💰 Income Analysis</div>', unsafe_allow_html=True)
    
    # Income Sources Breakdown
    st.subheader("Income Sources Distribution")
    
    income_sources = pd.DataFrame({
        'Source': ['Wages', 'Business Profit', 'Pension', 'Self-Production', 'Rent', 'Other'],
        'Percentage': [71.0, 17.6, 5.4, 2.4, 1.2, 2.4]
    })
    
    fig_income = px.pie(income_sources, values='Percentage', names='Source', 
                       title="Figure 4.4: Overall Distribution of Income Sources")
    st.plotly_chart(fig_income, use_container_width=True)
    
    st.info("""
    **Key Finding:** Wages dominate household income (71%), showing most families rely on salaries or daily wages 
    as their primary income source. Business profits are the second largest contributor.
    """)
    
    # Rural vs Urban Income Sources
    st.subheader("Rural vs Urban Income Sources")
    
    rural_urban_income = pd.DataFrame({
        'Source': ['Wages', 'Business Profit', 'Pension', 'Self-Production'],
        'Rural (%)': [68.9, 20.5, 1.9, 3.1],
        'Urban (%)': [71.6, 16.8, 6.3, 0.7]
    })
    
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(name='Rural', x=rural_urban_income['Source'], y=rural_urban_income['Rural (%)']))
    fig_comparison.add_trace(go.Bar(name='Urban', x=rural_urban_income['Source'], y=rural_urban_income['Urban (%)']))
    fig_comparison.update_layout(title="Figure 4.5: Breakdown of Income Sources (Rural vs Urban)", barmode='group')
    st.plotly_chart(fig_comparison, use_container_width=True)

# EXPENDITURE ANALYSIS SECTION
elif section == "Expenditure Analysis":
    st.markdown('<div class="section-header">🛒 Expenditure Analysis</div>', unsafe_allow_html=True)
    
    # Overall Expenditure
    st.subheader("Overall Expenditure Distribution")
    
    expenditure_categories = pd.DataFrame({
        'Category': ['Food', 'Power/Fuel', 'Miscellaneous', 'Cosmetics/Toiletries', 'Communication', 'Others'],
        'Percentage': [43.6, 18.9, 6.2, 5.8, 4.5, 21.0]
    })
    
    fig_exp = px.bar(expenditure_categories, x='Category', y='Percentage',
                    title="Figure 4.6: Overall Distribution of Expenditure Categories")
    st.plotly_chart(fig_exp, use_container_width=True)
    
    st.info("""
    **Key Finding:** Food consumes 43.6% of total budget, showing it's the primary financial priority. 
    Combined with power/fuel (18.9%), these essentials account for over 60% of household spending.
    """)
    
    # Rural vs Urban Expenditure
    st.subheader("Rural vs Urban Expenditure Patterns")
    
    rural_urban_exp = pd.DataFrame({
        'Category': ['Food', 'Power/Fuel', 'Communication', 'EMIs'],
        'Rural (%)': [47.1, 18.5, 4.3, 2.4],
        'Urban (%)': [42.4, 19.2, 4.7, 4.1]
    })
    
    fig_exp_compare = go.Figure()
    fig_exp_compare.add_trace(go.Bar(name='Rural', x=rural_urban_exp['Category'], y=rural_urban_exp['Rural (%)']))
    fig_exp_compare.add_trace(go.Bar(name='Urban', x=rural_urban_exp['Category'], y=rural_urban_exp['Urban (%)']))
    fig_exp_compare.update_layout(title="Figure 4.7: Rural vs Urban Expenditure Comparison", barmode='group')
    st.plotly_chart(fig_exp_compare, use_container_width=True)

# REGIONAL COMPARISON SECTION
elif section == "Regional Comparison":
    st.markdown('<div class="section-header">🏙️ Regional Comparison</div>', unsafe_allow_html=True)
    
    # Income and Expenditure Bar Chart
    st.subheader("Urban vs Rural Income and Expenditure")
    
    comparison_data = pd.DataFrame({
        'Region': ['Rural', 'Urban'],
        'Income': [17158, 26290],
        'Expenditure': [11578, 15552]
    })
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name='Income', x=comparison_data['Region'], y=comparison_data['Income'], marker_color='blue'))
    fig_bar.add_trace(go.Bar(name='Expenditure', x=comparison_data['Region'], y=comparison_data['Expenditure'], marker_color='orange'))
    fig_bar.update_layout(title="Figure 4.1: Rural vs Urban Income and Expenditure")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.info("""
    **Key Insight:** Urban households earn ₹9,132 more and spend ₹3,974 more monthly than rural households. 
    The smaller spending gap compared to income gap suggests urban families save more or spend on non-essentials.
    """)
    
    # Distribution Plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Simulated income distribution
        st.subheader("Income Distribution")
        st.image("https://via.placeholder.com/600x400/4CAF50/FFFFFF?text=Income+Distribution+Plot", 
                caption="Figure 4.2: Income and Expenditure Distribution")
    
    with col2:
        # Simulated regional distribution
        st.subheader("Regional Distribution")
        st.image("https://via.placeholder.com/600x400/2196F3/FFFFFF?text=Regional+Distribution+Plot", 
                caption="Figure 4.3: Income and Expenditure Distribution by Region")

# INEQUALITY MEASURES SECTION
elif section == "Inequality Measures":
    st.markdown('<div class="section-header">⚖️ Inequality Measures</div>', unsafe_allow_html=True)
    
    # Gini Coefficients
    st.subheader("Income and Expenditure Inequality (Gini Coefficients)")
    
    gini_data = pd.DataFrame({
        'Region': ['Rural', 'Urban'],
        'Income Gini': [0.416, 0.345],
        'Expenditure Gini': [0.226, 0.266]
    })
    
    st.dataframe(gini_data, use_container_width=True)
    
    st.info("""
    **Critical Finding:** 
    - Rural income inequality is **higher** (0.416 vs 0.345)
    - Urban expenditure inequality is **higher** (0.266 vs 0.226)
    - This reveals different types of inequality: rural has earning disparity, urban has lifestyle disparity
    """)
    
    # Decile Analysis
    st.subheader("Decile Analysis - Income Distribution")
    
    decile_data = pd.DataFrame({
        'Decile': ['D1 (Poorest)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Richest)'],
        'Rural Income': [2800, 4800, 6800, 8800, 11000, 13500, 16500, 20500, 26000, 32000],
        'Urban Income': [10120, 13500, 16000, 18500, 21000, 24000, 27500, 32000, 38000, 48000]
    })
    
    st.dataframe(decile_data, use_container_width=True)
    
    st.info("""
    **Key Insight:** The income gap grows dramatically across deciles. Urban poorest (D1) earn more than 
    rural middle-class (D5), showing massive location-based advantage.
    """)

# DEMOGRAPHIC ANALYSIS SECTION
elif section == "Demographic Analysis":
    st.markdown('<div class="section-header">👥 Demographic Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender Distribution
        gender_data = pd.DataFrame({
            'Category': ['Balanced', 'Male Majority', 'Female Majority', 'Only Males', 'Only Females'],
            'Percentage': [34.8, 27.4, 14.0, 12.3, 11.5]
        })
        fig_gender = px.bar(gender_data, x='Category', y='Percentage', 
                           title="Figure 4.8: Household Distribution by Gender")
        st.plotly_chart(fig_gender, use_container_width=True)
        
        # Education Distribution
        education_data = pd.DataFrame({
            'Level': ['All Literate', 'Matriculation', 'Graduates Dominated', 'All Illiterates', 'Some Literates'],
            'Percentage': [33.1, 25.4, 15.2, 13.8, 12.5]
        })
        fig_edu = px.bar(education_data, x='Level', y='Percentage', 
                        title="Figure 4.12: Household Distribution by Education")
        st.plotly_chart(fig_edu, use_container_width=True)
    
    with col2:
        # Household Size
        size_data = pd.DataFrame({
            'Size': ['4 Members', '3 Members', '2 Members', '5 Members', '6+ Members'],
            'Percentage': [30.0, 26.3, 18.5, 12.8, 12.4]
        })
        fig_size = px.bar(size_data, x='Size', y='Percentage', 
                         title="Figure 4.9: Household Distribution by Size")
        st.plotly_chart(fig_size, use_container_width=True)
        
        # Occupation Distribution
        occupation_data = pd.DataFrame({
            'Occupation': ['Self-Employed', 'Wage Labor', 'Professional', 'Farmer', 'Others'],
            'Percentage': [20.4, 15.3, 12.8, 11.5, 40.0]
        })
        fig_occ = px.bar(occupation_data, x='Occupation', y='Percentage', 
                        title="Figure 4.10: Household Distribution by Occupation")
        st.plotly_chart(fig_occ, use_container_width=True)

# ECONOMETRIC FINDINGS SECTION
elif section == "Econometric Findings":
    st.markdown('<div class="section-header">📊 Econometric Findings</div>', unsafe_allow_html=True)
    
    # OLS Regression Results
    st.subheader("OLS Regression: Key Drivers of Savings Behavior")
    
    ols_coefficients = pd.DataFrame({
        'Variable': ['Organised Farmer', 'Small/Marginal Farmer', 'Over 15 Members', 
                    'Business & Salaried Employees', 'Meghalaya State', 'All Illiterates'],
        'Coefficient': [0.95, 0.66, -0.72, -0.27, -0.54, 0.27],
        'Impact': ['Reduces Savings', 'Reduces Savings', 'Increases Savings', 
                  'Increases Savings', 'Increases Savings', 'Reduces Savings']
    })
    
    st.dataframe(ols_coefficients, use_container_width=True)
    
    st.info("""
    **Key Findings from OLS:**
    - **Farmers struggle most** with savings (positive coefficients)
    - **Large joint families** save most (negative coefficient of -0.72)
    - **Formal employment** boosts savings capacity
    - **Education** is crucial for financial security
    """)
    
    # Quantile Regression
    st.subheader("Quantile Regression: Urban Income Premium")
    
    quantile_data = pd.DataFrame({
        'Quantile': ['10th (Poorest)', '25th', '50th (Median)', '75th', '90th (Richest)'],
        'Urban Premium (₹)': [816, 975, 1090, 1112, 941]
    })
    
    fig_quantile = px.line(quantile_data, x='Quantile', y='Urban Premium (₹)', 
                          markers=True, title="Figure 4.25: Urban Income Premium Across Quantiles")
    st.plotly_chart(fig_quantile, use_container_width=True)
    
    st.info("""
    **Critical Insight:** Urban advantage exists at ALL income levels, peaking at 75th percentile (₹1,112). 
    This means cities benefit upper-middle class the most, but provide advantages to poor households too.
    """)
    
    # PSM Results
    st.subheader("Propensity Score Matching: True Urban Bonus")
    
    psm_results = pd.DataFrame({
        'Metric': ['ATT (Urban Premium)', '95% CI Lower', '95% CI Upper', 'Standard Error', 'Matched Samples'],
        'Value': ['₹3,314', '₹3,159', '₹3,506', '₹89', '90,164 households']
    })
    
    st.dataframe(psm_results, use_container_width=True)
    
    st.success("""
    **Final Conclusion:** After controlling for all observable characteristics, the pure "city effect" 
    adds **₹3,314** to monthly household income. This is the true economic advantage of urban locations.
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Research Dashboard** | Consistent with Chapter 4 Findings | "
    "**Data Source:** National Income & Expenditure Survey"
)
