import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
import os

# Set up paths to load data effectively
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "world_population_data_clean.csv"

# --- Dashboard Setup ---
st.set_page_config(page_title="World Population Analysis", layout="wide")
sns.set_theme(style="whitegrid", palette="muted")


# --- Helper Load Data Function ---
@st.cache_data
def load_data():
    if not PROCESSED_DATA_PATH.exists():
        st.error(f"Cannot find processed data at {PROCESSED_DATA_PATH}. Please run the pipeline first.")
        return pd.DataFrame()
    return pd.read_csv(PROCESSED_DATA_PATH)


df = load_data()


# --- Sidebar / Navigation ---
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Overview (Problem, Hypothesis, Conclusion)", "Exploratory Insights", "Visual Analysis"]
)


# --- 1. Overview Section ---
if menu == "Overview (Problem, Hypothesis, Conclusion)":
    st.title("🌍 World Population Data Analysis Dashboard")
    st.image("https://plus.unsplash.com/premium_photo-1661963057116-2584fb41d1db?auto=format&fit=crop&w=1200&q=80", use_container_width=True)

    st.header("Statement of the Problem")
    st.write("""
    Understanding global demographic shifts and population distribution is critical for policymakers, 
    economists, and researchers deciding on resource allocation, urban planning, and environmental impact strategies. 
    This project aims to process, clean, and analyze historical and modern world population data to identify trends, 
    pinpoint rapid-growth areas, and understand how population density is distributed across different continents over half a century.
    """)

    st.header("Hypothesis")
    st.write("""
    We hypothesize that:
    1. **Regional Dominance**: Asian and African continents will demonstrate the highest population totals and steepest growth patterns over the timeline from 1970 to 2023.
    2. **Growth Stagnation**: European countries will reflect flattened or declining population trajectories in recent years.
    3. **Density vs. Area**: Total population sizes will show a low direct correlation with population density, as massive nations might have vast uninhabited terrains compared to highly dense smaller regions.
    """)

    st.header("Conclusion Based on Observation")
    st.write("""
    Based on our exploratory data analysis and visual observations:
    - **Demographic Boom**: Asia is solidly leading in global population, with India and China accounting for massive portions of total world inhabitants. Africa is displaying the steepest growth gradient entering the 2020s.
    - **Population Density Extremes**: Our analysis confirmed that high population explicitly does not guarantee high density. Micro-states and specific geographic constraints dictate density extremes, while massive nations like Russia display strikingly low population density.
    - **Timeline Stagnation**: European nations show flattened timelines across the plotted decades, indicating stabilizing or shrinking demographics compared to the sharply rising lines of the developing continents. 
    - **Correlations**: The Pearson correlation heatmap reinforces that land area has almost no correlation with population growth rate, isolating demographic shifts to specific socio-economic and regional factors rather than raw geographic size alone.
    """)


# --- 2. Exploratory Insights ---
elif menu == "Exploratory Insights":
    st.title("📊 Exploratory Insights")
    
    if df.empty:
        st.stop()

    st.subheader("Data Shape")
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    st.subheader("Data Overview (First 5 Rows)")
    st.dataframe(df.head())

    st.subheader("Data Summary (Descriptive Statistics)")
    st.dataframe(df.describe())

    st.subheader("Null Values in Dataset")
    st.write(df.isnull().sum())
    
    st.subheader("Density Extremes")
    valid_density = df.dropna(subset=['density_(km²)'])
    if not valid_density.empty:
        most_dense = valid_density.loc[valid_density['density_(km²)'].idxmax()]
        least_dense = valid_density.loc[valid_density['density_(km²)'].idxmin()]
        st.success(f"**Most Densely Populated:** {most_dense['country']} ({most_dense['density_(km²)']} ppl/km²)")
        st.warning(f"**Least Densely Populated:** {least_dense['country']} ({least_dense['density_(km²)']} ppl/km²)")


# --- 3. Visual Analysis Section ---
elif menu == "Visual Analysis":
    st.title("📉 Visual Analysis")
    
    if df.empty:
        st.stop()

    # 1. Top Populated Countries
    st.subheader("1. Top 10 Most Populated Countries in 2023")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    top_countries = df.sort_values(by='2023_population', ascending=False).head(10)
    sns.barplot(data=top_countries, x='2023_population', y='country', palette='viridis', ax=ax1)
    ax1.set_title('Top 10 Most Populated Countries in 2023')
    ax1.set_xlabel('Population in 2023')
    ax1.set_ylabel('Country')
    st.pyplot(fig1)

    # 2. Population by Continent
    st.subheader("2. Total Population by Continent in 2023")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    continent_pop = df.groupby('continent')['2023_population'].sum().sort_values(ascending=False).reset_index()
    sns.barplot(data=continent_pop, x='continent', y='2023_population', palette='magma', ax=ax2)
    ax2.set_title('Population by Continent in 2023')
    ax2.set_xlabel('Continent')
    ax2.set_ylabel('Total Population')
    st.pyplot(fig2)

    # 3. Population Timeline
    st.subheader("3. Continent-wise Population Timeline (1970 - 2023)")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    pop_columns = [
        '1970_population', '1980_population', '1990_population', 
        '2000_population', '2010_population', '2015_population', 
        '2020_population', '2022_population', '2023_population'
    ]
    continent_timeline = df.groupby('continent')[pop_columns].sum()
    continent_timeline.columns = [col.split('_')[0] for col in continent_timeline.columns]
    continent_timeline = continent_timeline.T
    sns.lineplot(data=continent_timeline, markers=True, dashes=False, linewidth=2.5, ax=ax3)
    ax3.set_title('Continent-wise Population Timeline (1970 - 2023)')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Total Population')
    ax3.grid(True)
    st.pyplot(fig3)

    # 4. Density boxplot
    st.subheader("4. Continent-wise Population Density")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='continent', y='density_(km²)', palette='Set2', ax=ax4)
    ax4.set_yscale('log')
    ax4.set_title('Continent-wise Population Density (Log Scale)')
    ax4.set_xlabel('Continent')
    ax4.set_ylabel('Density (km²) - Log Scale')
    st.pyplot(fig4)

    # 5. Correlation Heatmap
    st.subheader("5. Pearson Correlation Heatmap")
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr(method='pearson')
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5, ax=ax5)
    ax5.set_title('Pearson Correlation Heatmap of Dataset Features')
    st.pyplot(fig5)