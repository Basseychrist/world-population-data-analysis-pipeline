import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set visualization style
sns.set_theme(style="whitegrid", palette="muted")

def exploratory_insights(df: pd.DataFrame):
    """
    Prints exploratory insights including data shape, overview, info, summary, and null values.
    """
    print("="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    print("\n1. Data Shape:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n2. Data Overview (First 5 Rows):")
    print(df.head())
    
    print("\n3. Data Information:")
    df.info()
    
    print("\n4. Data Summary (Descriptive Statistics):")
    print(df.describe())
    
    print("\n5. Null Values in Dataset:")
    print(df.isnull().sum())
    print("="*50)


def plot_top_populated_countries_2023(df: pd.DataFrame, top_n: int = 10, save_dir: Path = None):
    """
    Plots the top N populated countries in 2023.
    """
    plt.figure(figsize=(12, 6))
    top_countries = df.sort_values(by='2023_population', ascending=False).head(top_n)
    
    sns.barplot(data=top_countries, x='2023_population', y='country', palette='viridis')
    plt.title(f'Top {top_n} Most Populated Countries in 2023')
    plt.xlabel('Population in 2023')
    plt.ylabel('Country')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'top_populated_countries_2023.png')
    plt.show()

def plot_population_in_continents_2023(df: pd.DataFrame, save_dir: Path = None):
    """
    Plots the total population in 2023 for each continent.
    """
    plt.figure(figsize=(10, 6))
    continent_pop = df.groupby('continent')['2023_population'].sum().sort_values(ascending=False).reset_index()
    
    sns.barplot(data=continent_pop, x='continent', y='2023_population', palette='magma')
    plt.title('Population by Continent in 2023')
    plt.xlabel('Continent')
    plt.ylabel('Total Population')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'population_by_continent_2023.png')
    plt.show()

def plot_continent_population_timeline(df: pd.DataFrame, save_dir: Path = None):
    """
    Plots the continent-wise population timeline from 1970 to 2023.
    """
    plt.figure(figsize=(14, 7))
    
    pop_columns = [
        '1970_population', '1980_population', '1990_population', 
        '2000_population', '2010_population', '2015_population', 
        '2020_population', '2022_population', '2023_population'
    ]
    
    # Group by continent and sum the populations for each year
    continent_timeline = df.groupby('continent')[pop_columns].sum()
    
    # Rename columns to just years for plotting
    continent_timeline.columns = [col.split('_')[0] for col in continent_timeline.columns]
    
    # Transpose to have years as index, continents as columns
    continent_timeline = continent_timeline.T
    
    sns.lineplot(data=continent_timeline, markers=True, dashes=False, linewidth=2.5)
    plt.title('Continent-wise Population Timeline (1970 - 2023)')
    plt.xlabel('Year')
    plt.ylabel('Total Population')
    plt.legend(title='Continent')
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'continent_population_timeline.png')
    plt.show()

def plot_continent_population_density(df: pd.DataFrame, save_dir: Path = None):
    """
    Plots the density for each continent.
    """
    plt.figure(figsize=(10, 6))
    
    # Using boxplot to show distribution of density inside continents
    sns.boxplot(data=df, x='continent', y='density_(km²)', palette='Set2')
    plt.yscale('log') # Log scale because density varies wildly
    plt.title('Continent-wise Population Density (Log Scale)')
    plt.xlabel('Continent')
    plt.ylabel('Density (km²) - Log Scale')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'continent_population_density.png')
    plt.show()

def display_density_extremes(df: pd.DataFrame):
    """
    Displays the most densely populated and least densely populated countries.
    """
    print("\n" + "="*50)
    print("DENSITY EXTREMES")
    print("="*50)
    
    # Dropna to avoid issues if any density is missing
    valid_density = df.dropna(subset=['density_(km²)'])
    
    most_dense = valid_density.loc[valid_density['density_(km²)'].idxmax()]
    least_dense = valid_density.loc[valid_density['density_(km²)'].idxmin()]
    
    print(f"Most Densely Populated: {most_dense['country']} ({most_dense['density_(km²)']} ppl/km²)")
    print(f"Least Densely Populated: {least_dense['country']} ({least_dense['density_(km²)']} ppl/km²)")
    print("="*50 + "\n")

def plot_pearson_correlation(df: pd.DataFrame, save_dir: Path = None):
    """
    Creates a Pearson correlation heatmap of features.
    """
    plt.figure(figsize=(14, 10))
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])
    
    corr_matrix = numeric_df.corr(method='pearson')
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5)
    plt.title('Pearson Correlation Heatmap of Dataset Features')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'pearson_correlation_heatmap.png')
    plt.show()

def run_all_analysis():
    """
    Helper function to run all analysis and save plots.
    """
    # Define paths
    BASE_DIR = Path(__file__).parent.parent
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "world_population_data_clean.csv"
    PLOTS_DIR = BASE_DIR / "images"
    
    # Create directory for plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not PROCESSED_DATA_PATH.exists():
        print(f"Processed data not found at {PROCESSED_DATA_PATH}. Please run pipeline.py first.")
        return
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # 1. Exploratory Insights
    exploratory_insights(df)
    
    # 2. Most Densely / Least Densely Populated
    display_density_extremes(df)
    
    # 3. Visualizations (Saved to images/)
    plot_top_populated_countries_2023(df, save_dir=PLOTS_DIR)
    plot_population_in_continents_2023(df, save_dir=PLOTS_DIR)
    plot_continent_population_timeline(df, save_dir=PLOTS_DIR)
    plot_continent_population_density(df, save_dir=PLOTS_DIR)
    plot_pearson_correlation(df, save_dir=PLOTS_DIR)
    
    print(f"All visualizations saved successfully to {PLOTS_DIR}.")

if __name__ == "__main__":
    run_all_analysis()
