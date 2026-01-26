import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(df):
    """
    Perform basic data analysis on the given DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame to analyze.
        
    Returns:
        None
    """
    print("Data Summary:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nUnique Values per Column:")
    print(df.nunique())
    
    print("\nUnique Violations:")
    unique_violations = df['Violation_Type'].value_counts()
    print(unique_violations)
    
    # Plotting the distribution of Violation Types
    violation_fines = df.groupby("Violation_Type")["Fine_Amount"].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=violation_fines.index, y=violation_fines.values)
    plt.xticks(rotation=90)
    plt.title("Total Fine Amount by Violation Type")
    plt.xlabel("Violation Type")
    plt.ylabel("Total Fine Amount")
    plt.tight_layout()
    plt.show()
        
if __name__ == "__main__":
    data_path = "datasets/dataset.csv"
    df = pd.read_csv(data_path)
    analyze_data(df)