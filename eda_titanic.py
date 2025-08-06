import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the visual style for the plots
sns.set_style("whitegrid")

# --- Step 2: Load the Dataset ---
try:
    df = pd.read_csv('titanic.csv')
    print("Titanic dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'titanic.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# --- Step 3: Initial Data Inspection ---
print("\n--- First 5 Rows of the Dataset ---")
print(df.head())

print("\n--- Dataset Info ---")
df.info()

# --- Step 4: Generate Summary Statistics ---
# [cite_start]This provides statistics for numerical columns like Age, Fare, etc. [cite: 6]
print("\n--- Summary Statistics for Numerical Features ---")
print(df.describe())

# --- Step 5: Handle Missing Data (Simple Imputation) ---
# For EDA purposes, we'll fill missing 'Age' values with the median.
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)
print(f"\n Missing 'Age' values filled with median age: {median_age}")

# --- Step 6: Univariate Analysis (Visualizing Single Features) ---

# [cite_start]Histogram for Age distribution [cite: 6]
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers', fontsize=16)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('1_age_distribution.png')
plt.show()

# Count plot for survival
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count (0 = Died, 1 = Survived)', fontsize=16)
plt.xlabel('Survived')
plt.ylabel('Count')
plt.savefig('2_survival_count.png')
plt.show()

# Count plot for Passenger Class
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', data=df)
plt.title('Passenger Count by Class', fontsize=16)
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.savefig('3_pclass_count.png')
plt.show()

# --- Step 7: Bivariate Analysis (Visualizing Relationships) ---

# [cite_start]Boxplot of Age by Passenger Class [cite: 6]
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age Distribution by Passenger Class', fontsize=16)
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.savefig('4_age_by_pclass.png')
plt.show()

# Bar chart of survival rate by Sex
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=df, ci=None)
plt.title('Survival Rate by Gender', fontsize=16)
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.savefig('5_survival_by_gender.png')
plt.show()

# Bar chart of survival rate by Passenger Class
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=df, ci=None)
plt.title('Survival Rate by Passenger Class', fontsize=16)
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.savefig('6_survival_by_pclass.png')
plt.show()


# --- Step 8: Multivariate Analysis ---

# [cite_start]Correlation Matrix for numerical features [cite: 7]
plt.figure(figsize=(12, 8))
# Select only numeric columns for correlation calculation
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.savefig('7_correlation_matrix.png')
plt.show()

# [cite_start]Pairplot to see relationships between key numerical variables [cite: 7]
# We'll use a subset of columns for readability
pairplot_cols = ['Survived', 'Pclass', 'Age', 'Fare']
sns.pairplot(df[pairplot_cols], hue='Survived', diag_kind='kde')
plt.savefig('8_pairplot.png')
plt.show()

print("\n Exploratory Data Analysis complete. All charts have been displayed and saved as PNG files.")
