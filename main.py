import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# STEP 1: IMPORT AND EXPLORE THE DATASET
print("=" * 80)
print("STEP 1: IMPORTING AND EXPLORING THE DATASET")
print("=" * 80)

# Load the dataset
df = pd.read_csv('Titanic.csv')

# Display basic information
print("\n First 5 rows of the dataset:")
print(df.head())

print("\n Dataset Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n Dataset Info:")
print(df.info())

print("\n Statistical Summary:")
print(df.describe())

print("\n Missing Values Count:")
print(df.isnull().sum())

print("\n Missing Values Percentage:")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0])

# STEP 2: HANDLE MISSING VALUES
print("\n" + "=" * 80)
print("STEP 2: HANDLING MISSING VALUES")
print("=" * 80)

# making a copy to preserve original data
df_cleaned = df.copy()

# Handle 'Age' - Fill with median (robust to outliers)
median_age = df_cleaned['Age'].median()
df_cleaned['Age'].fillna(median_age, inplace=True)
print(f"\n Filled 'Age' missing values with median: {median_age:.2f}")

# Handle 'Embarked' - Fill with mode (most frequent value)
if 'Embarked' in df_cleaned.columns and df_cleaned['Embarked'].isnull().sum() > 0:
    mode_embarked = df_cleaned['Embarked'].mode()[0]
    df_cleaned['Embarked'].fillna(mode_embarked, inplace=True)
    print(f" Filled 'Embarked' missing values with mode: {mode_embarked}")

# Handle 'Cabin' - High percentage missing, so we'll create a binary feature
if 'Cabin' in df_cleaned.columns:
    df_cleaned['Has_Cabin'] = df_cleaned['Cabin'].notna().astype(int)
    df_cleaned.drop('Cabin', axis=1, inplace=True)
    print("Created 'Has_Cabin' binary feature and dropped 'Cabin' column")

# Drop 'Fare' missing values if any (usually very few)
if 'Fare' in df_cleaned.columns and df_cleaned['Fare'].isnull().sum() > 0:
    df_cleaned['Fare'].fillna(df_cleaned['Fare'].median(), inplace=True)
    print("Filled 'Fare' missing values with median")

print("\nMissing values after cleaning:")
print(df_cleaned.isnull().sum())

# STEP 3: ENCODE CATEGORICAL FEATURES
print("\n" + "=" * 80)
print("STEP 3: ENCODING CATEGORICAL FEATURES")
print("=" * 80)

# Identify categorical columns
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns found: {categorical_cols}")

# Drop non-useful columns
columns_to_drop = ['PassengerId', 'Name', 'Ticket']
existing_cols_to_drop = [col for col in columns_to_drop if col in df_cleaned.columns]
df_cleaned.drop(existing_cols_to_drop, axis=1, inplace=True)
print(f"Dropped columns: {existing_cols_to_drop}")

# Encode 'Sex' using Label Encoding (binary: male/female)
if 'Sex' in df_cleaned.columns:
    le_sex = LabelEncoder()
    df_cleaned['Sex'] = le_sex.fit_transform(df_cleaned['Sex'])
    print(f"\nEncoded 'Sex': {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

# Encode 'Embarked' using One-Hot Encoding
if 'Embarked' in df_cleaned.columns:
    embarked_dummies = pd.get_dummies(df_cleaned['Embarked'], prefix='Embarked', drop_first=True)
    df_cleaned = pd.concat([df_cleaned, embarked_dummies], axis=1)
    df_cleaned.drop('Embarked', axis=1, inplace=True)
    print("One-Hot Encoded 'Embarked' (dropped first category to avoid multicollinearity)")

print("\nData types after encoding:")
print(df_cleaned.dtypes)

print("\nDataset shape after encoding:")
print(f"Rows: {df_cleaned.shape[0]}, Columns: {df_cleaned.shape[1]}")

# STEP 4: NORMALIZE/STANDARDIZE NUMERICAL FEATURES
print("\n" + "=" * 80)
print("STEP 4: FEATURE SCALING")
print("=" * 80)

# Identify numerical columns (excluding binary target 'Survived')
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
numerical_cols = [col for col in numerical_cols if col in df_cleaned.columns]

print(f"\nNumerical columns to scale: {numerical_cols}")

# Display statistics before scaling
print("\nStatistics BEFORE scaling:")
print(df_cleaned[numerical_cols].describe())

# Apply StandardScaler (mean=0, std=1)
scaler = StandardScaler()
df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

print("\nStatistics AFTER scaling:")
print(df_cleaned[numerical_cols].describe())

# STEP 5: VISUALIZE AND HANDLE OUTLIERS
print("\n" + "=" * 80)
print("STEP 5: DETECTING AND REMOVING OUTLIERS")
print("=" * 80)

# Create boxplots to visualize outliers
fig, axes = plt.subplots(1, len(numerical_cols), figsize=(15, 4))
fig.suptitle('Boxplots for Outlier Detection (After Scaling)', fontsize=16, y=1.02)

for idx, col in enumerate(numerical_cols):
    axes[idx].boxplot(df_cleaned[col].dropna())
    axes[idx].set_title(col)
    axes[idx].set_ylabel('Scaled Value')

plt.tight_layout()
plt.savefig('outliers_boxplot.png', dpi=300, bbox_inches='tight')
print("\nBoxplot saved as 'outliers_boxplot.png'")
plt.show()

# Remove outliers using IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

initial_rows = len(df_cleaned)

# Apply outlier removal to numerical columns
for col in numerical_cols:
    df_cleaned = remove_outliers_iqr(df_cleaned, col)

final_rows = len(df_cleaned)
removed_rows = initial_rows - final_rows

print(f"\nRemoved {removed_rows} rows containing outliers")
print(f"Final dataset shape: {df_cleaned.shape}")

# FINAL RESULTS
print("\n" + "=" * 80)
print("FINAL CLEANED DATASET")
print("=" * 80)

print("\nFirst 10 rows of cleaned dataset:")
print(df_cleaned.head(10))

print("\nFinal Dataset Info:")
print(df_cleaned.info())

print("\nFinal Statistical Summary:")
print(df_cleaned.describe())

# Visualize correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Heatmap of Features', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\nCorrelation heatmap saved as 'correlation_heatmap.png'")
plt.show()

# Save the cleaned dataset
df_cleaned.to_csv('titanic_cleaned.csv', index=False)
print("\nCleaned dataset saved as 'titanic_cleaned.csv'")

print("\n" + "=" * 80)
print("DATA CLEANING AND PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 80)