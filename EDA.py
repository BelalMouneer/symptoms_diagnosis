import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv('original_data.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check column names
print("\nColumn Names:")
print(data.columns)

# Distribution of Diseases
sns.countplot(x='Disease', data=data)
plt.title('Distribution of Diseases')
plt.show()

# Distribution of Numerical Features
num_features = data.select_dtypes(include=['float64', 'int64']).columns
for feature in num_features:
    sns.displot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Standardize the numeric variables using StandardScaler
scaler = StandardScaler()
data_standardized = data.copy()


# Pairplot for visualizing relationships
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))
sns.pairplot(data_standardized, hue='Disease', diag_kind='kde')
plt.show()

