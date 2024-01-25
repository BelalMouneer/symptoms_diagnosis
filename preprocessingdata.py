import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from main_prediction_app import *
# Step 1: Load the data
data = pd.read_csv('original_data.csv')  # Replace 'your_dataset.csv' with the actual file path
print("Initial Data:")
print(data.head())

# Step 2: Explore the dataset and visualize initial data
disease_counts_original = data['Disease'].value_counts()
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
sns.barplot(y=disease_counts_original.index, x=disease_counts_original.values, alpha=0.6)
plt.title('Distribution of Diseases Before Preprocessing')
plt.ylabel('Diseases', fontsize=12)
plt.xlabel('Counts', fontsize=12)

# Step 3: Preprocess the data
# You may not need to replace empty strings with NaN, as you already have symptoms in columns

# Drop empty columns (those with all missing values after replacement)
data.dropna(axis=1, how='all', inplace=True)

# Remove duplicate records
data.drop_duplicates(inplace=True)

# Save the final dataset after preprocessing
data.to_csv('preprocessed_new_dataset.csv', index=False)

# Melt the dataframe to have [Disease, Features] as the only non-empty columns
data = pd.melt(data, id_vars=['Disease'], value_vars=data.columns[1:])

print("Preprocessed Data:")
print(data.head())

# Step 4: Visualize the preprocessed data
plt.subplot(1, 2, 2)
sns.barplot(x=data['Disease'], y=data['value'], alpha=0.6)
plt.xlabel('Disease', fontsize=12)
plt.ylabel('Symptoms', fontsize=12)
plt.title('Preprocessed Data')

plt.tight_layout()
plt.show()

# Additional: Describe the datasets
print("\nOriginal Dataset Description:")
print(data.describe())

print("\nPreprocessed Dataset Description:")
print(data.describe())

# Step 5: Split the data into training and testing sets
train_index, test_index = train_test_split(data.index, test_size=0.2, random_state=42)
train_data = data.loc[train_index]
test_data = data.loc[test_index]

print("\nTraining Data:")
print(train_data.head())
print("Testing Data:")
print(test_data.head())

# Step 6: Train a machine learning model on the training data
X_train, X_test, y_train, y_test = train_test_split(symptoms_encoded, target, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train a Logistic Regression classifier
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)


print("\nTrained Model:")
print(lr_model.predict(train_data[['Disease', 'value']]))

# Step 7: Evaluate the performance of the trained model on the testing data
test_pred = lr_model.predict(test_data[['Disease', 'value']])
conf_mat = confusion_matrix(test_data['Disease'], test_pred)  
accuracy = accuracy_score(test_data['Target'], test_pred)
print(f"\nConfusion Matrix: {conf_mat}")
print(f"Accuracy: {accuracy:.3f}")

# Step 8: Visualize the performance of the trained model on the testing data
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, cmap='Blues')
plt.xlabel('True Positives')
plt.ylabel('False Positives')
plt.title('Confusion Matrix')
plt.show()

print(f"Accuracy: {accuracy:.3f}")
