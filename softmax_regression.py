import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# Load your data
data = pd.read_csv('new_dataset.csv')  # Replace 'your_data.csv' with your actual file name or path

# Extract relevant columns
symptoms = data[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5',
                  'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10',
                  'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15',
                  'Symptom_16', 'Symptom_17']]
target = data['Disease']

# Label encode the target variable
le = LabelEncoder()
target_encoded = le.fit_transform(target)

# One-hot encode symptoms
symptoms_encoded = pd.get_dummies(symptoms)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(symptoms_encoded, target_encoded, test_size=0.2, random_state=42)

# Train a Multinomial Logistic Regression (Softmax Regression) classifier
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
softmax_model.fit(X_train, y_train)
softmax_predictions = softmax_model.predict(X_test)
softmax_f1_score = f1_score(y_test, softmax_predictions, average='micro')  # Add average='micro' to handle multiclass classification
print(f"Multinomial Logistic Regression F1-Score: {softmax_f1_score:.4f}")


def preprocess_input(symptom_string, symptoms_encoded_columns):
    name = symptom_string.split(',')
    sorted_symptoms = sorted(name)  # Sort the symptoms alphabetically
    new_data = pd.DataFrame(0, index=[0], columns=symptoms_encoded_columns)
    for i, symptom in enumerate(sorted_symptoms):
        column_name = 'Symptom_' + str(i+1) + '_' + symptom
        if column_name in new_data.columns:
            new_data[column_name] = 1
        #else:
            #print(f"Warning: Column {column_name} not found in the training data. Check the symptom name and position.")
    return sorted_symptoms, new_data

# Usage
frontend = " infection, skin_rash, dischromic _patches, itching, nodal_skin_eruptions"
sorted_symptoms, new_data = preprocess_input(frontend, symptoms_encoded.columns)

# Predict with Multinomial Logistic Regression (Softmax Regression)
softmax_prediction = softmax_model.predict(new_data)

# Decode the predicted label to get the original class
predicted_disease = le.inverse_transform([softmax_prediction[0]])

print(f"Symptoms Order: {sorted_symptoms}")
print(f"Multinomial Logistic Regression Prediction: {predicted_disease[0]}")
