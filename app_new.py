from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

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

# Train a Multinomial Logistic Regression (Softmax Regression) classifier
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
softmax_model.fit(symptoms_encoded, target_encoded)


def preprocess_input(symptom_string, symptoms_encoded_columns):
    name = symptom_string.split(',')
    sorted_symptoms = sorted(name)  # Sort the symptoms alphabetically
    new_data = pd.DataFrame(0, index=[0], columns=symptoms_encoded_columns)
    for i, symptom in enumerate(sorted_symptoms):
        column_name = 'Symptom_' + str(i+1) + '_' + symptom
        if column_name in new_data.columns:
            new_data[column_name] = 1
    return sorted_symptoms, new_data


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symptoms_input = request.form['symptoms']
        sorted_symptoms, new_data = preprocess_input(symptoms_input, symptoms_encoded.columns)

        # Predict with Multinomial Logistic Regression (Softmax Regression)
        softmax_prediction = softmax_model.predict(new_data)

        # Decode the predicted label to get the original class
        predicted_disease = le.inverse_transform([softmax_prediction[0]])

        return render_template('index_new.html', symptoms=sorted_symptoms, prediction=predicted_disease[0])

    return render_template('index_new.html', symptoms=None, prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
