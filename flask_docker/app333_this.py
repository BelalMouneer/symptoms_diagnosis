from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import json

app = Flask(__name__)

# Load your data
data = pd.read_csv('new_dataset.csv')

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
features = list(X_train.columns)
# Train a Multinomial Logistic Regression (Softmax Regression) classifier
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
softmax_model.fit(X_train, y_train)
softmax_predictions = softmax_model.predict(X_test)
softmax_f1_score = f1_score(y_test, softmax_predictions, average='micro')

# Train a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train a Logistic Regression classifier
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)


voting_clf = VotingClassifier(estimators=[('softmax', softmax_model), ('lr', lr_model), ('rf', rf_model)], voting='hard')
voting_clf.fit(X_train, y_train)
voting_predictions = voting_clf.predict(X_test)
voting_f1_score = f1_score(y_test, voting_predictions, average='micro')


print(f"Multinomial Logistic Regression F1-Score: {softmax_f1_score:.4f}")
print(f"Random Forest F1-Score: {f1_score(y_test, rf_model.predict(X_test), average='micro'):.4f}")
print(f"Logistic Regression F1-Score: {f1_score(y_test, lr_model.predict(X_test), average='micro'):.4f}")
print(f"Voting Classifier F1-Score: {voting_f1_score:.4f}")




@app.route('/')
def home():
    return render_template('index333.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms_input = request.form.get('symptoms')
        if symptoms_input is None:
            return render_template('error.html', message='No symptoms provided. Please enter symptoms.')

        symptoms_list = symptoms_input.split(',')

        new_data = pd.DataFrame(0, index=[0], columns=features)

        for i, symptom in enumerate(symptoms_list):
            try:
                index = features.index(f'Symptom_{i+1}_{symptom.lower()}')
            except ValueError:
                print(f"Symptom not found: Symptom_{i+1}_{symptom.lower()}")
                index = -1

            if index != -1:
                new_data[features[index]] = 1
            else:
                print(f"Symptom not found: Symptom_{i+1}_{symptom.lower()}")

        new_data = new_data[symptoms_encoded.columns]

        rf_prediction = le.inverse_transform(rf_model.predict(new_data))[0]
        lr_prediction = le.inverse_transform(lr_model.predict(new_data))[0]
        voting_prediction = le.inverse_transform(voting_clf.predict(new_data))[0]

        return render_template('result333.html', rf_prediction=rf_prediction, lr_prediction=lr_prediction, 
                               voting_prediction=voting_prediction, debug_info=json.dumps(request.form))




if __name__ == '__main__':
    app.run(debug=True)





