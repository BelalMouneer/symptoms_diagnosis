from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


data = pd.read_csv('new_dataset.csv')
X = data.drop('Disease', axis=1)
y = data['Disease']
features = X


# One-hot encode symptoms
symptoms_encoded = pd.get_dummies(symptoms)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(symptoms_encoded, target, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train a Logistic Regression classifier
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train) 



features = [
    'Symptom_1_ acidity', 'Symptom_1_ back_pain', 'Symptom_1_ bladder_discomfort', 'Symptom_1_ breathlessness', 'Symptom_1_ burning_micturition', 'Symptom_1_ chest_pain', 'Symptom_1_ chills', 'Symptom_1_ constipation', 'Symptom_1_ continuous_sneezing', 'Symptom_1_ cough', 'Symptom_1_ cramps', 'Symptom_1_ fatigue', 'Symptom_1_ headache', 'Symptom_1_ high_fever', 'Symptom_1_ indigestion', 'Symptom_1_ joint_pain', 'Symptom_1_ mood_swings', 'Symptom_1_ muscle_wasting', 'Symptom_1_ muscle_weakness', 'Symptom_1_ neck_pain', 'Symptom_1_ pain_during_bowel_movements', 'Symptom_1_ patches_in_throat', 'Symptom_1_ pus_filled_pimples', 'Symptom_1_ shivering', 'Symptom_1_ skin_rash', 'Symptom_1_ stiff_neck', 'Symptom_1_ stomach_pain', 'Symptom_1_ sunken_eyes', 'Symptom_1_ vomiting', 'Symptom_1_ weakness_in_limbs', 'Symptom_1_ weight_gain', 'Symptom_1_ weight_loss', 'Symptom_1_ yellowish_skin', 'Symptom_1_chills', 'Symptom_1_itching', 'Symptom_2_ abdominal_pain', 'Symptom_2_ acidity', 'Symptom_2_ anxiety', 'Symptom_2_ blackheads', 'Symptom_2_ bladder_discomfort', 'Symptom_2_ blister', 'Symptom_2_ breathlessness', 'Symptom_2_ bruising',
    'Symptom_2_ chest_pain', 'Symptom_2_ chills', 'Symptom_2_ cold_hands_and_feets', 'Symptom_2_ cough', 'Symptom_2_ cramps', 
    'Symptom_2_ dehydration', 'Symptom_2_ dizziness', 'Symptom_2_ fatigue', 'Symptom_2_ foul_smell_of urine', 'Symptom_2_ headache', 
    'Symptom_2_ high_fever', 'Symptom_2_ indigestion', 'Symptom_2_ joint_pain', 'Symptom_2_ knee_pain', 'Symptom_2_ lethargy', 
    'Symptom_2_ loss_of_appetite', 'Symptom_2_ mood_swings', 'Symptom_2_ nausea', 'Symptom_2_ neck_pain', 'Symptom_2_ nodal_skin_eruptions', 'Symptom_2_ pain_during_bowel_movements', 'Symptom_2_ pain_in_anal_region', 'Symptom_2_ patches_in_throat', 'Symptom_2_ pus_filled_pimples', 'Symptom_2_ restlessness', 'Symptom_2_ shivering', 'Symptom_2_ skin_peeling', 'Symptom_2_ skin_rash', 'Symptom_2_ stiff_neck', 'Symptom_2_ stomach_pain', 'Symptom_2_ sunken_eyes', 'Symptom_2_ sweating', 'Symptom_2_ swelling_joints', 'Symptom_2_ ulcers_on_tongue', 'Symptom_2_ vomiting', 'Symptom_2_ weakness_in_limbs', 'Symptom_2_ weakness_of_one_body_side', 'Symptom_2_ weight_gain', 'Symptom_2_ weight_loss', 'Symptom_2_ yellowish_skin', 'Symptom_3_ abdominal_pain', 'Symptom_3_ altered_sensorium', 'Symptom_3_ anxiety', 'Symptom_3_ blackheads', 'Symptom_3_ blister', 'Symptom_3_ bloody_stool', 'Symptom_3_ blurred_and_distorted_vision', 'Symptom_3_ breathlessness', 'Symptom_3_ bruising', 'Symptom_3_ burning_micturition', 'Symptom_3_ chest_pain', 'Symptom_3_ chills', 'Symptom_3_ cold_hands_and_feets', 'Symptom_3_ continuous_feel_of_urine', 'Symptom_3_ cough', 'Symptom_3_ dark_urine', 'Symptom_3_ dehydration', 'Symptom_3_ diarrhoea', 'Symptom_3_ dischromic _patches', 'Symptom_3_ dizziness', 'Symptom_3_ extra_marital_contacts', 'Symptom_3_ fatigue', 'Symptom_3_ foul_smell_of urine', 'Symptom_3_ headache', 'Symptom_3_ high_fever', 'Symptom_3_ hip_joint_pain', 'Symptom_3_ joint_pain', 'Symptom_3_ knee_pain', 'Symptom_3_ lethargy', 'Symptom_3_ loss_of_appetite', 'Symptom_3_ loss_of_balance', 'Symptom_3_ mood_swings', 'Symptom_3_ movement_stiffness', 'Symptom_3_ nausea', 'Symptom_3_ neck_pain', 'Symptom_3_ nodal_skin_eruptions', 'Symptom_3_ obesity', 'Symptom_3_ pain_in_anal_region', 'Symptom_3_ red_sore_around_nose', 'Symptom_3_ restlessness', 'Symptom_3_ scurring', 'Symptom_3_ silver_like_dusting', 'Symptom_3_ skin_peeling', 'Symptom_3_ spinning_movements', 'Symptom_3_ stomach_pain', 'Symptom_3_ sweating', 'Symptom_3_ swelling_joints', 'Symptom_3_ swelling_of_stomach', 'Symptom_3_ ulcers_on_tongue', 'Symptom_3_ vomiting', 'Symptom_3_ watering_from_eyes', 'Symptom_3_ weakness_of_one_body_side', 'Symptom_3_ weight_loss', 'Symptom_3_ yellowish_skin', 'Symptom_4_ abdominal_pain', 'Symptom_4_ altered_sensorium', 'Symptom_4_ bloody_stool', 'Symptom_4_ blurred_and_distorted_vision', 'Symptom_4_ breathlessness', 'Symptom_4_ burning_micturition', 'Symptom_4_ chest_pain', 'Symptom_4_ continuous_feel_of_urine', 'Symptom_4_ cough', 'Symptom_4_ dark_urine', 'Symptom_4_ diarrhoea', 'Symptom_4_ dischromic _patches', 'Symptom_4_ distention_of_abdomen', 'Symptom_4_ dizziness', 'Symptom_4_ excessive_hunger', 'Symptom_4_ extra_marital_contacts', 'Symptom_4_ family_history', 'Symptom_4_ fatigue', 'Symptom_4_ headache', 'Symptom_4_ high_fever', 'Symptom_4_ hip_joint_pain', 'Symptom_4_ irregular_sugar_level', 'Symptom_4_ irritation_in_anus', 'Symptom_4_ lack_of_concentration', 'Symptom_4_ lethargy', 'Symptom_4_ loss_of_appetite', 'Symptom_4_ loss_of_balance', 'Symptom_4_ mood_swings', 'Symptom_4_ movement_stiffness', 'Symptom_4_ nausea', 'Symptom_4_ obesity', 'Symptom_4_ painful_walking', 'Symptom_4_ passage_of_gases', 'Symptom_4_ red_sore_around_nose', 'Symptom_4_ restlessness', 'Symptom_4_ scurring', 'Symptom_4_ silver_like_dusting', 'Symptom_4_ small_dents_in_nails', 'Symptom_4_ spinning_movements', 'Symptom_4_ spotting_ urination', 'Symptom_4_ sweating', 'Symptom_4_ swelling_joints', 'Symptom_4_ swelling_of_stomach', 'Symptom_4_ swollen_legs', 'Symptom_4_ vomiting', 'Symptom_4_ watering_from_eyes', 'Symptom_4_ weight_loss', 'Symptom_4_ yellow_crust_ooze', 'Symptom_4_ yellowing_of_eyes', 'Symptom_4_ yellowish_skin', 'Symptom_5_ abdominal_pain', 'Symptom_5_ blurred_and_distorted_vision', 'Symptom_5_ breathlessness', 'Symptom_5_ chest_pain', 'Symptom_5_ cough', 'Symptom_5_ dark_urine', 'Symptom_5_ diarrhoea', 'Symptom_5_ distention_of_abdomen', 'Symptom_5_ dizziness', 'Symptom_5_ excessive_hunger', 'Symptom_5_ family_history', 'Symptom_5_ fatigue', 'Symptom_5_ headache', 'Symptom_5_ high_fever', 'Symptom_5_ history_of_alcohol_consumption', 'Symptom_5_ inflammatory_nails', 'Symptom_5_ internal_itching', 'Symptom_5_ irregular_sugar_level', 'Symptom_5_ irritation_in_anus', 'Symptom_5_ lack_of_concentration', 'Symptom_5_ lethargy', 'Symptom_5_ loss_of_appetite', 'Symptom_5_ loss_of_balance', 'Symptom_5_ mucoid_sputum', 'Symptom_5_ nausea', 'Symptom_5_ painful_walking', 'Symptom_5_ passage_of_gases', 'Symptom_5_ small_dents_in_nails', 'Symptom_5_ spotting_ urination', 'Symptom_5_ stiff_neck', 'Symptom_5_ sweating', 'Symptom_5_ swelling_joints', 'Symptom_5_ swollen_blood_vessels', 'Symptom_5_ swollen_legs', 'Symptom_5_ unsteadiness', 'Symptom_5_ yellow_crust_ooze', 'Symptom_5_ yellowing_of_eyes', 'Symptom_5_ yellowish_skin', 'Symptom_6_ abdominal_pain', 'Symptom_6_ blurred_and_distorted_vision', 'Symptom_6_ breathlessness', 'Symptom_6_ chest_pain', 'Symptom_6_ constipation', 'Symptom_6_ dark_urine', 'Symptom_6_ depression', 'Symptom_6_ diarrhoea', 'Symptom_6_ dizziness', 'Symptom_6_ family_history', 'Symptom_6_ fast_heart_rate', 'Symptom_6_ fluid_overload', 'Symptom_6_ headache', 'Symptom_6_ high_fever', 'Symptom_6_ history_of_alcohol_consumption', 'Symptom_6_ inflammatory_nails', 'Symptom_6_ internal_itching', 'Symptom_6_ loss_of_appetite', 'Symptom_6_ malaise', 'Symptom_6_ mucoid_sputum', 'Symptom_6_ nausea', 'Symptom_6_ obesity', 'Symptom_6_ painful_walking', 'Symptom_6_ prominent_veins_on_calf', 'Symptom_6_ puffy_face_and_eyes', 'Symptom_6_ stiff_neck', 'Symptom_6_ sweating', 'Symptom_6_ swelled_lymph_nodes', 'Symptom_6_ swollen_blood_vessels', 'Symptom_6_ unsteadiness', 'Symptom_6_ yellowing_of_eyes', 'Symptom_6_ yellowish_skin', 'Symptom_7_ abdominal_pain', 'Symptom_7_ blurred_and_distorted_vision', 'Symptom_7_ breathlessness', 'Symptom_7_ constipation', 'Symptom_7_ dark_urine', 'Symptom_7_ depression', 'Symptom_7_ diarrhoea', 'Symptom_7_ enlarged_thyroid', 'Symptom_7_ excessive_hunger', 'Symptom_7_ fast_heart_rate', 'Symptom_7_ fluid_overload', 'Symptom_7_ headache', 'Symptom_7_ irritability', 'Symptom_7_ loss_of_appetite', 'Symptom_7_ malaise', 'Symptom_7_ mild_fever', 'Symptom_7_ muscle_pain', 'Symptom_7_ nausea', 'Symptom_7_ obesity', 'Symptom_7_ phlegm', 'Symptom_7_ prominent_veins_on_calf', 'Symptom_7_ puffy_face_and_eyes', 'Symptom_7_ sweating', 'Symptom_7_ swelled_lymph_nodes', 'Symptom_7_ yellow_urine', 'Symptom_7_ yellowing_of_eyes', 'Symptom_8_ abdominal_pain', 'Symptom_8_ brittle_nails', 'Symptom_8_ chest_pain', 'Symptom_8_ diarrhoea', 'Symptom_8_ drying_and_tingling_lips', 'Symptom_8_ enlarged_thyroid', 'Symptom_8_ excessive_hunger', 'Symptom_8_ increased_appetite', 'Symptom_8_ irritability', 'Symptom_8_ loss_of_appetite', 'Symptom_8_ malaise', 'Symptom_8_ mild_fever', 'Symptom_8_ muscle_pain', 'Symptom_8_ muscle_weakness', 'Symptom_8_ nausea', 'Symptom_8_ phlegm', 'Symptom_8_ sweating', 'Symptom_8_ swelled_lymph_nodes', 'Symptom_8_ visual_disturbances', 'Symptom_8_ yellow_urine', 'Symptom_8_ yellowing_of_eyes', 'Symptom_9_ abdominal_pain', 'Symptom_9_ brittle_nails', 'Symptom_9_ chest_pain', 'Symptom_9_ diarrhoea', 'Symptom_9_ drying_and_tingling_lips', 'Symptom_9_ fast_heart_rate', 'Symptom_9_ increased_appetite', 'Symptom_9_ irritability', 'Symptom_9_ loss_of_appetite', 'Symptom_9_ malaise', 'Symptom_9_ mild_fever', 'Symptom_9_ muscle_weakness', 'Symptom_9_ pain_behind_the_eyes', 'Symptom_9_ phlegm', 'Symptom_9_ polyuria', 'Symptom_9_ slurred_speech', 'Symptom_9_ swelled_lymph_nodes', 'Symptom_9_ swollen_extremeties', 'Symptom_9_ throat_irritation', 'Symptom_9_ toxic_look_(typhos)', 'Symptom_9_ visual_disturbances', 'Symptom_9_ yellowing_of_eyes', 'Symptom_10_ abnormal_menstruation', 'Symptom_10_ acute_liver_failure', 'Symptom_10_ back_pain', 'Symptom_10_ belly_pain', 'Symptom_10_ depression', 'Symptom_10_ fast_heart_rate', 'Symptom_10_ irritability', 'Symptom_10_ malaise', 'Symptom_10_ mild_fever', 'Symptom_10_ muscle_pain', 'Symptom_10_ pain_behind_the_eyes',
    'Symptom_10_ polyuria', 'Symptom_10_ receiving_blood_transfusion',
'Symptom_10_ red_spots_over_body',
'Symptom_10_ redness_of_eyes',
'Symptom_10_ rusty_sputum',
'Symptom_10_ slurred_speech',
'Symptom_10_ swollen_extremeties',
'Symptom_10_ throat_irritation',
'Symptom_10_ toxic_look_(typhos)',
'Symptom_10_ yellowing_of_eyes',
'Symptom_11_ abnormal_menstruation',
'Symptom_11_ acute_liver_failure',
'Symptom_11_ back_pain',
'Symptom_11_ belly_pain',
'Symptom_11_ coma',
'Symptom_11_ depression',
'Symptom_11_ irritability',
'Symptom_11_ malaise',
'Symptom_11_ muscle_pain',
'Symptom_11_ palpitations',
'Symptom_11_ receiving_blood_transfusion',
'Symptom_11_ receiving_unsterile_injections',
'Symptom_11_ red_spots_over_body',
'Symptom_11_ redness_of_eyes',
'Symptom_11_ rusty_sputum',
'Symptom_11_ sinus_pressure',
'Symptom_11_ swelled_lymph_nodes',
'Symptom_11_ yellowing_of_eyes',
'Symptom_12_ abnormal_menstruation',
'Symptom_12_ coma',
'Symptom_12_ irritability',
'Symptom_12_ malaise',
'Symptom_12_ muscle_pain',
'Symptom_12_ palpitations',
'Symptom_12_ receiving_unsterile_injections',
'Symptom_12_ runny_nose',
'Symptom_12_ sinus_pressure',
'Symptom_12_ stomach_bleeding',
'Symptom_12_ swelled_lymph_nodes',
'Symptom_13_ abnormal_menstruation',
'Symptom_13_ congestion',
'Symptom_13_ malaise',
'Symptom_13_ muscle_pain',
'Symptom_13_ phlegm',
'Symptom_13_ red_spots_over_body',
'Symptom_13_ runny_nose',
'Symptom_13_ stomach_bleeding',
'Symptom_14_ chest_pain',
'Symptom_14_ congestion',
'Symptom_14_ phlegm',
'Symptom_14_ red_spots_over_body',
'Symptom_15_ blood_in_sputum',
'Symptom_15_ chest_pain',
'Symptom_15_ loss_of_smell',
'Symptom_16_ blood_in_sputum',
'Symptom_16_ loss_of_smell',
'Symptom_16_ muscle_pain',
'Symptom_17_ muscle_pain'
]



# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define a route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms_input = request.form.get('symptoms')
        if symptoms_input is None:
            return render_template('error.html', message='No symptoms provided. Please enter symptoms.')

        # Print request data for debugging
        print(f"Request Data: {request.data}")
        print(f"Symptoms Input: {symptoms_input}")

        symptoms_list = symptoms_input.split(',')

        new_data = pd.DataFrame(0, index=[0], columns=features)

        for i, symptom in enumerate(symptoms_list):
            try:
                index = features.index(f'Symptom_{i+1}_{symptom.lower()}')
            except ValueError:
                print(f"Symptom not found: Symptom_{i+1}_{symptom.lower()}")
                # Add additional print statements if needed
                # Print the contents of features to help with debugging
                # print("Features:", features)
                index = -1  # or some default value

            if index != -1:
                new_data[features[index]] = 1
            else:
                # Handle the case where the symptom is not found
                print(f"Symptom not found: Symptom_{i+1}_{symptom.lower()}")


            new_data[features[index]] = 1


        # Ensure the order of columns in new_data matches that in symptoms_encoded
        new_data = new_data[symptoms_encoded.columns]

        # Assuming 'rf_model' and 'lr_model' are trained models
        rf_prediction = rf_model.predict(new_data)
        lr_prediction = lr_model.predict(new_data)
        
        app.logger.info(f"Form Data: {request.form}")
        print(f"RF Prediction: {rf_prediction}")
        print(f"LR Prediction: {lr_prediction}")

        return render_template('result.html', rf_prediction=rf_prediction[0], lr_prediction=lr_prediction[0], debug_info=json.dumps(request.form))

if __name__ == '__main__':
    app.run(debug=True)


