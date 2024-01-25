import requests

url = "http://localhost:5000/predict"

# Symptoms data as a list
symptoms_data = [
    'symptoms_itching', 'symptoms_skin_rash', 'symptoms_nodal_skin_eruptions', 'symptoms_dischromic_patches',
    'symptoms_continuous_sneezing', 'symptoms_shivering', 'symptoms_chills', 'symptoms_watering_from_eyes',
    'symptoms_stomach_pain', 'symptoms_acidity', 'symptoms_ulcers_on_tongue', 'symptoms_vomiting', 'symptoms_cough',
    'symptoms_chest_pain', 'symptoms_yellowish_skin', 'symptoms_nausea', 'symptoms_loss_of_appetite',
    'symptoms_abdominal_pain', 'symptoms_yellowing_of_eyes', 'symptoms_burning_micturition',
    'symptoms_spotting_urination', 'symptoms_passage_of_gases', 'symptoms_internal_itching', 'symptoms_indigestion',
    'symptoms_muscle_wasting', 'symptoms_patches_in_throat', 'symptoms_high_fever', 'symptoms_extra_marital_contacts',
    'symptoms_fatigue', 'symptoms_weight_loss', 'symptoms_restlessness', 'symptoms_lethargy',
    'symptoms_irregular_sugar_level', 'symptoms_blurred_and_distorted_vision', 'symptoms_obesity',
    'symptoms_excessive_hunger', 'symptoms_increased_appetite', 'symptoms_polyuria', 'symptoms_sunken_eyes',
    'symptoms_dehydration', 'symptoms_diarrhoea', 'symptoms_breathlessness', 'symptoms_family_history',
    'symptoms_mucoid_sputum', 'symptoms_headache', 'symptoms_dizziness', 'symptoms_loss_of_balance',
    'symptoms_lack_of_concentration', 'symptoms_stiff_neck', 'symptoms_depression', 'symptoms_irritability',
    'symptoms_visual_disturbances', 'symptoms_back_pain', 'symptoms_weakness_in_limbs', 'symptoms_neck_pain',
    'symptoms_weakness_of_one_body_side', 'symptoms_altered_sensorium', 'symptoms_dark_urine', 'symptoms_sweating',
    'symptoms_muscle_pain', 'symptoms_mild_fever', 'symptoms_swelled_lymph_nodes', 'symptoms_malaise',
    'symptoms_red_spots_over_body', 'symptoms_joint_pain', 'symptoms_pain_behind_the_eyes', 'symptoms_constipation',
    'symptoms_toxic_look_typhos', 'symptoms_belly_pain', 'symptoms_yellow_urine', 'symptoms_receiving_blood_transfusion',
    'symptoms_receiving_unsterile_injections', 'symptoms_coma', 'symptoms_stomach_bleeding',
    'symptoms_acute_liver_failure', 'symptoms_swelling_of_stomach', 'symptoms_distention_of_abdomen',
    'symptoms_history_of_alcohol_consumption', 'symptoms_fluid_overload', 'symptoms_phlegm', 'symptoms_blood_in_sputum',
    'symptoms_throat_irritation', 'symptoms_redness_of_eyes', 'symptoms_sinus_pressure', 'symptoms_runny_nose',
    'symptoms_congestion', 'symptoms_loss_of_smell', 'symptoms_fast_heart_rate', 'symptoms_rusty_sputum',
    'symptoms_pain_during_bowel_movements', 'symptoms_pain_in_anal_region', 'symptoms_bloody_stool',
    'symptoms_irritation_in_anus', 'symptoms_cramps', 'symptoms_bruising', 'symptoms_swollen_legs',
    'symptoms_swollen_blood_vessels', 'symptoms_prominent_veins_on_calf', 'symptoms_weight_gain',
    'symptoms_cold_hands_and_feets', 'symptoms_mood_swings', 'symptoms_puffy_face_and_eyes',
    'symptoms_enlarged_thyroid', 'symptoms_brittle_nails', 'symptoms_swollen_extremeties',
    'symptoms_abnormal_menstruation', 'symptoms_muscle_weakness', 'symptoms_anxiety', 'symptoms_slurred_speech',
    'symptoms_palpitations', 'symptoms_drying_and_tingling_lips', 'symptoms_knee_pain', 'symptoms_hip_joint_pain',
    'symptoms_swelling_joints', 'symptoms_painful_walking', 'symptoms_movement_stiffness',
    'symptoms_spinning_movements', 'symptoms_unsteadiness', 'symptoms_pus_filled_pimples', 'symptoms_blackheads',
    'symptoms_scurring', 'symptoms_bladder_discomfort', 'symptoms_foul_smell_of_urine',
    'symptoms_continuous_feel_of_urine', 'symptoms_skin_peeling', 'symptoms_silver_like_dusting',
    'symptoms_small_dents_in_nails', 'symptoms_inflammatory_nails', 'symptoms_blister',
    'symptoms_red_sore_around_nose', 'symptoms_yellow_crust_ooze', 'symptoms_chills'
]

# Send a POST request with the JSON data
response = requests.post(url, json={"symptoms": symptoms_data})

# Print the response
print(response.json())
