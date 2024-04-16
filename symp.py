import streamlit as st
import pandas as pd
import joblib

# Load the trained RandomForestClassifier model
rf_model = joblib.load('disease_trained_model.pkl')

# Define the list of symptoms
symptoms_list = [
    'anorexia', 'abdominal_pain', 'anaemia', 'abortions', 'acetone', 'aggression', 'arthrogyposis', 'ankylosis',
    'anxiety', 'bellowing', 'blood_loss', 'blood_poisoning', 'blisters', 'colic', 'Condemnation_of_livers','conjunctivae',
    'coughing', 'depression', 'discomfort', 'dyspnea', 'dysentery', 'diarrhoea', 'dehydration', 'drooling',
    'dull', 'decreased_fertility', 'diffculty_breath', 'emaciation', 'encephalitis', 'fever', 'facial_paralysis',
    'frothing_of_mouth', 'frothing', 'gaseous_stomach', 'highly_diarrhoea', 'high_pulse_rate', 'high_temp',
    'high_proportion', 'hyperaemia', 'hydrocephalus', 'isolation_from_herd', 'infertility', 'intermittent_fever',
    'jaundice', 'ketosis', 'loss_of_appetite', 'lameness', 'lack_of-coordination', 'lethargy', 'lacrimation',
    'milk_flakes', 'milk_watery', 'milk_clots', 'mild_diarrhoea', 'moaning', 'mucosal_lesions', 'milk_fever',
    'nausea', 'nasel_discharges', 'oedema', 'pain', 'painful_tongue', 'pneumonia', 'photo_sensitization',
    'quivering_lips', 'reduction_milk_vields', 'rapid_breathing', 'rumenstasis', 'reduced_rumination',
    'reduced_fertility', 'reduced_fat', 'reduces_feed_intake', 'raised_breathing', 'stomach_pain', 'salivation',
    'stillbirths', 'shallow_breathing', 'swollen_pharyngeal', 'swelling', 'saliva', 'swollen_tongue',
    'tachycardia', 'torticollis', 'udder_swelling', 'udder_heat', 'udder_hardeness', 'udder_redness', 'udder_pain',
    'unwillingness_to_move', 'ulcers', 'vomiting', 'weight_loss', 'weakness'
]
# Define page title and description
st.title("Smart Cattle Disease Detection")
st.title("Choose Disease symptoms from below:s")
# Create a dropdown menu for each symptom
symptom_selections = {symptom: st.selectbox(symptom, ['No', 'Yes']) for symptom in symptoms_list}

# Create a button to make predictions
if st.button('Predict'):
    # Convert selected symptoms to input data format
    input_data = pd.DataFrame()
    for symptom, value in symptom_selections.items():
        input_data[symptom] = [1 if value == 'Yes' else 0]
    
    # Make prediction
    prediction = rf_model.predict(input_data)
    
    # Display prediction
    st.write(f"The predicted disease based on symptoms is: {prediction[0]}")
