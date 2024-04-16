import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained RandomForestClassifier model
rf_model = joblib.load('disease_trained_model.pkl')

# Load the saved disease detection model
model_path = 'C:\\Users\\ABEL\\Desktop\\CAPSTONE PROJECT\\cattle.h5'
disease_model = load_model(model_path)

# Define class labels for disease detection model
class_labels = {0: 'mouth_infection', 1: 'healthy', 2: 'lumpy_skin', 3: 'foot_infection'}

# Define list of symptoms for RandomForestClassifier model
symptoms_list = [
    'anorexia', 'abdominal_pain', 'anaemia', 'abortions', 'acetone', 'aggression', 'arthrogyposis', 'ankylosis',
    'anxiety', 'bellowing', 'blood_loss', 'blood_poisoning', 'blisters', 'colic', 'Condemnation_of_livers', 'conjunctivae',
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

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(640, 640)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Function to make predictions using disease detection model
def predict_disease(image_path):
    img_array = preprocess_image(image_path)
    prediction = disease_model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    return class_labels[predicted_class_index]

# Streamlit app
st.title('Smart Cattle Disease Detection')

# Navigation menu
option = st.sidebar.selectbox('Select Model', ['Image Based Disease Detection', 'Symptom Based Disease Prediction'])

if option == 'Image Based Disease Detection':
    # Display the diseases that the app can predict
    st.write("Classes that the model can predict:")
    for disease in class_labels.values():
        st.write(f"- {disease}")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Resize the image to the expected shape (640, 640)
        img_array = preprocess_image(uploaded_file)

        prediction = predict_disease(uploaded_file)
        st.write(f"Predicted Disease: {prediction}")

elif option == 'Symptom Based Disease Prediction':
    # Define page title and description
    st.title("Symptom Based Disease Prediction")
    st.title("Choose Disease symptoms from below:")
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
