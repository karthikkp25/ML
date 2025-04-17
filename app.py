import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load your trained model objects
classifier = joblib.load("classifier.pkl")
ct = joblib.load("column_transformer.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Prediction function
def predict_thyroid_recurrence(user_input_dict):
    user_df = pd.DataFrame([user_input_dict])

    columns_to_drop = [
        'Hx Radiothreapy',
        'Hx Smoking',
        'Thyroid Function',
        'Pathology',
        'Physical Examination',
        'Adenopathy'
    ]
    user_df = user_df.drop(columns=columns_to_drop)

    user_encoded = ct.transform(user_df)
    user_encoded[:, -1] = scaler.transform(user_encoded[:, -1].reshape(-1, 1))

    prediction = classifier.predict(user_encoded)
    prediction_label = le.inverse_transform(prediction)
    return prediction_label[0]

# Streamlit UI
st.title("🔬 Thyroid Cancer Recurrence Prediction")
st.markdown("Fill in the following fields to get a recurrence prediction:")

age = st.number_input("Age", min_value=1, max_value=120, value=35)

gender = st.selectbox("Gender", ['F', 'M'])
smoking = st.selectbox("Smoking", ['Yes', 'No'])
hx_smoking = st.selectbox("Hx Smoking", ['Yes', 'No'])
hx_radiotherapy = st.selectbox("Hx Radiotherapy", ['Yes', 'No'])

thyroid_function = st.selectbox("Thyroid Function", ['Euthyroid', 'Hypothyroid', 'Hyperthyroid'])

physical_exam = st.selectbox("Physical Examination", [
    'Single nodular goiter-left',
    'Single nodular goiter-right',
    'Multinodular goiter'
])

adenopathy = st.selectbox("Adenopathy", [
    'No', 'Right', 'Left', 'Bilateral', 'Posterior', 'Extensive'
])

pathology = st.selectbox("Pathology", [
    'Micropapillary', 'Hurthel cell', 'Papillary'
])

focality = st.selectbox("Focality", ['Uni-Focal', 'Multi-Focal'])
risk = st.selectbox("Risk", ['Low', 'Intermediate', 'High'])

t_stage = st.selectbox("T Stage", ['T1a','T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'])
n_stage = st.selectbox("N Stage", ['N0', 'N1a', 'N1b'])
m_stage = st.selectbox("M Stage", ['M0', 'M1'])

stage = st.selectbox("Stage", ['I', 'II', 'III', 'IVA', 'IVB'])
response = st.selectbox("Response", ['Excellent', 'Indeterminate', 'Structural Incomplete', 'Biochemical Incomplete'])

# Prediction Trigger
if st.button("🔮 Predict"):
    user_input = {
        'Age': age,
        'Gender': gender,
        'Smoking': smoking,
        'Hx Smoking': hx_smoking,
        'Hx Radiothreapy': hx_radiotherapy,
        'Thyroid Function': thyroid_function,
        'Physical Examination': physical_exam,
        'Adenopathy': adenopathy,
        'Pathology': pathology,
        'Focality': focality,
        'Risk': risk,
        'T': t_stage,
        'N': n_stage,
        'M': m_stage,
        'Stage': stage,
        'Response': response
    }

    result = predict_thyroid_recurrence(user_input)
    st.success(f"🎯 Predicted Recurrence Status: **{result}**")
