import streamlit as st
import pickle
import pandas as pd

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Churn Prediction App")

CreditScore = st.number_input("Credit Score", min_value=300, max_value=850)
geography = st.selectbox("Geography", options=["France", "Spain", "Germany"])
gender = st.selectbox("Gender", options=["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100)
Tenure = st.number_input("Tenure", min_value=0, max_value=10)
Balance = st.number_input("Balance", min_value=0, max_value=100000)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4)
HasCrCard = st.selectbox("Has Credit Card", options=[0, 1])
IsActiveMember = st.selectbox("Is Active Member", options=[0, 1])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0, max_value=200000)




if st.button("Predict"):
    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography': [0 if geography == "France" else 1 if geography == "Germany" else 2],
        'Gender': [1 if gender == "Male" else 0],
        'Age': [age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("The customer is likely to churn.")
    else:
        st.error("The customer is not likely to churn.")
