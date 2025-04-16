import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  MinMaxScaler
from sklearn.preprocessing import LabelEncoder

filename = 'RandomForestClassifier_heart.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.title('Heart Failure  Prediction App')

# Input field for the User
age=st.number_input('Age',min_value=0,max_value=120,value=18)
sex=st.selectbox('Sex',['M','F'])
chest_pain=st.selectbox('Chest Pain Type',['ATA','NAP','ASY','TA'])
resting_bp=st.number_input('Resting BP',min_value=0,max_value=200)
cholesterol=st.number_input('Cholesterol',min_value=0,max_value=603)
fasting_bs=st.selectbox('fasting Blood Sugar',[0,1])
resting_ecg=st.selectbox('Resting ECG',['Normal','ST','LVH'])
max_hr=st.number_input('Maximum HR',min_value=60,max_value=202)
exercise_angina=st.selectbox('Exersice Angina',['N','Y'])
oldpeak=st.number_input('Oldpeak',min_value=0.0,max_value=6.2)
st_slope=st.selectbox('ST Slope',['Up','Flat','Down'])

#Prepare the input data as a dictionary
input_data={'Age':age,"Sex":sex,'ChestPainType':chest_pain,'RestingBp':resting_bp,'Cholestrol':cholesterol,'FastingBS':fasting_bs,'RestingECG':resting_ecg,'MaxHR':max_hr,'ExerciseAngina':exercise_angina,'Oldpeak':oldpeak,'ST_Slope':st_slope}

#Convert input data to Dataframe
new_data=pd.DataFrame([input_data])

#load saved encorders
sex_encorder=LabelEncoder()
sex_encorder.classes_=np.array(['F','M'])

chest_pain_encoder = LabelEncoder()
chest_pain_encoder.classes_=np.array(['ATA','NAP','ASY','TA'])

resting_ecg_encoder = LabelEncoder()
resting_ecg_encoder.classes_=np.array(['Normal','ST','LVH'])

exercise_angina_encoder = LabelEncoder()
exercise_angina_encoder.classes_=np.array(['N','Y'])

st_slope_encoder = LabelEncoder()
st_slope_encoder.classes_=np.array(['Up','Flat','Down'])

#apply label encoding to categorical columns
new_data['Encorder_Sex']=sex_encorder.transform(new_data['Sex'])
new_data["Encoder_ChestPainType"] = chest_pain_encoder.transform(new_data["ChestPainType"])
new_data["Encoder_RestingECG"] = resting_ecg_encoder.transform(new_data["RestingECG"])
new_data["Encoder_ExerciseAngina"] = exercise_angina_encoder.transform(new_data["ExerciseAngina"])
new_data["Encoder_ST_Slope"] = st_slope_encoder.transform(new_data["ST_Slope"])

# Drop original columns as they are already encoded
new_data.drop(["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"], axis=1, inplace=True)

# Load the saved features list
df = pd.read_csv("features.csv")
columns_list = [col for col in df.columns if col != 'Unnamed: 0']

# Reindex to match the original column order
new_data = new_data.reindex(columns=columns_list, fill_value=0)

# Load the saved scaler
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Scale the new data
scaled_data = pd.DataFrame(loaded_scaler.transform(new_data), columns=columns_list)

# Load the RandomForest model
with open('RandomForestClassifier_heart.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Make predictions
prediction = loaded_model.predict(scaled_data)

# Output the prediction
# if prediction[0] == 1:
#     st.error("Prediction: Heart Disease Present")
# else:
#     st.success("Prediction: No Heart Disease")
# Create a button and check for clicks  
if st.button("Check Heart Disease Prediction"):  
    if prediction[0] == 1:  
        st.error("Prediction: Heart Disease Present")  
    else:  
        st.success("Prediction: No Heart Disease")
        st.snow()




