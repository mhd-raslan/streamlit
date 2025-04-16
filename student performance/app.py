import pickle
import streamlit as st
import pandas as pd
import numpy as np


# Set up Streamlit page config
st.set_page_config(page_title="Student Performance Prediction", layout="wide")

# Custom CSS to set background color to sky blue
page_bg_color = """
<style>
    body {
        background-color: #008B8B !important; /* Sky Blue */
    }
    .stApp {
        background-color: #008B8B !important;
    }
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

st.title('Student Performance Prediction App')
st.subheader('Enter Your Data :')

# Input fields for user data
name=st.text_input('Name')
Hours_Studied = st.number_input('Enter Hours Studied',min_value=1,max_value=44)
Attendance = st.number_input('Enter the Attendance of the Student',min_value=60,max_value=100)
Access_to_Resources= st.selectbox('Access to Resources', ['Low', 'Medium', 'High'])
Motivation_Level= st.selectbox('Motivation Level', ['Low', 'Medium', 'High'])

# Prepare input data
input_data = {
    'Hours_Studied': Hours_Studied,
    'Attendance': Attendance,
    'Access_To_Resources': Access_to_Resources,
    'Motivation_Level': Motivation_Level
}

# Convert input to a DataFrame
new_data = pd.DataFrame([input_data])

access_to_rsrc={'Low':1,'Medium':2,'High':3}
new_data['Access_To_Resources']=new_data['Access_To_Resources'].map(access_to_rsrc)

mtvn_mp={'Low':1,'Medium':2,'High':3}
new_data['Motivation_Level']=new_data['Motivation_Level'].map(mtvn_mp)


df = pd.read_csv("features.csv")
columns_list = [col for col in df.columns if col != 'Unnamed: 0']
new_data = new_data.reindex(columns=columns_list, fill_value=0)

with open('LinearRegression-Std.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Make a prediction  
if st.button('Predict'):  
    prediction = loaded_model.predict(new_data)  
    # Display the prediction result  
    st.write(f'### Prediction Result for {name}:')  
    st.write(f'The predicted percentage score is {prediction[0]:.2f}%') 
    if prediction[0]>40:
        st.success('passed')