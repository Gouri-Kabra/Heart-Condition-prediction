import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Heartify Web App

Predict condition of your Heart

""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe

def user_input_features():
    Age = st.sidebar.number_input('Enter your age: ')
    Sex  = st.sidebar.selectbox('Sex',('M', 'F'))
    ChestPainType = st.sidebar.selectbox('Chest pain type',('ATA','NAP', 'ASY', 'TA'))
    RestingBP = st.sidebar.number_input('Resting blood pressure: ')
    Cholesterol = st.sidebar.number_input('Cholestoral: ')
    FastingBS = st.sidebar.selectbox('Fasting blood sugar',(0,1))
    RestingECG = st.sidebar.selectbox('Resting electrocardiographic results: ', ('Normal','ST', 'LVH'))
    MaxHR = st.sidebar.number_input('Maximum heart rate achieved: ')
    ExerciseAngina = st.sidebar.selectbox('Exercise induced angina: ',('Y', 'N'))
    Oldpeak = st.sidebar.number_input('oldpeak ')
    ST_Slope = st.sidebar.selectbox('The slope of the peak exercise ST segmen: ', ('Up','Flat', 'Down'))
    

    data = {'Age': Age,
            'Sex': Sex, 
            'ChestPainType': ChestPainType,
            'RestingBP': RestingBP,
            'Cholesterol': Cholesterol,
            'FastingBS': FastingBS,
            'RestingECG': RestingECG,
            'MaxHR': MaxHR,
            'ExerciseAngina':ExerciseAngina,
            'Oldpeak':Oldpeak,
            'ST_Slope':ST_Slope,
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
heart_dataset = pd.read_csv('heart.csv')
heart_dataset = heart_dataset.drop(columns=['HeartDisease'])

df = pd.concat([input_df,heart_dataset],axis=0)

# convert categorical columns to numerical values
df.replace({'Sex':{'M':1, 'F':2},'ChestPainType':{'ATA':1,'NAP':2, 'ASY':3, 'TA':4},'RestingECG':{'Normal':1,'ST':2, 'LVH':3},
                      'ExerciseAngina':{'Y':1, 'N':2},'ST_Slope':{'Up':1,'Flat':2, 'Down':3}},inplace=True)


df = df[:1] # Selects only the first row (the user input data)

st.write(input_df)
# Reads in saved classification model
load_clf = pickle.load(open('model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
# st.write(prediction)
if prediction == 0:
    st.write('Your Heart is healthy :) ')
elif prediction == 1:
    st.write('Your Heart is Unhealthy :( ')    

st.subheader('Prediction Probability')
st.write(prediction_proba)