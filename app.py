import streamlit as st # type: ignore
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the trained model
model =tf.keras.models.load_model('model.h5')

## Load the encoder and scalar
with open('label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)

with open('ohe.pkl','rb') as file:
    ohe=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


## Streamlit app
st.title('Customer Churn Prediction')
## User Input
geography = st.selectbox('Geography',ohe.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',16,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out(['Geography']))


input_df = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

## Labeling Gender Column
# input_df['Gender'] = label_encoder.transform(input_df['Gender'])

## Applying Standard Scaler on Df
input_df_scaled = scaler.transform(input_df)

prediction = model.predict(input_df_scaled)

prediction_proba = prediction[0][0]

st.write('Churn Probability',prediction_proba)

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
