import streamlit as st
import pandas as pd
import pickle

# Load the pickled model
with open('rf_cement.pkl', 'rb') as file:
    model = pickle.load(file)

st.write("""
# Simple Cement Strength Prediction App

This app predicts the **Cement Strength** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    cement = st.sidebar.slider('cement', 140.5, 150.5, 165.8, 200.5)
    slag = st.sidebar.slider('slag', 0, 42, 200, 114)
    ash = st.sidebar.slider('ash', 0, 75, 90, 95)
    water = st.sidebar.slider('Water', 193, 200, 203, 228)
    superplastic = st.sidebar.slider('superplastic', 0, 3, 5, 10)
    coarseagg = st.sidebar.slider('coarseagg', 1000, 950, 830, 1050)
    fineagg = st.sidebar.slider('fineagg', 780, 820, 870, 900)
    age = st.sidebar.slider('Age', 15, 20, 25, 30)
    data = {'cement' : cement,
            'slag': slag,
            'ash': ash,
            'water': water,
            'superplastic': superplastic,
            'coarseagg' : coarseagg,
            'fineagg' : fineagg,
            'age' : age}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)
y_hat_test = model.predict(df)

st.write("b0 is", round(model.intercept_, 3))
st.write("b1 is", round(model.coef_[0], 3))
st.write("b2 is", round(model.coef_[1], 3))
st.write("b3 is", round(model.coef_[2], 3))
st.write("b4 is", round(model.coef_[3], 3))
st.write("b5 is", round(model.coef_[4], 3))
st.write("b6 is", round(model.coef_[5], 3))
st.write("b7 is", round(model.coef_[6], 3))
st.write("b8 is", round(model.coef_[7], 3))
st.write("yhat test is", y_hat_test)