import streamlit as st
import joblib

model = joblib.load('model_joblib_test')

st.title("Insurance Cost Prediction")

age = st.number_input("Age", min_value=0, max_value=100, value=25)
sex = st.selectbox("Sex",["Male","Female"])
bmi = st.number_input("BMI", min_value=0, max_value=100, value=25)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("smoker", ["yes","no"])

region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

region_map = {
    "Southwest": 1,
    "Southeast": 2,
    "Northwest": 3,
    "Northeast": 4
}
smoker_map = {
  "yes":1,
  "no":0
}
sex_map = {
  "Male":1,
  "Female":0
}



if st.button("Predict"):
    region_num = region_map[region]
    smoker_num = smoker_map[smoker]
    sex_num = sex_map[sex]
    result = model.predict([[age, sex_num, bmi, children, smoker_num,region_num]])
    st.write("Insurance Cost: ₹", result[0])