import streamlit as st
import joblib
import numpy as np

st.set_page_config(
page_title="Credit Score Classifier",
page_icon="💳",
layout="centered"
)

st.title("💳 Credit Score Band Classifier")

st.write("Enter financial details to predict credit score band")

model = joblib.load("credit_model.pkl")

income = st.number_input("Annual Income",20000,200000)

balance = st.number_input("Monthly Balance",1000,100000)

cards = st.number_input("Number of Credit Cards",1,15)

loan = st.number_input("Loan Amount",0,1000000)

late = st.number_input("Late Payments",0,50)

util = st.number_input("Credit Utilization %",1,100)

if st.button("Predict Credit Score"):

    data = np.array([[income,
    balance,
    cards,
    loan,
    late,
    util]])

    prediction = model.predict(data)

    labels = ['Poor','Average','Good','Excellent']

    result = labels[prediction[0]]

    st.subheader("Result")

    if result=="Poor":

        st.error("Credit Band : Poor")

    elif result=="Average":

        st.warning("Credit Band : Average")

    elif result=="Good":

        st.info("Credit Band : Good")

    else:

        st.success("Credit Band : Excellent")
