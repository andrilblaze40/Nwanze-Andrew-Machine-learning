import streamlit as st
import pandas as pd

st.title('🎈 Machine Learning App')

st.info('This is a Machine Learning App')

df = pd.read_csv("https://raw.githubusercontent.com/andrilblaze40/Nwanze-Andrew-Machine-learning/refs/heads/master/.streamlit/cleaned_breast_cancer_data.csv")
df
