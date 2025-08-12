import streamlit as st
import pandas as pd

st.title('🎈 Machine Learning App')

st.info('This is a Machine Learning App')
with st.expander("Data"):
  st.write("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/andrilblaze40/Nwanze-Andrew-Machine-learning/refs/heads/master/.streamlit/cleaned_breast_cancer_data.csv")
  df

  #st.write("**X**")
  #X = df.drop("diagnosis", inplace=True)
  #X
  
  #st.write("**y**")
  #y = df.diagnosis
  #y



