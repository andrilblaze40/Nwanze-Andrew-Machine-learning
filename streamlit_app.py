import streamlit as st
import pandas as pd

st.title('🎈 Machine Learning App')

st.info('This is a Machine Learning App')
with st.expander("Data"):
  st.write("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/andrilblaze40/Nwanze-Andrew-Machine-learning/refs/heads/master/.streamlit/cleaned_breast_cancer_data.csv")
  df

  st.write("**X**")
  X = df.drop("diagnosis", axis=1)
  X
  
  st.write("**y**")
  y = df.diagnosis
  y
with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='symmetry_mean', y='texture_mean', color='diagnosis')
      





