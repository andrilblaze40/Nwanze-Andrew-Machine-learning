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
 
# Input features
with st.sidebar:
  st.header('Input features')
  
  texture_worst = st.slider('texture worse', 32.1, 59.6, 43.9)
  texture_se = st.slider('texture_se', 13.1, 21.5, 17.2)
  texture_mean = st.slider('texture_mean', 172.0, 231.0, 201.0)
  symmetry_worst= st.slider('symmetry_worst', 2700.0, 6300.0, 4207.0) 
  symmetry_se= st.slider('symmetry_se', 2700.0, 6300.0, 4207.0)
  symmetry_mean= st.slider('symmetry_mean', 2700.0, 6300.0, 4207.0)
  smoothness_worst= st.slider('smoothness_worst', 2700.0, 6300.0, 4207.0)
  radius_worst= st.slider('radius_worst', 2700.0, 6300.0, 4207.0)
  smoothness_se= st.slider('smoothness_se', 2700.0, 6300.0, 4207.0)
  smoothness_mean= st.slider('smoothness_mean', 2700.0, 6300.0, 4207.0)
  
  
  # Create a DataFrame for the input features
  data = {'texture_worst': texture_worst,
          'texture_se': texture_se,
          'texture_mean': texture_mean,
          'symmetry_worst': symmetry_worst,
          'symmetry_se': symmetry_se,
          'symmetry_mean': symmetry_mean,
          'smoothness_worst': smoothness_worst,
          'radius_worst': radius_worst,
          'smoothness_se': smoothness_se,
          'smoothness_mean': smoothness_mean}
  input_df = pd.DataFrame(data, index=[0])
  input_predictors = pd.concat([input_df, X], axis=0)
      

						


