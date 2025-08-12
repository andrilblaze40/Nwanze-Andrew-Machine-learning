import streamlit as st
import pandas as pd

st.title('🎈 CANCER DIAGNOSIS APP')

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
  y
with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='symmetry_mean', y='texture_mean', color='diagnosis')
 
# Input features
with st.sidebar:
  st.header('Input features')
  
  texture_worst = st.slider('texture worse', -2.7, 2.7	, -0.0)
  texture_se = st.slider('texture_se', -2.6, 2.6, -0.0)
  texture_mean = st.slider('texture_mean', -6.8	, 6.0, -0.5)
  symmetry_worst= st.slider('symmetry_worst', -2.7, -2.7, 0.0) 
  symmetry_se= st.slider('symmetry_se', -6.0, 6.3, -0.0)
  symmetry_mean= st.slider('symmetry_mean', -7.2, 6.4, -0.4)
  smoothness_worst= st.slider('smoothness_worst', -11.9	, 11.7, -0.1)
  radius_worst= st.slider('radius_worst', -2.7, 2.7, -0.7)
  smoothness_se= st.slider('smoothness_se', -4.6, 6.6, 1.0)
  smoothness_mean= st.slider('smoothness_mean', -13.2, 10.3, -1.5)
  
  
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


with st.expander('Input features'):
  st.write('**Input predictors**')
  input_df
  st.write('**Combined predictors data**')
  input_predictors


# Data preparation
# Encode y		
  target_mapper = {'B': 0,
                 'M': 1,
                 }
 def target_encode(val):
  return target_mapper[val]

  y = y.apply(target_encode)
  y

