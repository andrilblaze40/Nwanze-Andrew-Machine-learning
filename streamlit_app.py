import streamlit as st
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler



st.title('🎈ML CANCER DIAGNOSIS APP')

st.info('MACHINE LEARNING PIPELINE')
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
  st.write('**Scatter plot of symetry_mean vs texture_mean**')
  st.scatter_chart(data=df, x='symmetry_mean', y='texture_mean', color='diagnosis')
  st.write('**Class Balance**')
  st.bar_chart(df["diagnosis"].value_counts(normalize=True))
 
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
  #target_mapper = {'B': 0,
                 #'M': 1,
                 #}
  #def target_encode(val):
  #return target_mapper[val]

  #y = y.apply(target_encode)
  

#with st.expander('Data Preparation'):
  #st.write('**Encoded y**')
  #y

# Model training
# Divide your dataset into training and test sets using a randomized split. Your test set should be 20% of your data. Be sure to set `random_state` to `42`.
# Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42)

  X_train.shape
  y_train.shape
  X_test.shape
  y_test.shape
with st.expander('Split Data'):
  st.write('**X_train**')
  X_train.shape
  st.write('**X_test**')
  X_test.shape
  st.write('**y_train**')
  y_train.shape
  st.write('**y_test**')
  y_test.shape

 
 
#  Create a new feature matrix `X_train_over` and target vector `y_train_over` by performing random over-sampling on the training data.
  over_sampler = RandomOverSampler(random_state=42)
  X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
  print("X_train_over shape:", X_train_over.shape)
 
with st.expander(' RandomOverSampler'):
  st.write('**X_train_over**')
  X_train_over
  st.write('**y_train_over**')
  y_train_over

# Create a classifier  that can be trained on `(X_train_over, y_train_over)`.
  clf = make_pipeline(
  SimpleImputer(),
  RandomForestClassifier(random_state=42))
  clf.fit(X_train_over, y_train_over)
with st.expander('RandomForestClassifier'):
  st.write('**Clf**')
  clf

# Perform cross-validation with your classifier using the over-sampled training data
  cv_scores = cross_val_score(clf, X_train_over, y_train_over, cv=5, n_jobs=-1)
with st.expander('Cross Validation'):
  st.write('**cv_scores**')
  cv_scores

# Hyperparameters that you want to evaluate for your classifier. 
  params = {
    "randomforestclassifier__n_estimators": range(25, 100, 25),
    "randomforestclassifier__max_depth": range(10,50,10)
}
  params
with st.expander('Hyperparameter tuning'):
  st.write('**params**')
  params

# Create a <code>GridSearchCV</code> named `model` that includes your classifier and hyperparameter grid. Be sure to set `cv` to 5, `n_jobs` to -1, and `verbose` to 1. 


  model = GridSearchCV(
  clf,
  param_grid=params,
  cv=5,
  n_jobs=-1,
  verbose=1)
  model
with st.expander('GridSearchCV'):
  st.write('**model**')
  model

# Fit your model to the over-sampled training data. 
  model.fit(X_train_over, y_train_over)
with st.expander('Fit Model'):
  st.write('**Fitted Model**')
  model.fit

  # Extract the cross-validation results from your model, and load them into a DataFrame
  cv_results = pd.DataFrame(model.cv_results_)
with st.expander('Cross Validation Results'):
  st.write('**cv_results**')
  cv_results

  # Extract the best hyperparameters from your model and assign them to <code>best_params</code>. 
  best_params = model.best_params_
  best_params
with st.expander('Best Hyperparameters'):
  st.write('**best_params**')
  best_params

# Test the quality of your model by calculating accuracy scores for the training and test data.
  acc_train = model.score(X_train_over, y_train_over)
  acc_test = model.score(X_test, y_test)
with st.expander('Accuracy Scores'):
  st.write('**Training Accuracy**')
  acc_train
  st.write('**Test Accuracy**')
  acc_test

# Compute confusion matrix
  cm = confusion_matrix(X_test, y_test)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  p = st.pyplot(disp)
with st.expander('Confusion Matrix Display'):
  st.write('**Confusion Matrix**')
  p
  
