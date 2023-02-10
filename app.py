import streamlit as st
import pandas as pd
import os
# import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import *

with st.sidebar:
    st.image("https://i.pinimg.com/564x/be/69/18/be6918c5d9b9d3807c87f330656b8a41.jpg")
    st.title("AutoML WebApp")
    choice = st.radio("Navigation",["Upload","Profiling","ML Training","Download"])
    
if os.path.exists("sourcefile.csv"):
    df = pd.read_csv("sourcefile.csv",index_col=None)

if choice == "Upload":
    st.title("Upload the data for modelling")
    file = st.file_uploader("Upload dataset")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("sourcefile.csv",index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ML Training":
    st.title("Machnie Learning")
    target = st.selectbox("Select Your Target", df.columns)
    if st.button("Train model"):
      setup(df,target=target)
      setup_df = pull()
      st.dataframe(setup_df)
      best_model = compare_models()
      compare_df = pull()
      st.info("This is the ML model")
      st.dataframe(compare_df)
      best_model
      save_model(best_model, 'best_model')

if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
      st.download_button("Download the model", f, "trained_model.pkl")

