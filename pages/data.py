import streamlit as st
import pandas as pd 
import os

st.title("Data page")

# Load data
@st.cache_data(persist=True)
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of the script
    file_path = os.path.join(base_dir, "..", "Data", "customer_data.csv")  
    data = pd.read_csv(file_path)
    return data

st.dataframe(load_data())

