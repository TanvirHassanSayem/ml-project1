import streamlit as st
import pandas as pd

# Title of the app
st.title("CSV File Uploader")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the DataFrame
    st.write("DataFrame:")
    st.dataframe(df)

    # Optionally, display some statistics about the DataFrame
    st.write("Statistics:")
    st.write(df.describe())
    
    