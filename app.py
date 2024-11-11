import streamlit as st
import pandas as pd
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import os

# Title of the app
st.title("CSV File Uploader and Notebook Runner")

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
    
    st.title("Run Jupyter Notebook in Streamlit")

# Function to run the notebook and return HTML output
def run_notebook(notebook_path):
    try:
        # Check if the notebook file exists
        if not os.path.exists(notebook_path):
            st.error(f"Notebook not found at path: {notebook_path}")
            return None

        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})

        # Convert the notebook to HTML
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'lab'  # Optional: Using JupyterLab template for better style
        (body, resources) = html_exporter.from_notebook_node(nb)

        return body

    except Exception as e:
        st.error(f"Error running the notebook: {e}")
        return None

# Specify the path to your Jupyter notebook
notebook_path = r"C:\Users\Sayem\Desktop\Web_Dev\ml-project1\Credit Card Fraud Detection - Decision Tree.ipynb"

# Run the notebook and get the HTML output
if st.button("Run Notebook"):
    notebook_output = run_notebook(notebook_path)
    if notebook_output:
        st.components.v1.html(notebook_output, height=800, scrolling=True)
