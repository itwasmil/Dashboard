import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit.proto.Selectbox_pb2 import Selectbox
import requests
pip install matplotlib

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import logging

import os

API_URL = os.environ.get('HEROKU_URL', 'http://localhost:5000')



logging.basicConfig(level=logging.INFO)

DATA_PATH = 'client_data.csv'
RAW_DATA_PATH = 'Xtest_raw.csv'
THRESHOLD_PATH = 'threshold.txt'
LOGO_PATH = "logo.png"

@st.cache_data
def load_data():
    df_test = pd.read_csv(DATA_PATH)
    df_test_raw = pd.read_csv(RAW_DATA_PATH)
    with open(THRESHOLD_PATH, 'r') as file:
        custom_threshold = float(file.read())
    return df_test, df_test_raw, custom_threshold


@st.cache_data
def generate_shap_plot(values, base_value, data):
    plt.clf()
    shap.waterfall_plot(shap.Explanation(values, base_value, data))
    plt.savefig('shap_plot.png', bbox_inches='tight')
    st.image('shap_plot.png')

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def compare_variable(selected_client, df_test_raw, selected_variable, client_code_1):
    st.title("Client Comparison to all the Clients")

    all_variables = df_test_raw.columns.drop(['SK_ID_CURR'])
    sns.set_palette("gray")

    plt.figure(figsize=(10, 4))

    if pd.api.types.is_numeric_dtype(df_test_raw[selected_variable]):
        # For numeric variables, create a histogram
        sns.histplot(x=selected_variable, data=df_test_raw, kde=True)
        sns.scatterplot(x=selected_client[selected_variable].values[0], y=[0], color='magenta', s=100, label=f"Client: {client_code_1}")
    else:
        # For categorical variables, create a count plot
        sns.countplot(x=selected_variable, data=df_test_raw)
        plt.xticks(rotation=45, ha='right')
        sns.scatterplot(x=df_test_raw[selected_variable].value_counts().index.tolist().index(selected_client[selected_variable].values[0]),
                        y=[0], color='magenta', s=100, label=f"Client: {client_code_1}")

    plt.xlabel(selected_variable)
    plt.title(f'Comparison of {selected_variable} for All Clients')
    plt.legend()
    st.pyplot()

def main():
    df_test, df_test_raw, custom_threshold = load_data()
    
    df_test_exp = df_test.drop("SK_ID_CURR", axis=1)

    
    st.sidebar.header('User Options')
    logo = Image.open(LOGO_PATH)
    st.sidebar.image(logo, width=200)

    st.title("Loan Repayment Predictor")
    
    client_code_1 = int(st.sidebar.selectbox("Select a client code:", df_test_raw['SK_ID_CURR'].unique()))

    selected_client = df_test_raw[df_test_raw['SK_ID_CURR'] == client_code_1]
    st.subheader(f"Descriptive Information for Client {client_code_1}")
    selected_client_data = df_test[df_test['SK_ID_CURR'] == client_code_1]
    st.write(selected_client)

    if st.button("Predict and Compare"):
        try:
            # Send request to Flask API for predictions and SHAP plot
            response = requests.post(f"{API_URL}/predict", json={"client_code_1": client_code_1})
            response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
            data = response.json()
            
            predictions = data['predictions']
            dat = data['dat']


            st.subheader("Predictions")
            st.write(f"Client {client_code_1} Prediction: {predictions['prediction_1']}")
            st.write(f"Probability of Repayment Issues: {predictions['probability_1']} ")
            st.write("********************************************************")
            
            shap_values_matrix = predictions['values']
            values = np.array(shap_values_matrix)
            base_value = predictions['base_values']
            result_series = pd.Series(dat)


            # SHAP plot
            st.title(f"Variables influence on the Prediction for client {client_code_1}")
            idx1 = df_test_raw[df_test_raw['SK_ID_CURR'] == int(client_code_1)].index[0]
            generate_shap_plot(values,base_value,result_series)
        except requests.RequestException as e:
            st.error(f"Error making prediction request: {e}")

    
    selected_variable = st.selectbox("Select a variable:", df_test_raw.drop("SK_ID_CURR", axis=1).columns)
    
    if st.button("Compare based on Variables"):
        compare_variable(selected_client, df_test_raw, selected_variable, client_code_1)
       
    
if __name__ == '__main__':
    main()




