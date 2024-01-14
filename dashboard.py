import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit.proto.Selectbox_pb2 import Selectbox
import requests
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import logging

import os

#heroku_api_url = 'http://localhost:5000'

heroku_api_url = "https://still-bayou-61593-4aed81ce9738.herokuapp.com"
logging.basicConfig(level=logging.INFO)

# Initialize session state
if 'data_displayed' not in st.session_state:
    st.session_state.data_displayed = False


DATA_PATH = 'client_data.csv'
RAW_DATA_PATH = 'Xtest_raw.csv'
MERGED = 'merged.csv'
THRESHOLD_PATH = 'threshold.txt'
LOGO_PATH = "logo.png"

@st.cache_data
def load_data():
    df_test = pd.read_csv(DATA_PATH)
    df_test_raw = pd.read_csv(RAW_DATA_PATH)
    merged_data = pd.read_csv(MERGED)
    with open(THRESHOLD_PATH, 'r') as file:
        custom_threshold = float(file.read())
    return df_test, df_test_raw, merged_data, custom_threshold

@st.cache_data
def generate_shap_plot(values, base_value, data):
    plt.clf()
    shap.waterfall_plot(shap.Explanation(values, base_value, data))
    plt.savefig('shap_plot.png', bbox_inches='tight')
    st.image('shap_plot.png')

st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache_data
def compare_variable(selected_client, df_test_raw, merged_data, selected_variable, client_code_1):
    st.title("Client Comparison to All the Clients")

    # Using the Greys palette
    palette = "Greys"
    sns.set()
    fig, ax = plt.subplots(figsize=(10, 6))

    if pd.api.types.is_numeric_dtype(df_test_raw[selected_variable]):

        colors = ['#454545', '#999999'] 
        
        for i, category in enumerate(merged_data['TARGET'].unique()):
            subset = merged_data[merged_data['TARGET'] == category]
            ax.boxplot(subset[selected_variable],
                       positions=[i + 1],  # Position on the x-axis
                       widths=0.6,
                       showfliers=True,
                       patch_artist=True,  # Enable coloring
                       boxprops=dict(facecolor=colors[i]))  # Set box color

        client_value = selected_client[selected_variable].values[0]
        ax.scatter(x=[i + 1], y=[client_value], color='magenta', s=100, label=f"Client ID: {client_code_1}")

        
        ax.set_xlabel('Risk Category')
        ax.set_ylabel('Selected Variable Values')
        ax.set_title('Matplotlib Boxplot')
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['No Defaulting Risk', 'Defaulting Risk'])
        
        # Show the legend
        ax.legend()        
        plt.show()       
        
    else:
        # For categorical variables, create a count plot
        ax = sns.countplot(x=selected_variable, data=merged_data, hue='TARGET', palette=palette)
        
        # Setting the legend labels manually
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=['No Defaulting Risk', 'Defaulting Risk'], title='Loan Status', loc='upper right')
        
        plt.xticks(rotation=45, ha='right')
        sns.scatterplot(x=merged_data[selected_variable].value_counts().index.tolist().index(selected_client[selected_variable].values[0]),
                        y=[0], color='magenta', s=100, label=f"Client ID: {client_code_1}")

    plt.xlabel(selected_variable)
    plt.title(f'Comparison of {selected_variable} for All Clients')
    
    st.pyplot()
    


def welcome_page():
    st.markdown("""
    <div style='text-align: center;'>
        <h1>Welcome to the Loan Predictor Dashboard</h1>
    </div>
""", unsafe_allow_html=True)    
    st.markdown("""
    <div style='text-align: center;'>
        <h3>We believe in fostering trust through transparency.</h3>
        <p>This dashboard has been designed to provide clear insights into the factors influencing loan approval decisions. By ensuring you understand the reasons behind loan approval or denial, you can explore and comprehend the factors that influence whether a client is likely to repay a loan successfully or face repayment challenges.</p>
        <p>To begin exploring, please click the button below.</p>
    </div>
""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start the Process"):
            st.session_state.page = "main"
            st.experimental_rerun()
    with col3: 
        st.image("arrow.png", width=50) 

def main():
    df_test, df_test_raw, merged_data, custom_threshold= load_data()
    
    df_test_exp = df_test.drop("SK_ID_CURR", axis=1)

    
    st.sidebar.header('User Options')
    logo = Image.open(LOGO_PATH)
    st.sidebar.image(logo, width=200)
    
    
    client_code_1 = int(st.sidebar.selectbox("Select a client code:", df_test_raw['SK_ID_CURR'].unique()))

    
    st.title("Loan Repayment Predictor")

    selected_client = df_test_raw[df_test_raw['SK_ID_CURR'] == client_code_1]
    st.subheader(f"Descriptive Information for Client {client_code_1}")
    selected_client_data = df_test[df_test['SK_ID_CURR'] == client_code_1]
    st.write(selected_client)

    if st.button("Predict"):
        try:
            # Send request to Flask API for predictions and SHAP plot
            response = requests.post(f"{heroku_api_url}/predict", json={"client_code_1": client_code_1})
            response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
            data = response.json()
            
            predictions = data['predictions']
            dat = data['dat']
            st.session_state.data_displayed = True

        
            st.subheader("Predictions")            

            if predictions['prediction_1'] == "Granted":
                prediction_value_style = "color: green; font-size: 200%;"
                repayment_phrase = "The client is predicted to have a smooth repayment process."
            elif predictions['prediction_1'] == "Not Granted":
                prediction_value_style = "color: red; font-size: 200%;"
                repayment_phrase = "The client WILL HAVE problems in repaying his debt."

            st.markdown(f"Client {client_code_1} Prediction: <span style='{prediction_value_style}'>{predictions['prediction_1']}</span>", unsafe_allow_html=True)
            
            # Display the specific phrase based on the prediction
            st.write(repayment_phrase)
            


# Display other details
            st.write(f"Probability of Repayment Issues: {predictions['probability_1']:.2f}%")
            st.write("********************************************************") 

            shap_values_matrix = predictions['values']
            values = np.array(shap_values_matrix)
            base_value = predictions['base_values']
            result_series = pd.Series(dat)

        except requests.RequestException as e:
            st.error(f"Error making prediction request: {e}")
   
    if st.session_state.data_displayed:
        option = st.checkbox('Select this case for more details')
        if option:
            response = requests.post(f"{heroku_api_url}/predict", json={"client_code_1": client_code_1})
            response.raise_for_status() 
            data = response.json()
            
            predictions = data['predictions']
            dat = data['dat']
             
            shap_values_matrix = predictions['values']
            values = np.array(shap_values_matrix)
            base_value = predictions['base_values']
            result_series = pd.Series(dat)

            st.title(f"Variables influence on the Prediction for client {client_code_1}")
            idx1 = df_test_raw[df_test_raw['SK_ID_CURR'] == int(client_code_1)].index[0]
            generate_shap_plot(values,base_value,result_series)

            
            st.write(f"This waterfall plot represent the cumulative effect of sequentially introduced positive or negative variables. Each bar/variable is color-coded to indicate whether it contributes positively (blue) or negatively (red) to the final understanding of a client having or not repayment issues. The length of each bar represents the magnitude of the change.")
           
            sorted_columns = sorted(df_test_raw.drop("SK_ID_CURR", axis=1).columns)
            selected_variable = st.selectbox("Select a variable:", sorted_columns)
            compare_variable(selected_client,df_test_raw, merged_data, selected_variable, client_code_1)
            st.write(f"For each variable taken into acount for the decision making process a graph with its distribtion and the clients emplacement.")

    if st.button("Reset"):
        st.session_state.clear()
        st.experimental_rerun()
if __name__ == '__main__':
    if 'page' not in st.session_state:
        st.session_state.page = "welcome"
        
    if st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "main":
        main()

