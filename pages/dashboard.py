import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
from yaml.loader import  SafeLoader
import streamlit_authenticator as stauth

# Set up Home page
st.set_page_config(
    page_title="Dashboard",
    layout="wide"
)



with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)



# load the dataset
data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Github projects\customer-churn-app\Data\customer_data.csv")

import streamlit as st
import plotly.express as px
import pandas as pd



def eda_dashboard():
    col1, col2 = st.columns(2)
    with col1:
        st.header('Exploratory Data Analysis Dashboard')
    
    col1, col2 = st.columns(2)
    with col1:
        hist = px.histogram(data, 'totalcharges', title='Total charges distribution', color='churn')
        st.plotly_chart(hist, use_container_width=True)

        hist = px.histogram(data, 'tenure', title='Tenure distribution', color='churn')
        st.plotly_chart(hist, use_container_width=True)

        corr = data[['tenure', 'monthlycharges', 'totalcharges']].corr()
        corr = px.imshow(corr, text_auto=True, title='Numerical columns correlations')
        st.plotly_chart(corr, use_container_width=True)

    with col2:
        hist = px.histogram(data, 'monthlycharges', title='Monthly charges distribution', color='churn')
        st.plotly_chart(hist, use_container_width=True)

        boxplot = px.box(data, y=['tenure', 'monthlycharges'], title='Tenure and Monthly Charges Boxplots', color='churn')
        st.plotly_chart(boxplot, use_container_width=True)

        boxplot = px.box(data, y=['totalcharges'], title='Total Charges Boxplot', color='churn')
        st.plotly_chart(boxplot, use_container_width=True)

def kpi_dashboard():
    col2, col3 = st.columns(2)
    with col2:
        st.header('Key Performance Indicator Dashboard')

    col1, col2 = st.columns(2)
    churn_rate = data.groupby(['churn', 'tenure'])['churn'].count()
    churn_rate = churn_rate.reset_index(name='Number of Customers')
    churn_rate = churn_rate.pivot(index='tenure', columns='churn', values='Number of Customers')
    churn_rate['Churn Rate'] = churn_rate['Yes'] / (churn_rate['No'] + churn_rate['Yes'])
    churn_rate = churn_rate.reset_index()
    churn_rate = px.line(churn_rate, x='tenure', y='Churn Rate', title='Churn Rate vs Tenure')
    st.plotly_chart(churn_rate, use_container_width=True)

def select_dashboard():
    if 'selected_dashboard' not in st.session_state:
        st.session_state['selected_dashboard'] = 'KPI Dashboard'
    
    dashboard = st.selectbox('Select Dashboard', ['KPI Dashboard', 'EDA Dashboard'], key='selected_dashboard')
    if st.session_state['selected_dashboard'] != dashboard:
        st.session_state['selected_dashboard'] = dashboard
        st.experimental_rerun()

    if st.session_state['selected_dashboard'] == 'KPI Dashboard':
        kpi_dashboard()

    if st.session_state['selected_dashboard'] == 'EDA Dashboard':
        eda_dashboard()

if __name__ == '__main__':
    select_dashboard()
