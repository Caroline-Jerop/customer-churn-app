import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

st .set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon='🏠',
    layout="wide",
)

# Main content
st.title("Customer Churn Prediction App")
st.markdown(""" This app uses machine learning to predict customer churn in Telco company""")

# Key features
st.subheader("""Key features""")
st.markdown(""" 
- Upload your CSV file containing your customer  churn
- Select the desired features for classification
- Choose a machine learning  model from the dropdown menu
- Click 'Classify'to get the predicted results
- The report includes metrics like accuracy,precission,recall
""")

# menu
st.subheader("""App feature""")
st.markdown("""
- **View Data** : Access proprietary data
- **Dashboard** : Explore interactive dashboard for insight
- **Predict**: Instantly see predictions for customer attrition.
- **History**: See past predictions made.

""")

# How to run the application
st.subheader("How to run the application")
st.code("""
# activate virtual environment
env/scripts/activate
streamlit run home.py
""", language='bash')

# Machine learning

st.subheader(" Machine Learning Intergration")
st.markdown(""" 
- **Model Selection**: Choose between two advanced models for accurate prediction
- **Seamless Intergration**: Intergrate predictions into your workflow with user-friendly interfaces
- **Probability Estimates**: Gain insights into the likelihood of  predicted outcomes.
""")
 

  
# Contact and GitHub Repository
st.subheader("Need Help?")
st.markdown("For collaboration contact me at [Carojerop95@gmail.com])")
st.button("Repository on Github",help= "https://github.com/Caroline-Jerop/customer-churn-app))")

