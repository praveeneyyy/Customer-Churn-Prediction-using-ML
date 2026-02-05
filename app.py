import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .churn {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .no-churn {
        background-color: #ccffcc;
        border-left: 5px solid #00cc00;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    
    return model_data["model"], model_data["features_names"], encoders

# Load data
model, feature_names, encoders = load_model_and_encoders()

# App title and description
st.title("🔮 Customer Churn Prediction System")
st.markdown("---")
st.markdown("Predict whether a customer is likely to churn based on their characteristics and service usage.")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📝 Make Prediction", "📊 Model Info", "ℹ️ About"])

# ===== TAB 1: PREDICTION =====
with tab1:
    st.subheader("Enter Customer Information")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Demographic Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], format_func=lambda x: "Yes" if x == "Yes" else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    
    with col2:
        st.markdown("### Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    st.markdown("---")
    st.markdown("### Service Information")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("#### Phone & Internet")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with col4:
        st.markdown("#### Online Services")
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    with col5:
        st.markdown("#### Streaming Services")
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    st.markdown("---")
    st.markdown("### Charges")
    
    col6, col7 = st.columns(2)
    
    with col6:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=65.0, step=0.01)
    
    with col7:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=0.01)
    
    st.markdown("---")
    
    # Make prediction button
    if st.button("🎯 Predict Churn", type="primary", use_container_width=True):
        # Create input dataframe
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for column, encoder in encoders.items():
            if column in input_df.columns:
                input_df[column] = encoder.transform(input_df[column])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📋 Prediction Result")
        
        if prediction == 1:
            st.markdown("""
            <div class="prediction-box churn">
                <h2>⚠️ Customer Likely to Churn</h2>
                <p style="font-size: 20px;"><strong>Risk Level:</strong> {:.2f}%</p>
            </div>
            """.format(prediction_proba[1] * 100), unsafe_allow_html=True)
            
            st.warning("This customer has a high probability of churning. Consider retention strategies.")
        else:
            st.markdown("""
            <div class="prediction-box no-churn">
                <h2>✅ Customer Likely to Stay</h2>
                <p style="font-size: 20px;"><strong>Retention Level:</strong> {:.2f}%</p>
            </div>
            """.format(prediction_proba[0] * 100), unsafe_allow_html=True)
            
            st.success("This customer has a low probability of churning. Good customer loyalty indicator.")
        
        # Show probability details
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("No Churn Probability", f"{prediction_proba[0]*100:.2f}%")
        with col_b:
            st.metric("Churn Probability", f"{prediction_proba[1]*100:.2f}%")

# ===== TAB 2: MODEL INFORMATION =====
with tab2:
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Details")
        st.info(f"""
        - **Model Type:** {type(model).__name__}
        - **Number of Features:** {len(feature_names)}
        - **Training Approach:** Random Forest Classifier with GridSearchCV
        - **Cross-Validation:** 5-Fold
        """)
    
    with col2:
        st.markdown("### Performance Metrics")
        st.info("""
        - **Test Accuracy:** ~81%
        - **Handling:** SMOTE for class imbalance
        - **Hyperparameter Tuning:** GridSearchCV optimization
        """)
    
    st.markdown("---")
    st.markdown("### Feature List")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Categorical Features:**")
        categorical_features = [f for f in feature_names if f not in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']]
        for i, feature in enumerate(categorical_features, 1):
            st.text(f"{i}. {feature}")
    
    with col2:
        st.markdown("**Numerical Features:**")
        st.text("1. SeniorCitizen")
        st.text("2. tenure")
        st.text("3. MonthlyCharges")
        st.text("4. TotalCharges")
    
    with col3:
        st.markdown("**Target Variable:**")
        st.text("Churn (0: No, 1: Yes)")

# ===== TAB 3: ABOUT =====
with tab3:
    st.subheader("About This Application")
    
    st.markdown("""
    ### Overview
    This Customer Churn Prediction System uses Machine Learning to predict whether a customer 
    is likely to leave a telecommunications service provider.
    
    ### Dataset
    - **Source:** Telco Customer Churn Dataset
    - **Total Records:** ~7,043 customers
    - **Target Variable:** Churn (Yes/No)
    
    ### Model Development
    The model was developed using:
    1. **Data Preprocessing:** Label encoding for categorical variables
    2. **Handling Imbalance:** SMOTE (Synthetic Minority Oversampling Technique)
    3. **Model Selection:** Random Forest Classifier
    4. **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
    
    ### Key Features Influencing Churn
    - Contract type (Month-to-month customers churn more)
    - Tenure (Longer tenure = lower churn)
    - Internet service type
    - Tech support availability
    - Online security services
    
    ### Use Cases
    - **Customer Retention:** Identify at-risk customers for proactive retention
    - **Resource Allocation:** Focus retention efforts on high-risk segments
    - **Business Planning:** Understand churn patterns and trends
    
    ### Important Notes
    ⚠️ This model should be used as a tool to support decision-making, not as the sole basis for critical business decisions.
    
    ---
    **Built with:** Streamlit, Scikit-learn, XGBoost
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <small>© 2024 Customer Churn Prediction System | ML Model Dashboard</small>
</div>
""", unsafe_allow_html=True)
