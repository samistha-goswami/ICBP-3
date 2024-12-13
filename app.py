import streamlit as st
import pandas as pd
import joblib

# Load model and dataset
def load_model():
    return joblib.load("churn_model.pkl")

def load_data():
    return pd.read_csv("datachurn_data.csv")

# Preprocess input data
def preprocess_input(input_df, train_columns):
    # Apply one-hot encoding
    input_df = pd.get_dummies(input_df)
    # Align columns to match training data
    input_df = input_df.reindex(columns=train_columns, fill_value=0)
    return input_df

# Load resources
model = load_model()
data = load_data()
train_columns = model.feature_names_in_  # Features used during training

# Streamlit app setup
st.title("Customer Churn Prediction")
st.sidebar.header("Input Features")
st.write("### About the Project")
st.write("This application predicts customer churn based on user-provided inputs.")

# User inputs
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ("Yes", "No"))
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0, 200, 50)
    total_charges = st.sidebar.slider("Total Charges ($)", 0, 10000, 500)

    contract = st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    dependents = st.sidebar.selectbox("Dependents", ("Yes", "No"))

    # Create dictionary of features
    features = {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "Contract": contract,
        "Dependents": dependents,
    }
    return pd.DataFrame([features])

# Capture user input
input_df = user_input_features()
st.write("### User Input Features", input_df)

# Preprocess input for prediction
processed_input = preprocess_input(input_df, train_columns)

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(processed_input)[0]
    prediction_prob = model.predict_proba(processed_input)[0][1]

    if prediction == 1:
        st.error(f"### Customer is likely to churn (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"### Customer is not likely to churn (Probability: {prediction_prob:.2f})")

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(data.head())
