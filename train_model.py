import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv("datachurn_data.csv")

# Preprocess data
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy:.2f}")
 
import os

# Create the 'models' directory if it doesn't exist
os.makedirs("models", exist_ok=True) 

# Save model
joblib.dump(model, "models/churn_model.pkl")