import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load student data
data = pd.read_csv("student_data.csv")

# Select input features
X = data[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "self_study_hours"
    ]
]

# Select the marks we want to predict
y = data["final_marks"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the machine learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict marks for the test data
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)

print("STUDENT PERFORMANCE PREDICTION")
print("--------------------------------")

print("Actual Marks:", list(y_test))
print("Predicted Marks:", predictions.round(2))

print("Mean Absolute Error:", round(mae, 2))

# Save the trained model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel trained successfully!")
print("Model saved as student_model.pkl")