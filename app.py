import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)

# Website title
st.title("Student Performance Prediction")

st.write(
    "Enter the student's information to estimate final marks."
)

# Load and display the complete student dataset
data = pd.read_csv("student_data.csv")

st.subheader("Student Dataset")

st.write("Total student records:", len(data))

st.dataframe(data, use_container_width=True)

# Take input from the user
study_hours = st.slider(
    "Study Hours Per Day", 0, 12, 4
)

attendance = st.slider(
    "Attendance Percentage", 0, 100, 75
)

previous_marks = st.slider(
    "Previous Marks", 0, 100, 60
)

self_study_hours = st.slider(
    "Self-Study Hours Per Day", 0, 12, 3
)

# Predict marks
if st.button("Predict Performance"):

    student = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_marks,
            self_study_hours
        ]],
        columns=[
            "study_hours",
            "attendance",
            "previous_marks",
            "self_study_hours"
        ]
    )

    result = model.predict(student)[0]

    # Keep displayed marks between 0 and 100
    result = max(0, min(100, result))

    st.subheader("Prediction Result")

    st.success(f"Estimated Final Marks: {result:.1f}/100")

    if result >= 80:
        st.write("Performance Category: Excellent")
    elif result >= 65:
        st.write("Performance Category: Good")
    elif result >= 45:
        st.write("Performance Category: Average")
    else:
        st.write("Performance Category: Needs Improvement")