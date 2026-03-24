import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset for accuracy
data = pd.read_csv("student_data.csv")
X = data[['study_hours', 'sleep_hours', 'mobile_usage_hours']]
y = data['marks']

# Predict for accuracy
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# Page config
st.set_page_config(
    page_title="Student Predictor",
    page_icon="🎓",
    layout="centered"
)

# Title
st.markdown("<h1 style='text-align: center;'>🎓 Student Result Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter study behavior and get predicted marks</p>", unsafe_allow_html=True)

# Accuracy display
st.markdown("### 📊 Model Performance")
colA, colB = st.columns(2)

with colA:
    st.metric("R² Score", round(r2, 3))

with colB:
    st.metric("Mean Error", round(mae, 2))

st.markdown("---")

# Inputs in columns
col1, col2, col3 = st.columns(3)

with col1:
    study = st.text_input("📚 Study Hours", placeholder="e.g. 6")

with col2:
    sleep = st.text_input("😴 Sleep Hours", placeholder="e.g. 7")

with col3:
    mobile = st.text_input("📱 Mobile Usage", placeholder="e.g. 3")

st.markdown("")

# Predict button
if st.button("🔮 Predict Result", use_container_width=True):

    try:
        study = float(study)
        sleep = float(sleep)
        mobile = float(mobile)

        # Prediction
        marks = model.predict([[study, sleep, mobile]])
        marks = float(marks[0])
        marks = round(marks, 2)

        marks = max(0, min(100, marks))

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        st.metric("Predicted Marks", f"{marks}")
        st.progress(int(marks))

        if marks >= 40:
            st.success("✅ PASS 🎉")
            st.balloons()
        else:
            st.error("❌ FAIL 😢")

        st.markdown("### 💡 Smart Suggestion")

        if study < 5:
            st.warning("Increase study hours for better score 📚")
        elif mobile > 6:
            st.warning("Reduce mobile usage 📵")
        elif sleep < 6:
            st.warning("Improve sleep cycle 😴")
        else:
            st.success("Great routine! Keep going 🔥")

    except:
        st.error("⚠️ Please enter valid numeric values")

st.markdown("---")
st.caption("Made with ❤️ using Machine Learning")
st.caption("Created by Shubham Thakor - [GitHub](https://github.com/shubham0182) | [Instagram](https://www.instagram.com/_shubhhhh_012/)")
