import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ----------------- Page Config -----------------
st.set_page_config(page_title="🧠 Mental Health Chatbot", layout="wide")

# ----------------- Load Model -----------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
except Exception as e:
    st.error("❌ Error loading model or feature columns.")
    st.stop()

# ----------------- Header -----------------
st.markdown("""
<div style="background: linear-gradient(90deg, #d1e7ff, #ffffff); padding: 15px; border-radius: 10px; display: flex; align-items: center;">
    <img src="https://cdn-icons-png.flaticon.com/512/4213/4213710.png" width="60" style="margin-right: 20px;">
    <div>
        <h1 style="margin: 0; color: #003366;">🧠 Mental Health Chatbot + Predictor</h1>
        <marquee behavior="scroll" direction="left" scrollamount="5" style="color: #0066cc;">
            Your wellbeing matters — get a personalized mental health risk score and support tips.
        </marquee>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
st.sidebar.image("logo.jpeg", width=280)
st.sidebar.markdown("""
<div style="background-color:#cce5ff; padding:10px; border-radius:10px; text-align:center;">
  <marquee behavior="scroll" direction="left" scrollamount="5" style="color:#003366;">
    🧠 Welcome to the Mental Health Chatbot - You're not alone.
  </marquee>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("### 💬 Contact Us")
st.sidebar.markdown("📧 **Email:** ks6047601@gmail.com")
st.sidebar.markdown("📞 **Helpline:** 123456789")
st.sidebar.markdown("---")
with st.sidebar.expander("💡 What This App Does"):
    st.write("""
    - Predict your mental health risk  
    - Show personalized tips  
    - Visual feedback with gauge and bar chart  
    """)

reaction = st.sidebar.radio("🧠 How are you feeling today?", ["😊 Great", "😐 Okay", "😟 Stressed", "😴 Tired", "😕 Anxious"])
mode = st.sidebar.selectbox("🎨 Choose Mode", ["Light Mode", "Dark Mode"])
if mode == "Dark Mode":
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #1e1e1e;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
st.sidebar.markdown("📝 **Tip of the Day**")
st.sidebar.info("“Mental health is just as important as physical health.”")

# ----------------- Form -----------------
step = st.progress(0)
with st.form("mental_health_form"):
    st.subheader("👤 Basic Information")
    name = st.text_input("Enter Your Full Name")
    Age = st.slider("Your Age", 18, 65, 30)
    Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    Country = st.selectbox("Country", ["United States", "India", "United Kingdom", "Germany", "Other"])
    self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])
    family_history = st.selectbox("Any family history of mental illness?", ["Yes", "No"])
    work_interfere = st.selectbox("How often does mental health interfere with work?", ["Never", "Rarely", "Sometimes", "Often"])
    step.progress(25)

    st.subheader("🏢 Workplace Environment")
    benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Do you know options for mental health care?", ["Yes", "No", "Not sure"])
    anonymity = st.selectbox("Is anonymity protected at your workplace?", ["Yes", "No", "Don't know"])
    leave = st.selectbox("Is medical leave easy to take for mental health?", ["Somewhat easy", "Very difficult", "Don't know", "Somewhat difficult"])
    step.progress(50)

    st.subheader("👥 Social Support & Resources")
    coworkers = st.selectbox("Can you talk to coworkers about mental health?", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Can you talk to your supervisor?", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Would you discuss MH in an interview?", ["Yes", "No", "Maybe"])
    obs_consequence = st.selectbox("Seen consequences of discussing MH?", ["Yes", "No"])
    submit = st.form_submit_button("🔍 Predict Risk")
    step.progress(75)

# ----------------- Prediction -----------------
if submit:
    if not name.strip():
        st.warning("Please enter your name to continue.")
    else:
        try:
            user_input = {
                "Age": Age,
                "Gender": Gender,
                "Country": Country,
                "self_employed": self_employed,
                "family_history": family_history,
                "work_interfere": work_interfere,
                "benefits": benefits,
                "care_options": care_options,
                "anonymity": anonymity,
                "leave": leave,
                "coworkers": coworkers,
                "supervisor": supervisor,
                "mental_health_interview": mental_health_interview,
                "obs_consequence": obs_consequence
            }

            df_input = pd.DataFrame([user_input])
            df_encoded = pd.get_dummies(df_input)
            df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

            prob = model.predict_proba(df_aligned)[0][1]
            score = round(prob * 100, 2)

            # Risk level
            if prob < 0.3:
                emoji = "😊"
                msg = "Low Risk: Great job maintaining your mental health!"
                bg_color = "#D0F0FD"
                border = "#0C7B93"
            elif 0.3 <= prob < 0.7:
                emoji = "😐"
                msg = "Moderate Risk: Consider reaching out or making small changes."
                bg_color = "#FFFACD"
                border = "#DAA520"
            else:
                emoji = "😟"
                msg = "High Risk: It’s important to seek support and talk to someone."
                bg_color = "#FFD1DC"
                border = "#B22222"

            st.markdown(f"""
            <div style='background-color:{bg_color}; padding: 25px; border-radius: 15px; border-left: 6px solid {border}; box-shadow: 0 2px 12px rgba(0,0,0,0.15);'>
                <h2 style='color: {border};'>Hello {name} 👋</h2>
                <h3 style='color: {border};'>Mental Health Risk: <b>{score:.1f}%</b> {emoji}</h3>
                <p style='font-size: 18px; color: {border};'>{msg}</p>
            </div>
            """, unsafe_allow_html=True)

            step.progress(100)

            # ----------------- Gauge Chart -----------------
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Mental Health Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 30], 'color': '#D0F0FD'},
                        {'range': [30, 70], 'color': '#FFE5B4'},
                        {'range': [70, 100], 'color': '#FFD1DC'}
                    ]
                }
            )), use_container_width=True)

        

            # ----------------- Tips -----------------
            with st.expander("💡 Mental Health Tips"):
                if score > 75:
                    st.markdown("- Book an appointment with a therapist")
                    st.markdown("- Try guided meditation and journaling")
                    st.markdown("- Avoid social isolation")
                elif score > 50:
                    st.markdown("- Take regular breaks")
                    st.markdown("- Talk to friends or support groups")
                    st.markdown("- Practice mindfulness or yoga")
                else:
                    st.markdown("- Maintain a work-life balance")
                    st.markdown("- Keep physical activity and hobbies")

            # ----------------- Certificate Generator -----------------
            def generate_certificate_html(name, score):
                return f"""
                <html>
                <head>
                    <style>
                        .certificate {{
                            border: 10px solid #0a4275;
                            padding: 30px;
                            text-align: center;
                            font-family: Arial, sans-serif;
                            background-color: #f0f8ff;
                            border-radius: 15px;
                            width: 80%;
                            margin: auto;
                        }}
                        .title {{
                            font-size: 32px;
                            font-weight: bold;
                            color: #003366;
                        }}
                        .score {{
                            font-size: 24px;
                            color: #006600;
                        }}
                        .footer {{
                            font-size: 16px;
                            margin-top: 30px;
                            color: #333;
                        }}
                    </style>
                </head>
                <body>
                    <div class="certificate">
                        <div class="title">🌿 Mental Health Wellness Certificate</div>
                        <p>This certifies that <strong>{name}</strong> has completed a self-assessment for mental health risk.</p>
                        <p class="score">Mental Health Risk Score: <strong>{score}%</strong></p>
                        <p class="footer">Generated by Mental Health Chatbot • Stay Safe • Stay Well 💚</p>
                    </div>
                </body>
                </html>
                """

            def get_download_link(html_content):
                b64 = base64.b64encode(html_content.encode()).decode()
                return f'<a href="data:text/html;base64,{b64}" download="mental_health_certificate.html">📥 <b>Download Your Certificate</b></a>'

            certificate_html = generate_certificate_html(name.strip(), score)
            st.markdown("---")
            st.markdown(get_download_link(certificate_html), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
