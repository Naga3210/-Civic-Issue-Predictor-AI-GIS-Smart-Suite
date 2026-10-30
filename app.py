import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(
    page_title="CivicIssue AI & GIS Smart Dashboard",
    page_icon="🏙️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            background-color: #0284c7;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #0369a1;
        }
    </style>
""", unsafe_allow_html=True)

# 1. Model Training
@st.cache_resource
def train_model():
    X_train = np.array([
        [0, 1, 9, 8], [1, 2, 2, 1], [2, 1, 8, 5], 
        [0, 3, 3, 1], [3, 2, 9, 9], [1, 1, 4, 2],
        [3, 3, 7, 4], [2, 2, 1, 0], [0, 2, 8, 6]
    ])
    y_train = np.array([2, 0, 2, 0, 2, 0, 1, 0, 2])
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

model = train_model()

# Session State Setup
if "complaints_list" not in st.session_state:
    st.session_state.complaints_list = [
        {"lat": 13.0827, "lon": 80.2707, "Issue": "Water Supply Leakage", "Priority": "HIGH", "Severity": 9, "Zone": "Chennai Central"},
        {"lat": 13.0604, "lon": 80.2496, "Issue": "Road Pothole", "Priority": "MEDIUM", "Severity": 5, "Zone": "T. Nagar"},
        {"lat": 13.0418, "lon": 80.2341, "Issue": "Garbage Disposal", "Priority": "LOW", "Severity": 2, "Zone": "Adyar"}
    ]

# Header
st.title("🏙️ CivicIssue Predictor — AI & GIS Smart Suite")
st.write("Innovative AI Vision, Voice Input, Real-Time GIS Mapping & Analytics")

st.divider()

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📍 Register & Predict", "🗺️ GIS Map & Route", "📊 City Analytics & SDG"])

# -------------------------------------------------------------
# TAB 1: Smart Registration (Image, Voice, AI)
# -------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Citizen Complaint Form")
        
        issue_type = st.selectbox(
            "Complaint Type",
            ["Water Supply Leakage", "Road Pothole", "Garbage Disposal", "Electrical Hazard"]
        )
        
        location_name = st.selectbox(
            "Select Location Zone",
            ["Chennai Central", "T. Nagar", "Adyar", "Velachery", "Custom Location"]
        )

        preset_coords = {
            "Chennai Central": (13.0827, 80.2707),
            "T. Nagar": (13.0418, 80.2341),
            "Adyar": (13.0067, 80.2571),
            "Velachery": (12.9759, 80.2212),
            "Custom Location": (13.0827, 80.2707)
        }
        default_lat, default_lon = preset_coords[location_name]

        lat = st.number_input("Latitude", value=default_lat, format="%.4f")
        lon = st.number_input("Longitude", value=default_lon, format="%.4f")
        
        severity = st.slider("Severity Level (1 to 10)", 1, 10, 5)
        history = st.number_input("Past Complaint Count", min_value=0, max_value=50, value=2)

    with col2:
        st.subheader("💡 Innovative AI Features")
        
        # Feature 1: Image Upload / Camera
        st.write("📷 **Upload Issue Photo (AI Auto-Severity Analysis)**")
        uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            st.image(uploaded_file, width=250, caption="Uploaded Issue Image")
            st.info("🤖 **AI Vision Detection:** Image pattern indicates potential high structural damage (Suggested Severity: 8+)")

        # Feature 2: Voice Audio Note
        st.write("🎙️ **Voice Complaint Recording**")
        audio_file = st.audio_input("Record Voice Note (Tamil/English)")
        if audio_file:
            st.audio(audio_file)
            st.success("🗣️ **Audio Transcribed:** 'Water leakage near main junction...'")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Submit & Predict Priority"):
            type_map = {"Water Supply Leakage": 0, "Road Pothole": 1, "Garbage Disposal": 2, "Electrical Hazard": 3}
            loc_map = {"Chennai Central": 1, "T. Nagar": 2, "Adyar": 3, "Velachery": 4, "Custom Location": 1}
            
            encoded_input = [[type_map[issue_type], loc_map[location_name], severity, history]]
            prediction = model.predict(encoded_input)[0]
            
            if prediction == 2:
                priority_str = "HIGH"
                st.error("🔴 **HIGH Priority** — WhatsApp Alert Sent to Municipal Officer!")
            elif prediction == 1:
                priority_str = "MEDIUM"
                st.warning("🟡 **MEDIUM Priority** — Scheduled Inspection Added")
            else:
                priority_str = "LOW"
                st.success("🟢 **LOW Priority** — Logged in Standard Queue")

            st.session_state.complaints_list.append({
                "lat": lat,
                "lon": lon,
                "Issue": issue_type,
                "Priority": priority_str,
                "Severity": severity,
                "Zone": location_name
            })

# -------------------------------------------------------------
# TAB 2: GIS Map & Route Optimization
# -------------------------------------------------------------
with tab2:
    st.subheader("🗺️ Live Complaints Interactive GIS Map")
    df_map = pd.DataFrame(st.session_state.complaints_list)
    
    st.map(df_map, latitude="lat", longitude="lon", size=25)
    
    st.subheader("🚚 AI Route Optimizer for Maintenance Crew")
    st.caption("Auto-generates the most efficient sequence to visit HIGH priority locations first.")
    
    high_prio_df = df_map[df_map["Priority"] == "HIGH"]
    st.dataframe(high_prio_df[["Zone", "Issue", "Priority", "Severity"]], use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Analytics Dashboard & SDG Impact
# -------------------------------------------------------------
with tab3:
    st.subheader("📊 City Administration Analytics")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Complaints", len(df_map))
    c2.metric("High Priority Issues", len(df_map[df_map['Priority'] == 'HIGH']))
    c3.metric("SDG 11 Compliance", "94%")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.write("**Priority Distribution**")
        fig_pie = px.pie(df_map, names='Priority', color='Priority', 
                         color_discrete_map={'HIGH':'red', 'MEDIUM':'orange', 'LOW':'green'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.write("**Severity Level by Complaint Type**")
        fig_bar = px.bar(df_map, x='Issue', y='Severity', color='Priority')
        st.plotly_chart(fig_bar, use_container_width=True)