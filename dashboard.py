import streamlit as st
from transformers import pipeline
from datetime import datetime
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbwxe6QwqyOPk-icikFapgqzXw-En--93zbfM74ZZ3TNnOV_0i-H8DzGgP-nB3AuHypLeQ/exec" 

# Set page to wide mode
st.set_page_config(page_title="Titan Crystal Suite", layout="wide", initial_sidebar_state="expanded")

# --- LIGHT CLEAN CSS THEME ---
st.markdown("""
<style>
    /* Main Background - Off-white/Light gray */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling - Light blue/gray */
    [data-testid="stSidebar"] {
        background-color: #E2E8F0 !important;
        border-right: 1px solid #CBD5E1;
    }
    
    /* Text Colors */
    h1, h2, h3 { color: #0F172A !important; font-weight: 600 !important; }
    p, label { color: #475569 !important; }

    /* Input Fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px;
    }
    
    /* The Submit Button - Dark Slate Blue */
    .stButton>button {
        background-color: #334155;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1E293B;
        color: #FFFFFF;
        transform: translateY(-1px);
    }

    /* Style the Info boxes at the bottom */
    div[data-testid="stAlert"] {
        background-color: #E0F2FE;
        color: #0369A1;
        border: none;
        border-radius: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- LOAD THE AI MODEL ---
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="./titan_sentiment_model")

classifier = load_model()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("<h2 style='color:#0F172A;'>Titan Crystal</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 0.8rem; margin-top: -15px;'>Electronics Suite</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("NAVIGATION", ["Review Hub", "Market Intel", "Dashboard", "Product Selector"])

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size: 0.8rem;'>? Help<br>← Sign Out</p>", unsafe_allow_html=True)


# ==========================================
# PAGE: REVIEW HUB 
# ==========================================
if page == "Review Hub":
    st.title("Review Entry Dashboard")
    st.markdown("Create high-fidelity product analysis reports. Your reviews drive the market intelligence engine of the Titan ecosystem.")
    st.markdown("---")
    
    # Top Row Inputs (Matches Prototype)
    col1, col2 = st.columns(2)
    with col1:
        product = st.selectbox("PRODUCT ECOSYSTEM", ["", "Apple", "Google"])
    with col2:
        analysis_date = st.date_input("ANALYSIS DATE")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Rating System
    user_feedback = st.radio("PERFORMANCE RATING", ["Excellent (5/5)", "Average (3/5)", "Poor (1/5)"], horizontal=True)
    
    # Text Area
    review_text = st.text_area("COMPREHENSIVE REVIEW", placeholder="Begin your technical evaluation here...", height=200)
    
    # Button Layout
    submit_col1, submit_col2 = st.columns()
    with submit_col2:
        if st.button("Submit Review ➢"):
            if product == "" or not review_text.strip():
                st.warning("Please fill in the Product and Review fields.")
            else:
                with st.spinner("Validating Review..."):
                    # 1. Run the AI
                    raw_output = classifier(review_text)
                    result = raw_output
                    
                    label = result['label']
                    score = result['score']
                    
                    # 2. Translate
                    label_dict = {"LABEL_0": "Negative", "LABEL_1": "Positive", "LABEL_2": "Neutral"}
                    final_sentiment = label_dict.get(label, label)
                    
                    st.success(f"**Review Validated:** Categorized as {final_sentiment} (Confidence: {score:.2f})")
                    
                    # 3. Send to Database
                    try:
                        payload = {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Brand": product,
                            "Review": review_text,
                            "Sentiment": final_sentiment,
                            "Feedback": user_feedback
                        }
                        
                        data = urllib.parse.urlencode(payload).encode('utf-8')
                        req = urllib.request.Request(WEBHOOK_URL, data=data)
                        
                        with urllib.request.urlopen(req, timeout=5) as response:
                            st.toast("✅ Saved to Crystal Suite Database.")
                            
                    except Exception as e:
                        st.error(f"Database sync failed: {e}")

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Bottom Info Cards
    card1, card2, card3 = st.columns(3)
    card1.info("**🛡️ Data Integrity**\n\nAll reviews are cross-referenced with global purchase telemetry to ensure authentic sentiment analysis.")
    card2.info("**✨ AI Synthesis**\n\nOur neural engine will extract key feature mentions to update real-time market share dashboards.")
    card3.info("**👁️ Review Privacy**\n\nIndividual contributor names are anonymized in the public Titan Intelligence export files.")

# ==========================================
# PAGE: MARKET INTEL (Visual Mockup)
# ==========================================
elif page == "Market Intel":
    st.title("Market Analysis & Review Insights")
    st.markdown("Real-time competitive landscape and customer sentiment telemetry.")
    st.markdown("---")
    
    col1, col2 = st.columns()
    with col1:
        st.subheader("Apple Inc. (AAPL)")
        chart_data = pd.DataFrame(np.random.randn(20, 1) * 5 + 189, columns=['Price'])