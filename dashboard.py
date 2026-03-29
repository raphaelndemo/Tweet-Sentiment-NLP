import streamlit as st
from transformers import pipeline
import yfinance as yf
import requests
from datetime import datetime
import pandas as pd
import urllib.request
import urllib.parse

# --- CONFIGURATION ---
# Replace with your Google Apps Script Web App URL
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbwxe6QwqyOPk-icikFapgqzXw-En--93zbfM74ZZ3TNnOV_0i-H8DzGgP-nB3AuHypLeQ/exec"

st.set_page_config(page_title="Titan Review Portal", layout="wide")

# --- LOAD THE AI MODEL ---
@st.cache_resource
def load_model():
    # This loads the 'safetensors' model folder we just pushed
    return pipeline("text-classification", model="./titan_sentiment_model")

classifier = load_model()

# --- NAVIGATION STATE ---
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

def toggle_view():
    st.session_state.show_dashboard = not st.session_state.show_dashboard

# --- VIEW 1: CUSTOMER REVIEW PORTAL ---
if not st.session_state.show_dashboard:
    st.title("📝 Customer Review Portal")
    
    product = st.selectbox("Select the product:", ["", "Apple Product", "Google Product"])
    review_text = st.text_area("Your Review:", placeholder="How was your experience?")
    user_feedback = st.radio("Rating:", ["Excellent", "Average", "Poor"], index=1)
    
    if st.button("Submit Review"):
        if product == "" or not review_text.strip():
            st.warning("Please fill in all fields.")
        else:
            # 1. Run the AI
            raw_output = classifier(review_text)
            result = raw_output[0]
            
            label = result['label']
            score = result['score']
            
            # 2. Translate the AI label to English
            label_dict = {"LABEL_0": "Negative", "LABEL_1": "Positive", "LABEL_2": "Neutral"}
            final_sentiment = label_dict.get(label, label)
            
            st.success(f"Review Analyzed as: **{final_sentiment}** (Confidence: {score:.2f})")
            
            # 3. Send to Google Sheets safely!
            try:
                payload = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Brand": product,
                    "Review": review_text,
                    "Sentiment": final_sentiment,
                    "Feedback": user_feedback
                }
                
                # The safe bypass to prevent freezing
                data = urllib.parse.urlencode(payload).encode('utf-8')
                req = urllib.request.Request(WEBHOOK_URL, data=data)
                
                with urllib.request.urlopen(req, timeout=5) as response:
                    st.toast("✅ Saved to Google Sheets!")
                    
            except Exception as e:
                st.error(f"Database error: {e}")
    
    st.divider()
    st.checkbox("Admin: View Dashboard", value=st.session_state.show_dashboard, on_change=toggle_view)

# --- VIEW 2: ADMIN ANALYTICS ---
else:
    st.title("📈 Admin Market Dashboard")
    st.button("← Back to Portal", on_click=toggle_view)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Apple (AAPL)")
        st.line_chart(yf.Ticker("AAPL").history(period="1mo")['Close'])
    with col2:
        st.subheader("Google (GOOGL)")
        st.line_chart(yf.Ticker("GOOGL").history(period="1mo")['Close'])

    st.subheader("⚠️ Trends & Issues")
    st.info("System currently monitoring live feedback for hardware sentiment shifts.")