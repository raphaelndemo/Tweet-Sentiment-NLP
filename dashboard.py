import streamlit as st
from transformers import pipeline
from datetime import datetime
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
import yfinance as yf

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Titan Crystal Suite",
    page_icon="📊",
    layout="wide",
)

WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbx6_SKVjZirMac0vyPgQUwh_YcElMPrufdhdJnHIjRYHk-t8ZGRLJE9fV79dChIPERXuw/exec"

# ---------------- CACHE STOCK DATA ----------------
@st.cache_data(ttl=300)
def load_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="7d", interval="1h")

# ---------------- SESSION STATE ----------------
if "sentiments" not in st.session_state:
    st.session_state.sentiments = []

# ---------------- APPLE-LEVEL UI ----------------
st.markdown("""
<style>
/* ===== GLOBAL ===== */
.stApp {
    background: linear-gradient(180deg, #F9FAFB 0%, #EEF2F7 100%);
    font-family: -apple-system, BlinkMacSystemFont, "San Francisco", sans-serif;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(0,0,0,0.05);
}

/* ===== HEADINGS ===== */
h1 {
    font-weight: 700;
    letter-spacing: -0.5px;
}
h2, h3 {
    font-weight: 600;
}

/* ===== CARDS ===== */
.card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(135deg, #007AFF, #005FCC);
    color: white !important;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
    border: none;
    box-shadow: 0 6px 18px rgba(0,122,255,0.25);
    transition: all 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(0,122,255,0.35);
}
.stButton>button:active {
    transform: scale(0.97);
}

/* ===== INPUTS ===== */
.stTextInput input, .stTextArea textarea {
    border-radius: 10px !important;
    border: 1px solid #E5E7EB !important;
    background: white !important;
}

/* ===== DIVIDER ===== */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #D1D5DB, transparent);
}

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Titan Crystal")
    st.caption("Electronics Intelligence Suite")
    st.markdown("---")
    page = st.radio("Navigation", ["Review Hub", "Market Intel", "Dashboard", "Product Selector"])
    st.markdown("---")
    st.caption("Help • Sign Out")

# ---------------- REVIEW HUB ----------------
if page == "Review Hub":
    @st.cache_resource
    def load_model():
        return pipeline("text-classification", model="./titan_sentiment_model")

    classifier = load_model()
    st.title("Review Entry")

    # 1. Create a "Switch" in the session state to track if a review was submitted
    if "review_submitted" not in st.session_state:
        st.session_state.review_submitted = False

    # 2. IF SUBMITTED: Show the Thank You screen
    if st.session_state.review_submitted:
        st.markdown('<div class="card" style="text-align: center; padding: 50px;">', unsafe_allow_html=True)
        st.subheader("🎉 Success!")
        
        # The clean, user-friendly message
        st.write("Your review has been saved. Thank you for your feedback!")
        
        # Show what the AI decided (without the decimals)
        st.success(f"**Analysis:** {st.session_state.last_sentiment}")
        
        st.write("") # Blank space
        
        # Button to reset the form
        if st.button("← Submit Another Review"):
            st.session_state.review_submitted = False
            try:
                st.rerun() 
            except AttributeError:
                st.experimental_rerun() 
                
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. IF NOT SUBMITTED: Show the normal input form
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            product = st.selectbox("Product", ["", "Apple", "Google"])
            rating = st.radio("Rating", ["1", "2", "3", "4", "5"], horizontal=True)
        with col2:
            review_text = st.text_area("Review", height=150)

        if st.button("Submit Review"):
            if not product or not review_text.strip():
                st.warning("Please complete all fields")
            else:
                with st.spinner("Analyzing..."):
                    # --- CRASH-PROOF AI SCANNER ---
                    raw_result = classifier(review_text)
                    raw_str = str(raw_result).upper()
                    
                    if 'POS' in raw_str or 'LABEL_1' in raw_str or 'LABEL_2' in raw_str:
                        sentiment = "Positive review"
                    elif 'NEG' in raw_str or 'LABEL_0' in raw_str:
                        sentiment = "Negative review"
                    else:
                        sentiment = "Neutral review"

                    st.session_state.sentiments.append({"time": datetime.now(), "sentiment": sentiment})
                    
                    # Save the sentiment so the Thank You screen can display it
                    st.session_state.last_sentiment = sentiment

                    # --- FIRE AND FORGET DATABASE ---
                    payload = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                        "Brand": product, 
                        "Review": review_text, 
                        "Sentiment": sentiment, 
                        "Rating": rating
                    }
                    try:
                        import urllib.parse
                        import subprocess
                        form_data = urllib.parse.urlencode(payload)
                        curl_command = ['curl', '-s', '-d', form_data, WEBHOOK_URL]
                        subprocess.Popen(curl_command)
                    except Exception:
                        pass 
                    
                    # Flip the switch to hide the form and show the Thank You screen!
                    st.session_state.review_submitted = True
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------- MARKET INTEL ----------------
elif page == "Market Intel":
    st.title("Market Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Apple")
        apple = load_stock_data("AAPL")
        if not apple.empty:
            st.line_chart(apple['Close'])
            st.metric("Price", f"${apple['Close'].iloc[-1]:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Google")
        google = load_stock_data("GOOGL")
        if not google.empty:
            st.line_chart(google['Close'])
            st.metric("Price", f"${google['Close'].iloc[-1]:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- DASHBOARD ----------------
elif page == "Dashboard":
    st.title("Sentiment Analytics")

    if st.session_state.sentiments:
        df = pd.DataFrame(st.session_state.sentiments)

        mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
        df['value'] = df['sentiment'].map(mapping)

        df = df.sort_values("time")
        df.set_index("time", inplace=True)

        trend = df['value'].resample('1min').mean().fillna(0)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Trend")
        st.line_chart(trend)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Distribution")
        st.bar_chart(df['sentiment'].value_counts())
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PRODUCT SELECTOR ----------------
elif page == "Product Selector":
    st.title("Product Comparison")

    col1, col2 = st.columns(2)

    with col1:
        brand1 = st.selectbox("Brand A", ["Apple", "Google"])
    with col2:
        brand2 = st.selectbox("Brand B", ["Apple", "Google"])

    df = pd.DataFrame({
        "Feature": ["Performance", "Battery", "Ecosystem"],
        brand1: np.random.randint(6, 10, 3),
        brand2: np.random.randint(6, 10, 3)
    })

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df)
    st.markdown('</div>', unsafe_allow_html=True)