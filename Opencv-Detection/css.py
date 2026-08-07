# css.py
import streamlit as st

def load_css():
    st.markdown("""
    <style>
    /* Main container */
    .stApp {
        background-color: #0f0f0f !important;
        color: #ff8c00 !important;
    }

    /* Sidebar container */
    [data-testid="stSidebar"] {
        background-color: #141414 !important;
        color: #ffa500 !important;
        border-right: 1px solid #2a2a2a;
    }

    /* Sidebar labels & text */
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span {
        color: #ffa500 !important;
        font-weight: 500;
    }

    /* Sidebar widgets / select boxes / inputs */
    [data-testid="stSidebar"] div[data-baseweb="select"] div {
        background-color: #1f1f1f !important;
        color: #ffa500 !important;
    }
    
    [data-testid="stSidebar"] input {
        background-color: #1f1f1f !important;
        color: #ffa500 !important;
    }

    /* Sliders */
    [data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] {
        color: #ff8c00 !important;
    }

    /* Headers in sidebar */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] .stSubheader {
        color: #ff8c00 !important;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
    }

    /* Main area headings */
    h1, h2, h3, .stHeader {
        color: #ffa500 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Info boxes & alerts */
    .stAlert {
        background-color: #1a1a1a !important;
        color: #ffa500 !important;
        border-radius: 8px;
        border-left: 5px solid #ff8c00 !important;
    }

    /* Image container */
    .stImage {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(255, 140, 0, 0.25);
    }

    /* Buttons and toggles */
    .stButton button {
        background-color: #ff8c00 !important;
        color: #000000 !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: background 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px rgba(255, 140, 0, 0.3);
    }
    .stButton button:hover {
        background-color: #ffa500 !important;
        box-shadow: 0 4px 12px rgba(255, 165, 0, 0.5);
    }

    /* Dividers */
    .stDivider {
        border-color: #2a2a2a !important;
    }
    </style>
    """, unsafe_allow_html=True)
