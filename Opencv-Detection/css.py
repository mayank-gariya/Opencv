import streamlit as st
from pathlib import Path

def load_css():
    """Load custom CSS from the styles folder."""
    css_file = Path(__file__).parent / "styles" / "main.css"
    if css_file.exists():
        with open(css_file, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning("CSS file not found. Proceeding with default styling.")
