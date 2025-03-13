import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="DataDomz Ensayo",
    page_icon="https://crazy-chrono.com/cdn/shop/files/Logo_Crazy_Chrono_v2_8ff975c0-d3ce-4869-8a94-ac8110ffab28_grande.png?v=1613229989",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Este es un título principal
st.title("Aprendiendo Streamlit")

# Título
st.header("Bases de Streamlit")

# Sub título
st.subheader("Streamlit")

# Párrafo
st.write("El Crono tenía una página y no nos había dicho")

logo_url = "https://crazy-chrono.com/cdn/shop/files/Logo_Crazy_Chrono_v2_8ff975c0-d3ce-4869-8a94-ac8110ffab28_grande.png?v=1613229989"
st.image(logo_url, width=200)

st.write("[![Star](https://img.shields.io/github/stars/fralfaro/MAT281_2023.svg?logo=github&style=social)](https://gitHub.com/fralfaro/MAT281_2023)")

