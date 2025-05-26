import streamlit as st
import pandas as pd
import time
BMSIT&M 2024-25 Page 5
st.set_page_config(page_title="Real-Time Data Quality Validation", layout="centered")
st.title("Real-Time Data Quality Validation")
placeholder = st.empty()
def load_data():
 try:
 df = pd.read_csv("live_data.csv")
 return df.tail(10)
 except:
 return pd.DataFrame(columns=["timestamp", "temperature", "humidity", "valid"])
while True:
 data = load_data()
 with placeholder.container():
 st.dataframe(data, use_container_width=True)
 time.sleep(2)