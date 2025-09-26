import streamlit as st
import pandas as pd

st.line_chart(pd.read_csv("./training/data.csv"))
