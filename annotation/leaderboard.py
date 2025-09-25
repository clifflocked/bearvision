import streamlit as st
from streamlit import session_state as ss

st.write("## Leaderboard")
top = sorted(ss.leaderboard.items(), key=lambda x: x[1], reverse=True)
lb_names = ""
lb_scores = ""

col1, col2 = st.columns([6, 1])

for i in range(len(top)):
    if i == 0:
        worth = "####"
    elif i == 1:
        worth = "#####"
    else:
        worth = ""

    lb_names += f"{worth} {i + 1}. {top[i][0]}\n"
    lb_scores += f"{worth} {top[i][1]} {' ' * 6}\n"

with col1:
    st.write(lb_names)

with col2:
    st.write(lb_scores)
