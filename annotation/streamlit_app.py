import streamlit as st
from collections import defaultdict

st.set_option("client.toolbarMode", "minimal")

pages = {
    "bearvision": [
        st.Page("leaderboard.py", title="Leaderboard"),
        st.Page("annotate.py", title="Annotate"),
        st.Page("plots.py", title="Graphs")
    ]
}

if not "leaderboard" in st.session_state:
    st.session_state.leaderboard = defaultdict(int)

pg = st.navigation(pages)
pg.run()
