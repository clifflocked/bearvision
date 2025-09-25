import streamlit as st
from collections import defaultdict

pages = {
    "bearvision": [
        st.Page("leaderboard.py", title="Leaderboard"),
        st.Page("annotate.py", title="Annotate"),
    ]
}

if not "leaderboard" in st.session_state:
    st.session_state.leaderboard = defaultdict(int)

pg = st.navigation(pages)
pg.run()
