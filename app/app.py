"""Simple educational Streamlit explorer for locally saved GDELT metadata."""
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

st.set_page_config(page_title="European Security Monitor — Demo", layout="wide")
st.title("European Security Monitor")
st.caption("Educational news metadata explorer · Not a live alerting or intelligence service")

DB = Path(__file__).resolve().parents[1] / "data" / "demo_articles.db"
if not DB.exists():
    st.info("No saved data yet. Run `python src/update_data.py` and reload this page.")
    st.stop()

with sqlite3.connect(DB) as connection:
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
    ).fetchall()
    if not tables:
        st.info("Database has no article table yet. Run the updater.")
        st.stop()
    df = pd.read_sql_query(
        """SELECT title, topic, seen_date, source_country, source_domain, url
        FROM articles ORDER BY seen_date DESC LIMIT 2000""",
        connection,
    )

if df.empty:
    st.info("No records available in the local demo database.")
    st.stop()

topics = sorted(df["topic"].dropna().unique().tolist())
selected = st.multiselect("Topic (illustrative keyword label)", topics, default=topics)
filtered = df[df["topic"].isin(selected)]
st.metric("Saved articles in selected view", len(filtered))
st.bar_chart(filtered["topic"].value_counts())
st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
    column_config={"url": st.column_config.LinkColumn("Original source")},
)
st.warning(
    "This demo does not verify articles, distinguish unique incidents, "
    "estimate threat levels, or generate alerts. Topic labels can be wrong."
)
