import streamlit as st
import pandas as pd
import torch
from datetime import datetime, timedelta
from transformers import pipeline

# ---------- CRM CORE ----------
def load_data():
    try:
        return pd.read_csv("clients.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Name", "Email", "Last Contacted", "Status", "Next Follow-Up"])

def save_data(df):
    df.to_csv("clients.csv", index=False)

st.title("SmartCRM – AI-Enhanced Client Tracker 🧠💼")

df = load_data()

st.subheader("Add New Client")
name = st.text_input("Client Name")
email = st.text_input("Email")
status = st.selectbox("Status", ["New", "Contacted", "In Discussion", "Closed"])
last_contacted = st.date_input("Last Contacted")
next_follow_up = last_contacted + timedelta(days=7)

if st.button("Add Client"):
    new_data = {
        "Name": name,
        "Email": email,
        "Last Contacted": last_contacted,
        "Status": status,
        "Next Follow-Up": next_follow_up
    }
    df = df.append(new_data, ignore_index=True)
    save_data(df)
    st.success("Client added!")

st.subheader("📊 Client Database")
st.dataframe(df)

st.subheader("📅 Clients Needing Follow-Up")
today = datetime.today().date()
follow_up_df = df[pd.to_datetime(df["Next Follow-Up"]).dt.date <= today]
st.dataframe(follow_up_df)
st.subheader("SmartCRM AI Assistant 💬")

from transformers import pipeline
chatbot = pipeline("text-generation", model="google/flan-t5-base")  # keep it simple or use Hugging Face free models

query = st.text_input("Ask a business-related question:")

if st.button("Get AI Reply"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        result = chatbot(query, max_length=100, do_sample=True)
        st.success("AI Reply:")
        st.write(result[0]["generated_text"])

