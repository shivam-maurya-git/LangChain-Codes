from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import streamlit as st
# Streamlit is an open-source Python library for building interactive web apps using only Python. It's ideal for creating dashboards, data-driven web apps, reporting tools


load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

st.header("Reseach Tool") # main heading

user_input = st.text_input("Enter your prompt.") # input box with title "Enter your prompt."

# When button is clicked
if st.button('Summarize'): # button with name
    result = model.invoke(user_input) #create results
    st.write(result.content)   # Write on display