from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

st.title('Google Gemini Text Summarizer... Generate short summary of long texts')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("API Key not found. Please check your .env file.")
else:
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)

    template = PromptTemplate(
        template='Give a 5 line summary on {text}',
        input_variables=['text']
    )

    user_input = st.text_input("Enter long text to generate a summary")

    if st.button('Submit') and user_input.strip():
        prompt = template.invoke({'text': user_input})
        result = model.invoke(prompt)
        st.write(result.content)
