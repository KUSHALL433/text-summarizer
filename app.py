from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Summary Generator",
    page_icon="📝",
)
st.title('Text Summarizer.',anchor=False)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("API Key not found. Please check your .env file.")
else:
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)

    template = PromptTemplate(
        template='Give a 5 line summary on {text}',
        input_variables=['text']
    )

    user_input = st.text_input("Enter a topic or large text to generate a summary")

    if st.button('Submit') and user_input.strip():
        prompt = template.invoke({'text': user_input})
        result = model.invoke(prompt)
        st.write(result.content)
    else:
        st.warning("Please enter some text or topic.")




