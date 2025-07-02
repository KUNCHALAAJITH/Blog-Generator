import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers

#Ffunction to get response from LLama 2 model

def getLLamaresponse(input_text, no_words, blog_style):
    try:
        # Convert no_words to int safely
        word_count = int(no_words)
    except ValueError:
        word_count = 500  # fallback default

    # Heuristic: average word length + space ≈ 1.3 tokens per word
    estimated_tokens = int(word_count * 1.5)

    # LLaMA model
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={
            'max_new_tokens': estimated_tokens,
            'temperature': 0.7,         # increase slightly for fluency
            'top_p': 0.9,
            'repetition_penalty': 1.2
        }
    )

    # Prompt template
    prompt = PromptTemplate(
        input_variables=["input_text", "no_words", "blog_style"],
        template="Write a {blog_style} blog post about {input_text} with approximately {no_words} words.Avoid repetition and keep it coherent."
    )

    # Generate response
    response = llm(prompt.format(input_text=input_text, no_words=no_words, blog_style=blog_style))
    return response






st.set_page_config(page_title="Blog Generator", page_icon=":robot_face:", layout="wide", initial_sidebar_state="expanded")
st.header("Blog Generator with Llama 2")
input_text = st.text_input("Enter your blog topic or title:", placeholder="e.g., The Future of AI in Healthcare")

# creating 2 more columns for additional 2 fiels
col1, col2= st.columns([5,5])

with col1:
    no_words = st.text_input("Number of words:", placeholder="e.g., 500", value="500")
with col2:
    blog_style=st.selectbox("Blog Style:", options=["Informative", "Conversational", "Technical", "Casual"], index=0)
submit= st.button("Generate Blog")
if submit:
    if input_text and no_words:
        st.write(getLLamaresponse(input_text,no_words,blog_style))