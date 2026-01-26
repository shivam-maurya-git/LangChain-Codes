from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,load_prompt

from dotenv import load_dotenv
import streamlit as st
# Streamlit is an open-source Python library for building interactive web apps using only Python. It's ideal for creating dashboards, data-driven web apps, reporting tools


load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

st.header("Reseach Tool") # main heading

# Creating all dropdowns
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )
 # selectbox creates dropdown
style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# We have take prompt from external file as fstring
# or, we can also create prompt here using Prompt Template (more validation)
# or, we can create prompt in another file using Prompt Template and load here
input_variables = {'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input}

# Loading f-string template from external file
template = load_prompt('prompt_template_fstring.json')
# When button is clicked
if st.button('Summarize'): # button with name
# Way 1 : Invoke template and then invoke model 
      prompt = template.invoke(input_variables)
      result = model.invoke(prompt)
# Way 2 : Create a chain (pipeline and invoke it)
#| → pipe operator (passes output → input)
#     chain = template | model
#     # Invoked by chain containing ChatModel and template from template.json
#     result = chain.invoke(input_variables)
      st.write(result.content)   # Write on display