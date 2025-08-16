from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

load_dotenv()
@st.cache_resource #It loads the model once, keeps it in memory, and reuses it across reruns.
def load_model():
    model = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ,
        device=-1,  # CPU 
        task="text-generation",
        pipeline_kwargs=dict(
            temperature=0.7,
            max_new_tokens=350,
            do_sample=True,
            return_full_text=False # Important to avoid prompt echo in output
        )
    )
    print("Model loaded:", model)
    return model
model = load_model()

# Streamlit UI
st.set_page_config(page_title="Research Tool", layout="centered")
st.header("Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)
st.write(paper_input)
style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)
st.write(style_input)
length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)
st.write(length_input)
#template
template=load_prompt('template.json')

#fill placeholders in the template # invoke() replaces placeholders with the selected values

if st.button('Summarise'):
  try:
    chain= template | model
    result=chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input})
    st.markdown(result)
  except Exception as e:
      st.error(f"Model invocation failed: {e}")





