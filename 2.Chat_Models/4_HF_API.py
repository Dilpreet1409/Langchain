import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint #hf-api
from dotenv import load_dotenv
print("HF TOKEN:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
load_dotenv()
llm = HuggingFaceEndpoint(
     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", #model used
     task="text-generation"
)
model= ChatHuggingFace(llm=llm)
result= model.invoke("What is the capital of India")
print(result.content)
