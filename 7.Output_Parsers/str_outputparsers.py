import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint #hf-api
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


print("HF TOKEN:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
load_dotenv()
llm = HuggingFaceEndpoint(
     repo_id="google/gemma-2-2b-it", #model used
     task="text-generation"
)
model= ChatHuggingFace(llm=llm)

# 1st prompt -> Detailed report
template1=PromptTemplate(
   template="Write a detail report on {topic}",
   input_variables=['topic']
)
# 2nd prompt -> Summary
template2=PromptTemplate(
   template="Write a five line summary on the following text,/n {text}",
   input_variables=['text']
)
prompt1=template1.invoke({'topic':'Blockchain'})
result=model.invoke(prompt1)

prompt2= template2.invoke({'text':result.content})
result1=model.invoke(prompt2)

print(result1.content) 




