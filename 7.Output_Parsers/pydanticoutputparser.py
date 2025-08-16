import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint #hf-api
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

print("HF TOKEN:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

load_dotenv()
llm = HuggingFaceEndpoint(
     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", #model used
     task="text-generation"
)
model= ChatHuggingFace(llm=llm)

class Person (BaseModel):
    name: str = Field(description= 'Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description= 'Name of the city the person belongs to')

parser= PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name age and city of a fictional {place} person. \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
    )
chain= template|model|parser
final_result= chain.invoke({'place':'Sri Lankan'})
print(final_result)

# prompt=template.invoke({'place':'Indian'})

# result= model.invoke(prompt)

# final_result=parser.parse(result.content)

# print(final_result)