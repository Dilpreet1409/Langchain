import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint #hf-api
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

print("HF TOKEN:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
load_dotenv()
llm = HuggingFaceEndpoint(
     repo_id="google/gemma-2-2b-it", #model used
     task="text-generation"
)
model= ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template= PromptTemplate(
    template='Give me the name age city of a fictional person. \n {format_instruction}',  # The {format_instruction} is a placeholder that will be filled using partial_variables below.
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instruction()} # partial_variables are values that are set ahead of time (not at runtime). Here, we're filling in the {format_instruction} part of the prompt using the result from parser.get_format_instruction()
)

# prompt=template.format()

# result= model.invoke(prompt)

# final_result= parser.parse(result.content)

# print(final_result)

# print(type(final_result))

chain = template | model | parser

result = chain.invoke()

print(result)

# major flaw of json parser is that it does not enforce schema 