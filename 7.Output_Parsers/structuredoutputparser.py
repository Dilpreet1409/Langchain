import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint #hf-api
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

print("HF TOKEN:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

load_dotenv()
llm = HuggingFaceEndpoint(
     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", #model used
     task="text-generation"
)
model= ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template= PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction:':parser.get_format_instructions()}
)

# prompt= template.invoke({'topic':'Black hole'})
# result= model.invoke(prompt)
# final_result= parser.parse(result.content)
# print(final_result)

chain = template| model| parser
result= chain.invoke({'topic':'Black hole'})
print(result)

#drawback of structured output parser is that you cant do data validation