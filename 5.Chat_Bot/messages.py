from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFacePipeline.from_model_id(
    model_id= " ",
    task="text-generation"
)
messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content="Tell me about Langchain")
]

result = llm.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)