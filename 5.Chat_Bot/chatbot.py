from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFacePipeline.from_model_id(
    model_id="",
    task="text-generation",
)

chat_history=[
    SystemMessage(content='You are a helpful AI assistant')
]

while True: #infinitely run until exit input received 
    user_input= input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result=llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content)) #both append user and ai convo
    print("AI: ",result.content)

