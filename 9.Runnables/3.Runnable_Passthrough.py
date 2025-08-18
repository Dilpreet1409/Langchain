'''RunnablePassthrough is a special runnable primitive that simply returns 
the input as output without modifying it'''

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt1=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model=ChatOpenAI()

parser= StrOutputParser() 

prompt2=PromptTemplate(
    template='Explain the following text {text}',
    input_variables=['text']
)

joke_generator_chain=RunnableSequence(prompt1,model,parser)

parallel_chain= RunnableParallel({
    'joke': RunnablePassthrough(),
    'Explaination': RunnableSequence(prompt2,model,parser)
})

final_chain= RunnableSequence(joke_generator_chain,parallel_chain)

passthrough= RunnablePassthrough()

print(final_chain.invoke({'topic':"AI"}))
