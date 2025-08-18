''' RunnableSequence is a sequential chain of Runnables in Langchain that executes each step
ne after another, passing the output at one step as the input to the next.
It is useful when you need to compose multiple Runnables together in a structured workflow.'''

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt1=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model=ChatOpenAI()

parser= StrOutputParser() #Define output parser to extract plain string from model response


prompt2=PromptTemplate(
    template='Explain the following text {text}',
    input_variables=['text']
)

chain= RunnableSequence(prompt1,model,parser,prompt2,model,parser) 

# #chain = RunnableSequence(
#     prompt1,     # Generate joke prompt
#     model,       # LLM generates a joke
#     parser,      # Extract string joke
#     prompt2,     # Create explanation prompt using joke
#     model,       # LLM explains the joke
#     parser       # Extract explanation text
# )

print(chain.invoke({'topic':'AI'}))


