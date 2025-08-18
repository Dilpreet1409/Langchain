''' Runnable Branch is control flow component n Langchain that allows you to conditionally route 
input data to different chains or runnables based on custom logic.
It functions like an if/elif/else block for chains- where you define a set of condition function,
each associated with a runnable(eg. LLM call, prompt chain, or tool)
The first matchmaking condition is executed. If no condition matches, a default
runnable is used(if provided)'''

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch


load_dotenv()

prompt1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Summarise the following text \n {text}',
    input_variables=['text']
)

model=ChatOpenAI()

parser=StrOutputParser()

#report_gen_chain=RunnableSequence(prompt1,model,parser)
report_gen_chain= prompt1 | model | parser #LCEL (LangChain Expression Language)

branch_chain=RunnableBranch(
    (lambda x:len(x.split())>500, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))

