from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
    Summarize the research paper titled "{paper_input}" for a {style_input} audience.
    Keep the explanation {length_input}. Avoid repeating this prompt in the answer.
    Focus only on the core ideas, mathematical insights, and analogies if possible.
    If information is missing, respond with "Insufficient information available."
    """,
    input_variables=["paper_input", "style_input", "length_input"]
)

template.save("template.json")