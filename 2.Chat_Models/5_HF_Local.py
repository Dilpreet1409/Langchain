from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import warnings
warnings.filterwarnings("ignore")

llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
        do_sample=True
    )
)
model=ChatHuggingFace(llm=llm)
result= model.invoke("what is the Capital of India? ")
print("Preparing model input...")
result = model.invoke("What is the capital of India?")
print("Model response:", result.content)
