from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# HuggingFaceEndpoint : When we are using API
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    # repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # TinyLlama/TinyLlama-1.1B-Chat-v1.0 does NOT support HF chat_completion
    # ChatHuggingFace expects a chat-compatible inference provider
    task="text-generation"
    
)

model = ChatHuggingFace(llm=llm)

results = model.invoke("What are transformers in AI?")

print(results)