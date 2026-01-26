from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = "gpt-4",temperature=0.5,max_completion_tokens=10)
results = model.invoke("What are the transformers in AI?")

# print(results)
print(results.content)