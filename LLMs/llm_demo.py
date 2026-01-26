from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI("gpt-4o-mini")

result = llm.invoke("What is the transfomer in AI?")

print(result)