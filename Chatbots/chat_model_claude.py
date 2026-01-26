from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv

model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

results = model.invoke("What are the use case of gen ai?")

print(results)