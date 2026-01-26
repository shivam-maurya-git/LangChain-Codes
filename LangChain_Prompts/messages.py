from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Chat History
messages = [
SystemMessage(content = "You are a pirateGPT"),
HumanMessage(content = "Who is Barabosa?")
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))
#AIMessage() = Initialize an AIMessage

print(messages)