from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = [
    SystemMessage(content = "You are a teacher."
                  )
]

# Creating loop message system
# We need to exit from chatbot, otherwise it will keep running in console
while True:
    user_input = input("You : ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input == "exit":
        break
    # result = model.invoke(user_input)
# We are sending whole chat now to LLM
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI : ", result.content)

print(chat_history) 
# we also need to save the message by which one [user or ai]
# So, that LLM can understand easily and we also.
# This issue solved by LangChain