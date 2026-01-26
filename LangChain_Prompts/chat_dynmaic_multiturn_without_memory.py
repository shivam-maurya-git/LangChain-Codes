from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Somewhat similar to PromptTemplate
# chat_template = ChatPromptTemplate(
#  [SystemMessage(content="You are a helpful {domain} expert"),
#  HumanMessage(content="Explain {topic} in 4 bullet points.")
# ]
# )

# Saving as tuple
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert"),
    ("human", "Explain {topic} in simple terms.")
])

chat_history = []
# print(prompt)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
def main():
  while True:
   domain = input("enter domain name : ")
   topic = input("enter topic name: ")
   input_variables = {
      "domain" : domain,
      "topic" : topic
   }
   chat_history.append(SystemMessage(domain))
   chat_history.append(HumanMessage(topic))
   prompt = chat_template.invoke(input_variables)
   results = model.invoke(prompt)
   print("AI  :", results.content)
   chat_history.append(AIMessage(results.content))
   user_input = input("AI : Any more questions on same topic or (No/new topic) ")
   chat_history.append(HumanMessage(user_input))
   if user_input == 'No':
     break
   
   elif user_input == "new topic":
      continue
   else:
     while True:
       results = model.invoke(user_input)
       print("AI  :", results.content)
       chat_history.append(AIMessage(results.content))
       user_input = input("AI : Any more questions on same topic or (No/new topic) ")
       if user_input == "No":
         return
       elif user_input == "new topic":
         break
       else:
         pass
main()

print(chat_history)

# In this code, we are saving chat history, but we are not using it for invoking.