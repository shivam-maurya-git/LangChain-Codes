from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Loading API Key
load_dotenv()

# We can also load chat_history from a chat_history file
chat_history = []

# Saving as tuple
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Explain {topic} in simple terms.")
])

# Chat Model
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

# Chat Model Function : Multi-turn conversation + dynamic Prompt + Context (Chat History)

def main():
  while True:
   domain = input("enter domain name : ")
   topic = input("enter topic name: ")

   input_variables = {
      "domain" : domain,
      "topic" : topic,
      "chat_history": chat_history
   }

   chat_history.append(SystemMessage(domain))
   chat_history.append(HumanMessage(topic))

   prompt = chat_template.invoke(input_variables)
   results = model.invoke(prompt)
   print("AI  :", results.content)
   chat_history.append(AIMessage(results.content))

   # Next steps for user:
   user_input = input("AI : Any more questions on same topic or (No/new topic) ")
   chat_history.append(HumanMessage(user_input))

   if user_input == 'No':
     break
   
   elif user_input == "new topic":
      continue
   else:
     while True:
       
       input_variables = {
         "domain" : domain,
      "topic" : topic,
      "chat_history": chat_history,
      "user_input":user_input
   }
    #    results = model.invoke(user_input) # It will not use chat_history
       prompt = chat_template.invoke(input_variables)
       results = model.invoke(prompt)
       print("AI  :", results.content)
       chat_history.append(AIMessage(results.content))

       user_input = input("AI : Any more questions on same topic or (No/new topic) ")
       if user_input == "No":
         return
       elif user_input == "new topic":
         break # Goes to outer loop : new domain
       else:
         pass
main()

print(chat_history)
