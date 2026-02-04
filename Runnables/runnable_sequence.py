from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template= "Write a poem about {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template= "Write explantion of {poem}",
    input_variables=["poem"]
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser,prompt2,model, parser)
#parser output from model will act as input for prompt2 

result = chain.invoke({"topic":"Ai"}) # automatcailly output of first prompt will go in second

print(result)