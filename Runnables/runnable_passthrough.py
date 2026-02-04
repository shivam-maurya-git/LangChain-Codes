from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

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

poem_gen_chain = RunnableSequence(prompt1, model, parser)
#parser output from model will act as input for prompt2 

mid_chain = RunnableParallel(
    {"topic" : RunnablePassthrough(),
     "explnation" : RunnableSequence(prompt2, model,parser)}
)
# Note here we do not need to pass explanation as input varaibles
final_chain = RunnableSequence(poem_gen_chain,mid_chain)
result = final_chain.invoke({"topic":"Ai"}) # automatcailly output of first prompt will go in second

print(result)