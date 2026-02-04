from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt = PromptTemplate(
    template= "Write a poem about {topic}",
    input_variables=['topic']
)


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

poem_gen_chain = RunnableSequence(prompt,model,parser)

def word_Counter(text):
    return len(text.split())

word_counter = RunnableLambda(word_Counter)

mid_chain = RunnableParallel({
"topic" : RunnablePassthrough(),
"counter" : RunnableLambda(poem_gen_chain)
}
)
final_chain = RunnableSequence(poem_gen_chain,mid_chain)
result = final_chain.invoke({'topic':"AI"})

print(result)