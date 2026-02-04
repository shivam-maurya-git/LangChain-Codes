from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableBranch
from pydantic import BaseModel
from typing import Literal

load_dotenv()

class email_cat(BaseModel):
    type : Literal["Complain","Refund","General Query"]
model = model1 = ChatHuggingFace(llm = HuggingFaceEndpoint(
     repo_id="MiniMaxAI/MiniMax-M2.1",
                          task = "text-generation"))
parser1  = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=email_cat,)

prompt1 = PromptTemplate(
    template="Categorize the following email in format - {format_instruction} and email - {email}",
    input_variables=['email'],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template="Give a sorry reply to user who have filed complain email"
)

prompt3 = PromptTemplate(
    template="Give a timeline for refund to user who have filed email or refund"
)
prompt4 = PromptTemplate(
    template="Give a answer of user's general query"
)

cat_gen_chain = RunnableSequence(prompt1,model,parser2) # Need to parse using Pydantic Output Parser
# it will give output as "type='Complain'" but object type will be <class '__main__.email_cat'>
print(type(cat_gen_chain.invoke({'email' : "Your products are very bad."})))

# In current LangChain (≥0.1.x / ≥0.2.x), the first argument of each condition pair must be a callable that returns a boolean — not a string.
branch_chain = RunnableBranch(
    (lambda x : x["type"]=='Complain',RunnableSequence(prompt2,model,parser1)),
    (lambda x : x["type"]=='Refund',RunnableSequence(prompt3,model,parser1)),
    RunnableSequence(prompt4,model,parser1),
)

# Connvert Pydantic object into dict

to_dict = RunnableLambda(lambda x: x.model_dump())


final_chain = RunnableSequence(cat_gen_chain,to_dict,branch_chain)

result = final_chain.invoke({'email' : "Your products are very bad."})
print(result)