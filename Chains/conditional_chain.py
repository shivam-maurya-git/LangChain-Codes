from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda
load_dotenv()

parser1 = StrOutputParser()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
class Feedback(BaseModel):
    sentiment : Literal["Positive","Negative"] = Field(description= "Sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)
promp1 = PromptTemplate(
    template= 'Classify the sentiment of the following feedback into positive or negative. Feedback : {feedback},\n {format_instruction}',
    input_variables= ['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
    )

classifier_chain = promp1 | model | parser2

feedback = "This is a very good phone and much better than your old phone."

result1 = classifier_chain.invoke({'feedback':feedback})
# Ouput of above chain need to be consistent in type setting (Positive , not pos, good), so that we can exceute condtional chain correctly

# print(result1.sentiment)
# print(type(result1)) #<class '__main__.Feedback'>

prompt2 = PromptTemplate(
    template = 'write an appropraite response to user for the positive feedback {feedback}',
    input_variables= ['feedback']
)

prompt3 = PromptTemplate(
    template = 'write an appropraite response to user for this negative feedback {feedback}',
    input_variables= ['feedback']
)

branch_chain = RunnableBranch( # Tuples  : Condition , What to do if condition true
(lambda x : x.sentiment =='Positive', prompt2 | model | parser1),
(lambda x : x.sentiment =='Negative', prompt3 | model | parser1),
(RunnableLambda(lambda x : "could not find sentiment"))
# We need to convert this lambda into runnable function because this is not going to chain
#RunnableLambda : Converts it into chain

)

final_chain = classifier_chain | branch_chain

result = final_chain.invoke({"feedback":feedback})
print(result)