from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# JSON output parser can be used when some application need json format output
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.1",
                          task = "text-generation")

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name : str = Field(description= "Name of person")
    age : int = Field(gt=18, description="Age of person")
    city : str = Field(description= "Name of city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template  = PromptTemplate(
    template="Generate the name, age and city of a fictional {country} person \n {format_instruction}",
input_variables=["country"],
partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.invoke({"country" : "India"})
# print(prompt)  
# prompt will tell user's request along with whole prompt.

# result = model.invoke(prompt)

# print(result)
# print(type(result)) #<class 'langchain_core.messages.ai.AIMessage'>

# final_result = parser.parse(result.content)

# print(final_result)
#name='Arjun Mehta' age=28 city='Mumbai'

chain = template | model | parser

print(chain.invoke({"country":"soviet"})) #name='Mikhail Ivanovich Volkov' age=35 city='Leningrad'





