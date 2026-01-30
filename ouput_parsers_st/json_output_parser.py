from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# JSON output parser can be used when some application need json format output
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.1",
                          task = "text-generation")

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template  = PromptTemplate(
    template='Create a sample story with title, penname, genre \n ' \
'{format_instruction}',
input_variables=[],
partial_variables={'format_instruction':parser.get_format_instructions()}
# Because it fills before runtime, and infomed by parser.get_format_instructions, not the user.
# So, now this partial_variable fill {format_instruction} by help of parser object, which is json type
)

prompt = template.format() # no need to write within format, because it is static prompt
# print(prompt)  #Create a sample story with title, penname, genre 
#  Return a JSON object.

result = model.invoke(prompt)

print(type(result))
# <class 'langchain_core.messages.ai.AIMessage'> with information that content need to json type
#  Very important : overall all models give output in JSON format, but some of them do not have capcity to print content as seprately also a JSON object.

# final_result = parser.parse(result.content)
# It will give parse it is a proper JSON object and format
# print(final_result) # python treat json object as type dict

### WIth use of chain ##
# chain = template | model | parser
# result = chain.invoke({}) # You need to always pass one arguments, even though we do not have any input, we will pass a blank dict

# print(result)
