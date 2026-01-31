from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template= 'Write 5 facts about planet {planet_name}',
    input_variables=["planet_name"]
)

llm = HuggingFaceEndpoint(repo_id="DeepHat/DeepHat-V1-7B",
                          task = "text-generation")

prompt2 = PromptTemplate(
    template = "Translate following text into American English {text}",
    input_variables=["text"]
)
parser = StrOutputParser()
model = ChatHuggingFace(llm = llm)

chain = prompt | model | parser | prompt2 | model | parser

result = chain.invoke({'planet_name':"mercury"})

print(result)

# chain.get_graph().print_ascii()
