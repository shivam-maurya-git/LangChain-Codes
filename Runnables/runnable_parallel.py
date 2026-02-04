from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

model1 = ChatHuggingFace(llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    
    task="text-generation"))

model2 = ChatHuggingFace(llm= HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.7-Flash",
    
    task="text-generation"))

prompt = PromptTemplate(template="Write a {type} about {topic}",
                        input_variables=["type","topic"]
                        )
parser = StrOutputParser()

chain = RunnableParallel( # dict format
{
    'tweet' : RunnableSequence(prompt,model1, parser),
    'linkedin' : RunnableSequence(prompt,model2,parser)
}
)

result = chain.invoke({"type":"linkedin","topic":"AI"})
# It will give both tweet and linkedin (do not confuse with conditionals)
print(result)