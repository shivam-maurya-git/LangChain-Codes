from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableLambda
# text = """
# LangChain is an open source framework with a pre-built agent architecture and integrations for any model or tool — so you can build agents that adapt as fast as the ecosystem evolves

# LangChain is the easiest way to start building agents and applications powered by LLMs. With under 10 lines of code, you can connect to OpenAI, Anthropic, Google, and more. LangChain provides a pre-built agent architecture and model integrations to help you get started quickly and seamlessly incorporate LLMs into your agents and applications.
# We recommend you use LangChain if you want to quickly build agents and autonomous applications. Use LangGraph, our low-level agent orchestration framework and runtime, when you have more advanced needs that require a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency.
# LangChain agents are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. You do not need to know LangGraph for basic LangChain agent usage.
# """

loader = PyPDFLoader("doc_loaders\Challenges_in_teaching_laws_of_exponents.pdf")

data = loader.load() # Will create document objects
splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator=" "
)

# # result = splitter.split_text(text) # Gives a list of chunks
# result = splitter.split_documents(data) # Split document objects
# # and each chunk returned will be also a document object
# print(result)

doc_loader = RunnableLambda(lambda x : x.load())
data = RunnableLambda(lambda x  : splitter.split_documents(x))

chain = doc_loader | data
result = chain.invoke(loader)
print(result)