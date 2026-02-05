from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document # Document object
from dotenv import load_dotenv

load_dotenv()

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding_model = OpenAIEmbeddings()

# from_documents : Create a Chroma vectorstore from a list of documents
# Vector store creation + adding documents.
vector_store = Chroma.from_documents(
documents= documents,
embedding= embedding_model,
collection_name='sample'
)
#If a persist_directory is specified, the collection will be persisted there. Otherwise, the data will be ephemeral in-memory.

# We can get documents directly from vector store and also with the help of ret on vector store
# vector_store.similarity_search( # Directly from vector store
#     query="Who among these are a bowler?",
#     k=2 #Number of results to return. Defaults to 4.
# ) 
# vector store like Croma already have (in-built) capcity of similarity search but such vector stores uses one kind of search startgey, but using your own ret, you can experiment with various types of search startgies.

#Return VectorStoreRetriever initialized from this VectorStore.
ret = vector_store.as_retriever(search_kwargs={"k":2}) #Keyword arguments to pass to the search function.

query = "What is Croma used for?"
results = ret.invoke(query)

for i, doc in enumerate(results): # doc : 
    print(i)
    print(doc.page_content)