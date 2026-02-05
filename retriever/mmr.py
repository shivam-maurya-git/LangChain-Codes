from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

load_dotenv()

embedding_model = OpenAIEmbeddings()

vector_store = FAISS.from_documents(
    documents= docs,
embedding= embedding_model
)

ret = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":3,"lambda_mult":1}
#lambda_mult [0,1] : controls diversity of top results
# 0 : same as vector ret, 1 : very diverse
)

query = "What is langchain?"
results = ret.invoke(query)

for i, doc in enumerate(docs): # doc : 
    print(i)
    print(doc.page_content)