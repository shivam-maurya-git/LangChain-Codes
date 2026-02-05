from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings()

# Create FAISS vector store
vectorstore = FAISS.from_documents(
    documents=all_docs, 
    embedding=embedding_model)


# MultiQueryRetriever will use LLM to generate subqueries
multiquery_retriever = MultiQueryRetriever.from_llm(
    #normal vector type search (for ranking top results while running each subquery) + can use mmr also
    #This sets the retriever to return top-5 documents for each generated query (not overall). After deduplication there may be more than 5 results.
    # if you want only top 5 :apply re-ranking and then select top-5.
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)

query = "How to improve energy levels and maintain balance?"

results= multiquery_retriever.invoke(query)

for i, doc in enumerate(results): # doc : 
    print(i)
    print(doc.page_content)


