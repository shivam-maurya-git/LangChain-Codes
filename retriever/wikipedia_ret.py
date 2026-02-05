from langchain_community.retrievers import WikipediaRetriever

ret = WikipediaRetriever(
    top_k_results=2, lang='en'
)

query = "top grosssing indian movies"

docs = ret.invoke(query) # able to use invoke --> ret is an runnable
# print(docs) # list of document objects : each item is document object type have meta data and page content


for i, doc in enumerate(docs): # doc : 
    print(i)
    print(doc.page_content)