from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001", dimensions = 768
)
# By default, it outputs a 3072-dimensional embedding, but you can truncate it to a smaller size without losing quality to save storage space. We recommend using 768, 1536, or 3072 output dimensions.

documents = [
"Sachin Tendulkar is a legendary Indian cricketer known as the “God of Cricket” for his unmatched consistency and records",

"Lionel Messi is an Argentine footballer famous for his exceptional dribbling, vision, and goal-scoring ability",

"Serena Williams is an American tennis player who dominated women’s tennis with her power and mental strength",

"P. V. Sindhu is an Indian badminton star and Olympic medalist known for her aggressive playing style",

"Usain Bolt is a Jamaican sprinter widely regarded as the fastest man in the world"
]

query = "Provide some information about Usain Bolt"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarity_score = cosine_similarity([query_embedding],doc_embedding)[0] # need 2-D list
# similarity_score prints 2-d list, so select [0]
# print(similarity_score)

# enumerate is useful for obtaining an indexed list.
# Ex : [(0, score_value_1),(1,score_value_2),(2,score_value_3)]
# we need enumerate because we want to maintain that which vector associated with which sentence in documents, so, before sorting we are enumerating.
# And creating index helps in accessing most matched sentence.
index, score = sorted(list(enumerate(similarity_score)),key = lambda x:x[1])[-1]
# by deafult sorted in ascending order

print(documents[index])
