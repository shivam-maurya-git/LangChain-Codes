from langchain_huggingface import HuggingFaceEmbeddings

# need to install : pip install sentence-transformers
# before using
embedding = HuggingFaceEmbeddings(model_name = 
                                  "sentence-transformers/all-MiniLM-L6-v2"
                                  )
text = "Virat Kohli is captain of Indian team"

vector = embedding.embed_query(text)
print(vector)

# we can use embed_documents also.