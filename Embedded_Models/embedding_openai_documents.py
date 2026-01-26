from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#By default, the length of the embedding vector is 1536 for text-embedding-3-small or 3072 for text-embedding-3-large
# length : dimensions
#more dimensions : more capturing of deatils or context
embedding = OpenAIEmbeddings(model='text-embedding-3-small',dimensions=32)

documents = [ "Virat Kohli has 85 centuaries",
             "Julian Assanage is activist",
             "Paris is in Europe"


]
result = embedding.embed_documents(documents)

print(result)
# Will print list of 3 vectors and each vector will have 32 values.