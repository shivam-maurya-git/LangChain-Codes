from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#By default, the length of the embedding vector is 1536 for text-embedding-3-small or 3072 for text-embedding-3-large
# length : dimensions
#more dimensions : more capturing of deatils or context
embedding = OpenAIEmbeddings(model='text-embedding-3-small',dimensions=32)

result = embedding.embed_query("I am a boy.")

print(result)