# %%
collection_name = "chap_03"
connection_string = "http://localhost:6333"

# %%
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embedding")
# Define and load an embedding model, nomic it's optmized for italian
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# %%
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

print("Loading vectors")
client = QdrantClient(
    url=connection_string,
)

# Retrieve embedding
qdrant = QdrantVectorStore(
    client=client,
    embedding=embeddings_model,
    collection_name=collection_name,

)

print("Prepare retriever")
# %%
# Get retrieved
retriever = qdrant.as_retriever()

# %%
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

print("Prepare prompt")
prompt = ChatPromptTemplate([
    ("system", """
     You are a helpful assistant answer the question based only on the following context: {context}.
     Answer in Italian. Always add in the anser the bible chapter and verses.
     Useful informations:
     1 cubit = 45,72 cm
     """),
    ("user", "Question: {question}")
])

# %%
# Import model
llm = OllamaLLM(model='gemma3')

# %%
from langchain_core.runnables import chain

print("Create chain")
@chain
def qa(question):
    # Get relevant docs
    retrieved_docs = retriever.invoke(question)
    formatted = prompt.invoke({"context":retrieved_docs, "question":question})
    return llm.invoke(formatted)



