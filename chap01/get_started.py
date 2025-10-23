# %% Import libraries
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage

# %% Instance model
model = OllamaLLM(model="gemma3:latest", temperature=0.2)

# %% Invoke the model
model.invoke("The sky is")

# %% Create messages
system_message = SystemMessage(content="You are a helpful assistant that provides concise answers max 10 chars")

# %% Invoke the model with messages
model.invoke([system_message, HumanMessage(content="The sky is")])


