# %% Import libraries
from langchain_ollama import ChatOllama
from pydantic import BaseModel

# %% Instance model
model = ChatOllama(model="gemma3:latest")

# %% Define structured output
class CityInfo(BaseModel):
    '''Full answer for the user'''
    answer: str
    '''Province where the city is located'''
    province: str
    '''Name of the city'''
    city: str
    '''City inhabitants'''
    inhabitants: int

# %% Create structured output version of the model
structured_response = model.with_structured_output(CityInfo)

# %% Invoke the model
structured_response.invoke('Where is Adrano?')


