# %%
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import OllamaLLM

# %%
template = PromptTemplate.from_template("Rispondi alla domanda in maniera semplice e coincisa massimo 10 caratteri: {question}")

# %%
prompt = template.invoke({"question": "Colore del cielo"})

# %%
model = OllamaLLM(model="gemma3:latest")

# %%
model.invoke(prompt)

# %%
template = ChatPromptTemplate.from_messages([
    ("system", "You're an helpful assistant that answers in coincise way (max 10 characters)."),
    ("user", "{question}"),
])

# %%
model.invoke(template.invoke({
    "question": "What's the sky color"
}))


