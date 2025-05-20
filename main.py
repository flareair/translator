from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

if __name__ != "__main__":
    raise Exception("Not run as main file!")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.2")

chain = prompt | model

result = chain.invoke({"question": "What is LangChain?"})

print(result)