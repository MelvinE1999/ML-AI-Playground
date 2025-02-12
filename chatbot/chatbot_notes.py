# code pulled from this video: https://www.youtube.com/watch?v=d0o89z134CQ
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# template is provided to the model and helps improve how the model will respond to queries
template = """
Answer the Following.

Here is the conversation history: {context}

Question: {question}

Answer: 
"""

model = OllamaLLM(model="llama3") # specification of model that will handle the query
prompt = ChatPromptTemplate.from_template(template) # takes the template made above and injects it to the model prompt every query
chain = prompt | model # uses langchain and chains the prompt and model


def handle_conversation():
    context = "" 
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ") # gains user query
        if user_input.lower() == "exit":
            break

        result = chain.invoke({"context": context, "question": user_input}) # invokes the selected model and gets back the response of the model
        print("Bot", result)
        context += f"\nUser: {user_input}\nAI: {result}" # adds chat history to allow references to past convos


if __name__ == "__main__":
    handle_conversation()