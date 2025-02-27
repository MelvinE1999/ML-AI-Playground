from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate



step_by_step_template = """
You are an AI assisstant with expertise in organizing tasks based on task prioritation.
You will be provided a list of tasks with the goal of organizing the list based on priority.
To create this list use a combination of the provided context and these steps. Make sure to follow each and every step in order not skipping one.

Context:
{context}

Steps:
1. Respond back and ask them this exact question "Are there any task(s) you'd like to give the highest priority today?" and wait for the response If and only if the question is not in the context
2. Answer these questions giving points to each task as provided and only as provided going through each task keeping track of the scores so as to be used in later steps:
    * How much time will it take to accomplish this task based on your best judgement? (The more time the more points. With the range being 1-10 points)
    * Is the task an essential task? If so give it 5 points and if not then give it 1 point.
    * Does this task affect others? If so give it 5 points and if not then give it 1 point.
    * Would you consider the task urgent or not? If so give it 15 points and if not then give it 1 point.
3. Double the points acquired by all task(s) that were said in response to "Are there any task(s) you'd like to give the highest priority today?" context.
4. Now rank them based purely on the points they earned. (Ordering from most to least)
5. Respond back with only the list that you have created in step 4 (exclude the points) following this format:
    1. Task 1
    2. Task 2
    ...
    n-1. Task n-1
    n. Task n

Question: {question}

Answer: 
"""

template = """
You are an AI assistant whos goal will be to craft a balanced to-do list that will be achieveable within a 24 hour time window. 
Based on the tasks given to you suggest a balanced to-do list that balances each activity effectively, taking into consideration any priorities or constraints given if applicable.
Make sure to number every task in sequential order.
On this list only include tasks that were given only.
When you return back your list only give the list.

Here is the conversational history: {context}

Question: {question}

Answer: 

"""

model = OllamaLLM(model="llama3.2") 
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def handle_query():
    context = "" 
    print("Welcome to the AI Agent! I'm here to help you organize your tasks for today.")
    while True:
        user_input = input("Type 'exit' to quit.\nYou: ")
        if user_input.lower() == "exit":
            break

        result = chain.invoke({"context": context, "question": user_input})
        print(result[0])
        print("Bot: ", result)
        context += f"\nUser: {user_input}\nAI: {result}" 


if __name__ == "__main__":
    handle_query()