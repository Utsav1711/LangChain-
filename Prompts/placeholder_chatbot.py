# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# # Call chat Tamplate
# chat_tamplate = ChatPromptTemplate([
#     ('system', 'You are a helpful Customer Support Assistant.'),
#     MessagesPlaceholder(variable_name='chat_history'),
#     ('human', '{query}'),
# ])

# chat_history = []
# #Load Tamplate
# with open('chat_history.txt') as file:
#     chat_history.extend(file.readlines())

# print(chat_history)

# #Create 
# prompt = chat_tamplate.invoke({
#     'chat_history': chat_history, 
#     'query': 'Where is My Refund ?'
# })
# print(prompt)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Create chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Customer Support Assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])

# Load chat history
chat_history = []

with open("chat_history.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line.startswith("Human:"):
            chat_history.append(HumanMessage(content=line.replace("Human:", "").strip()))
        elif line.startswith("AI:"):
            chat_history.append(AIMessage(content=line.replace("AI:", "").strip()))

print("Chat History:")
print(chat_history)

# Invoke prompt
prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Where is my refund?"
})

print("\nFinal Prompt:")
print(prompt)
