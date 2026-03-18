# To create Dynamic Templates Multi-Turn Conversation 
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', 'Explain in simple term , what is {topic}'),
])

prompt = chat_template.invoke({
    'domain': 'science',
    'topic': 'quantum computing'
})

print(prompt) 