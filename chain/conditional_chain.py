from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment : Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

parser2 =PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
    template="Write a appropriate responce to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write a appropriate responce to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive',prompt2 | model1 | parser),
    (lambda x : x.sentiment == 'negative',prompt3 | model1 | parser),
    RunnableLambda(lambda x:"Could not be find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback" : "This is the best movie of govinda"})

print(result)

chain.get_graph().print_ascii() 