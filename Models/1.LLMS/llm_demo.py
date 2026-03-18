from langchain_openai import OpenAI
import dotenv

dotenv.load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)

response = llm.invoke("Explain the theory of relativity in simple terms.")
print(response)