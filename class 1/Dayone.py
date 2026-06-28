from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm=ChatGroq(model_name="llama-3.3-70B-versatile")
response=llm.invoke("""You are an expert in AI. You are someone can predict were ai goes. Now tell me which ways and how
should i use ai and ai tools as an computer science student wh is looking for jobs in this job market""")                       # invoke function does it all it  takes the api body it knows abou t the api host it also has the authorization   

answer = response

second_prompt = f"""
based on the generated answer provide me 5 point summary of answer  and represent all the developer 
informtaion in a clean clear cut format inside info header in json
Answer:{answer}
"""
response=llm.invoke(second_prompt)
print(response.content)

