from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm=ChatGroq(model_name="llama-3.3-70B-versatile")

user_t=input("enter you required topic:\n")

prompt1=PromptTemplate(
    template="""
    You are an expert in {topic}.And has multiple years of experience and reserch in the {topic}. now write me a well defined \
    and good blog on the {topic}? 
    """,
    input_variables=["topic"])

#generation of score based on the quality of the post generated (0-10)

prompt2=PromptTemplate(
    template="""
    you are an expert senior in the blog and articles review you can say which ones will be read by many and which will \
    not so review this post.generate a score from (0-10) 0 being the least and 10 is the most post:{text}
"""
)

my_chain= prompt1 | llm  | prompt2 | llm            #langchain expression language

response= my_chain.invoke({"topic":user_t})

print(response)