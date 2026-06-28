from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate , SystemMessagePromptTemplate , HumanMessagePromptTemplate , AIMessagePromptTemplate
from typing import TypedDict

# yo hamile hamro structure ko fomrat bhaneko 
class Output_Structure(TypedDict):
    Title: str
    Content : str
    Keywords: list[str]
    Author_Name: str 


load_dotenv()
llm= ChatGroq(model_name="llama-3.3-70B-versatile")
# yesma hamile llm lai hamro structure ko barema bhaneko prompt ma nai sidaii lm ma 
structured_llm=llm.with_structured_output(Output_Structure)

# Blog Prompt Generator with proper format
# in this we give examples to check if the ysstem gives output based on our examples or not 
user_t= input("Enter Blog Topic:")

chat_prompt= ChatPromptTemplate.from_messages([
   SystemMessagePromptTemplate.from_template("""
    You are a professional Blog Writer. Help me write a professional and structured blog aout {topic}. /
    The blogs should be honest and factual with lots of depth but must not be boring and should reach a large number of people\
    This blog should contain high seo optimized tile keyword and detailed technical explanation . At the last add\
    my name as Aslesha Dawadi.

    """),
    HumanMessagePromptTemplate.from_template("Write me a blog about {topic}")

])

prompt=chat_prompt.format_messages(topic=user_t)


response=structured_llm.invoke(prompt)
print(response) 