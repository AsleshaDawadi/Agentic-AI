# make a system where  user gives a topic and the output must comein the format of five points but it muct be in a json format 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate , SystemMessagePromptTemplate , HumanMessagePromptTemplate , AIMessagePromptTemplate
from typing import TypedDict

# yo hamile hamro structure ko fomrat bhaneko 
class Output_Structure(TypedDict):
    question1:str
    options1:list[str]
    question2:str
    option2:list[str]
    question3:str
    option3:list[str]
    question4:str
    option4:list[str]
    question5:str
    option5:list[str]


load_dotenv()
llm= ChatGroq(model_name="llama-3.3-70B-versatile")
# yesma hamile llm lai hamro structure ko barema bhaneko prompt ma nai sidaii lm ma 
structured_llm=llm.with_structured_output(Output_Structure)

# Blog Prompt Generator with proper format
# in this we give examples to check if the ysstem gives output based on our examples or not 
user_t= input("Enter the required Subject:")

chat_prompt= ChatPromptTemplate.from_messages([
   SystemMessagePromptTemplate.from_template("""
    You are a best teacher in the world of {topic} who generates quite the good questions with this following subject of {topic}\
    now you are to generate five MCQ questions with with four options for each questions respectively to select where one of them\
    should must be a correct answer of the respective questions  fir the options write a b c d respectively for each options
    """),
    HumanMessagePromptTemplate.from_template("Write me MCQ about the subject {topic}")

])

prompt=chat_prompt.format_messages(topic=user_t)


response=structured_llm.invoke(prompt)
print(response) 