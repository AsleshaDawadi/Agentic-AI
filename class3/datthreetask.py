from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate , SystemMessagePromptTemplate , HumanMessagePromptTemplate , AIMessagePromptTemplate
from typing import TypedDict
import json

class Output_Structure(TypedDict):
    topic:str
    content:str
    keywords:list[str]
    Sentiments:str

class Outputs(TypedDict):
    point1:str
    point2:str
    point3:str
    point4:str
    point5:str

load_dotenv()
llm= ChatGroq(model_name="llama-3.3-70B-versatile")

structured_llm=llm.with_structured_output(Output_Structure)

with open("class3/article.txt", "r", encoding="utf-8") as file:
    article = file.read()


prompt1= ChatPromptTemplate.from_messages([
   SystemMessagePromptTemplate.from_template("""
    You are a best article critiser in the world if the article is bad but you review the points of the article very nicely with \
    now you have to review this article and point out its topic or the title of the article and then the content of the artcile properly \
    then identify the important and useful keywords in this article and give the output in this fomrat. 
    """),
    HumanMessagePromptTemplate.from_template("Review this {topic}")

])

prompt=prompt1.format_messages(topic=article)
response=structured_llm.invoke(prompt)

with open("info.txt", "w", encoding="utf-8") as file:
    json.dump(response, file, indent=4)



#Task two

structured_llm2=llm.with_structured_output(Outputs)

prompt2= ChatPromptTemplate.from_messages([
   SystemMessagePromptTemplate.from_template(""" you are an best summarizer so now you have to 
    summarize an article in five points such as point1, point2, point3, point4, point5 . where article = :{topic}
    """),
    HumanMessagePromptTemplate.from_template("Summarize this {topic}")

])

prompt=prompt2.format_messages(topic=article)
response2=structured_llm2.invoke(prompt)

with open("summary.txt", "w", encoding="utf-8") as file:
    json.dump(response2, file, indent=4)


prompt3=ChatPromptTemplate.from_messages([
   SystemMessagePromptTemplate.from_template("""
    You are a best article critiser in the world if the article is bad but you review the points of the article
    you can easily find out the sentimant of the article weathe rit has positve or the neagtive then score the sentiment of the article \
    to be postive or negative in the range of 0 to 10 with 0 being th eleast positive and 10 bieng th emost positive
    """),
    HumanMessagePromptTemplate.from_template("Review this {topic}")
])

sentiment_prompt = prompt3.format_messages(topic=article)
response3 = llm.invoke(sentiment_prompt)
print(response3.content)

