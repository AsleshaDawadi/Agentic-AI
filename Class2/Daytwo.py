from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()
llm=ChatGroq(model_name="llama-3.3-70B-versatile")

static_prompt=PromptTemplate(
    template="""
    You are an expert in {topic}. You are someone can predict were {topic} goes.
    Now tell me which ways and how the world is moving twords {topic}. 
    Now as someone who is moving towords {topic} what {chapter} should i learn ? 
    """,
    input_variables=["topic", "chapter"]
)
user_topics=input("Enter your wanted topic:")
user_chapters=input("Enter your wanted chapter to learn:")
                
prompt=static_prompt.format(topic=user_topics , chapter=user_chapters )

response=llm.invoke(prompt)
print(response)                                 


