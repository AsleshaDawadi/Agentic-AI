from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import  SystemMessage , HumanMessage , AIMessage


load_dotenv()
llm= ChatGroq(model_name="llama-3.3-70B-versatile")

history=[]

history.append(SystemMessage(content="You are a converstational bot which is polite friendly respectful, and talanted a genius\
                            so every message you give has good polite and friendly tone")) 

while True:
    user_input=input("Type your message or type exit to end your converstation:\n")
    history.append(HumanMessage(content=user_input))
    if user_input.lower()=='exit':
        break
    response=llm.invoke(history)
    print("AI:",response.content)

    history.append(AIMessage(content=response))
