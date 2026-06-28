from langchain_groq import ChatGroq
from dotenv import load_dotenv
# tools
from langchain_core.tools import tool
from langchain.messages import ToolMessage

load_dotenv()
llm=ChatGroq(model_name="llama-3.3-70B-versatile")

messages=[]
# since these tools that we defined the llm have no idea about what these tools do now to
# let them know we have to mention and describe each and every function and what it do so we use comments to do it in every function we have defined we write inside "_"
@tool
def add(a: float,b:float)-> float: 
    "this function add two values"
    return a+b

@tool
def sub(a: float,b:float)-> float:
    """
    This function is used to substract two values
    """
    return a-b

@tool
def mul(a: float,b:float)-> float: 
    " This function is used for multiplying two values"
    return a*b

@tool
def div(a: float,b:float)-> float: 
    "This function is used for division "
    return a/b

@tool
def mod(a: float,b:float)-> float: 
    "this function is used for finding remainder or modulus"
    return a%b

tools=[add, sub, mul, div , mod]

llm_with_tools=llm.bind_tools(tools)

query=""" I have two red balls and three black balls in a basket .
        then i removed three balls from the basket. then i added twice the amount of balls i had left in the basket
         . How many balls do i have in total ?
     """

messages.append(("human", query))
response=llm_with_tools.invoke(messages)
messages.append(response)

for tool_call in response.tool_calls:
    selected_tool={
        "add":add,
        "sub":sub,
        "mul":mul,
        "div":div,
        "mod":mod
    }[tool_call['name']]

    tool_result=selected_tool.invoke(tool_call['args'])
    messages.append(
        ToolMessage(content=str(tool_result),
        tool_call_id=tool_call['id'])
        )

final_res=llm_with_tools.invoke(messages)
print(messages)
print("\n\n")
print(final_res)
