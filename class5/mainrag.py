from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm= ChatGroq(model_name="llama-3.3-70B-versatile")

# step1: loading document this helps us load the document 
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader("fi.pdf")
document=loader.load()

# step 2 : we chunk ( breaking down the text into various chunks) this recursive character text splitters it counts th enumber of words nd then spilts
# character text splitters cuts right down the defined  number of words or characters
from langchain_text_splitters import RecursiveCharacterTextSplitter 
# here chunk size defines the chunk size and the chunk overlap is used for overlapping the words so that the chunks have relation for linking up
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap =100
)
docs=text_splitter.split_documents(document)

#Step 3 : Embedding Generatioon
from langchain_huggingface import HuggingFaceEmbeddings
model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#step 4: Vector Store
# we need a storage foe this so we have various different storeages options we use Chroma
from langchain_chroma import Chroma
vector_store=Chroma.from_documents(
    documents=docs,
    embedding=model,
    collection_name="embedding_records_for_mainrag"
)

#step 5: Retriver 
# this is al our stragey on what basis and how many results to get this stragey on the basis of user query 
retriever=vector_store.as_retriever(search_kwargs={"k":5})

# step6: prompt defination here we also have defined the gurdalllias so that the model donet hullanicate 
from langchain_core.prompts import PromptTemplate
prompt=PromptTemplate(
    template="""
        You are an expert at retrival . use the following context to answer the user query. \
        If the context is not present or not sufficient to answer the query just say I don't know.
        context={context}
        query={query}
        """,
        imput_variables=['context','query']
)
query=input("Enter Your query")

relevant_chunks=retriever.invoke(query)
context="\n\n".join(doc.page_content for doc in relevant_chunks)
chain = prompt|llm
response=chain.invoke({
    "context":context,
    "query":query
})

print(response)