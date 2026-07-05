from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm= ChatGroq(model_name="llama-3.3-70B-versatile")

# load the youtube url and extract th econtnents from the videos
from langchain_community.document_loaders import YoutubeLoader
loader=YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=kqtD5dpn9C8")
document=loader.load()

#chunking 
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks=text_splitter.split_documents(document)

# Embedding Generation
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector Store
from langchain_chroma import Chroma
vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=model,
    collection_name="embedding_records_for_youtube"
)
# storage
chat_history_Archive=vector_store.persist()

#retriver
retriever=vector_store.as_retriever(search_kwargs={"k":5})

#prompt defination
from langchain_core.prompts import PromptTemplate
prompt=PromptTemplate(
    template="""
        You are an expert AI assistant specialized in analyzing video transcripts.
        Use ONLY the following pieces of retrieved context to answer the user's query at the end.
        Guideline:
        1. must answer strictyly on the provided context
        2. if the context does not contain the answer, respond with "I don't know."
        3. dont make up any facts or information that is not present in the context.
        context={context}
        query={query}
        """,
        input_variables=['context','query']
)
query=input("Enter Your query:")

relevant_chunks=retriever.invoke(query)
context="\n\n".join(doc.page_content for doc in relevant_chunks)
chain = prompt|llm
response=chain.invoke({
    "context":context,
    "query":query
})

print(response.content)  
#readeravskh@gmail.com