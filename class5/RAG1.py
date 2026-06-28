#RAG 
# here we will go on about the embedding

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

do1="Machine learning is an approach that is used by computers to learn data."
do2="Football is a sport played all around the world."
query="what kind of games are people used to ?"

doc1_embedding=model.embed_query(do1)
doc2_embedding=model.embed_query(do2)
query_embedding=model.embed_query(query)



score1=cosine_similarity([query_embedding],[doc1_embedding])[0][0]
score2=cosine_similarity([query_embedding],[doc2_embedding])[0][0]

print("Score1:",score1)
print("Score2:",score2)
# if similaroty score is one then both the vectors are exactly similar 
#if the similarity score is -1 the nthey are completely opposite
