from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

docs = open("docs.txt").read().split("\n")

tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(docs)

query = input("Question: ")

bm25_scores = bm25.get_scores(
    query.split()
)

query_embedding = model.encode([query])[0]

vector_scores = np.dot(
    embeddings,
    query_embedding
)

hybrid_scores = bm25_scores + vector_scores

best_doc = docs[np.argmax(hybrid_scores)]

print("\nAnswer:")
print(best_doc)
