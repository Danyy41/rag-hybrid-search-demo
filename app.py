from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

docs = open("docs.txt").read().split("\n")

tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs)

query = input("Question: ")

bm25_scores = bm25.get_scores(query.split())

query_embedding = model.encode([query])[0]
vector_scores = np.dot(embeddings, query_embedding)

hybrid_scores = bm25_scores + vector_scores
best_doc = docs[np.argmax(hybrid_scores)]

prompt = f"""
You are an internal documentation assistant.

Answer the user's question using only the context below.

Question:
{query}

Context:
{best_doc}

Answer clearly and mention that the answer is based on the internal document.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nRetrieved document:")
print(best_doc)

print("\nGenerated answer:")
print(response.choices[0].message.content)

