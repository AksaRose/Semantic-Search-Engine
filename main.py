from sentence_transformers import SentenceTransformer
import numpy as np
import faiss     
k = 2
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

q =["How is the whether today?"] 
qemb = model.encode(q)
query = np.array(qemb)
# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)

emb = np.array(embeddings)
index = faiss.IndexFlatL2(embeddings.shape[1]) 
index.add(emb) 
print(index.ntotal)

dist,ind = index.search(query,k)
print(dist, ind)
indices = list(ind[0])
distance = list(dist[0])
for i in indices:
    print(f'{sentences[i]} distance: {distance[i]}')

