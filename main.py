from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
nltk.download("punkt_tab")
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss     
k = 2
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

reader = PdfReader("Artificial.pdf")
number_of_pages = len(reader.pages)
text = ""
for i in range(number_of_pages):
    page = reader.pages[i]
    text += page.extract_text()

sentences = nltk.sent_tokenize(text)
current_chunk,chunks,count =[],[],0
for sentence in sentences:
    count += len(sentence.split())
    current_chunk.append(sentence)
    if count>100:
        chunks.append(current_chunk)
        current_chunk,count=[],0

print(f"Total chunks: {len(chunks)}")

q =["What is the relevance of Blockchain?"] 
qemb = model.encode(q)
query = np.array(qemb)
# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(chunks)

emb = np.array(embeddings)
index = faiss.IndexFlatL2(embeddings.shape[1]) 
index.add(emb) 
print(index.ntotal)

dist,ind = index.search(query,k)
print(dist, ind)
indices = list(ind[0])
distance = list(dist[0])
for i in range(k):
    print(f'{chunks[indices[i]]} distance: {distance[i]}')

