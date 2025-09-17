# Semantic Search Engine

Semantic Search Engine is an intelligent PDF-based search tool that allows you to ask natural language questions and get semantically relevant answers from documents.  
It uses **Sentence Transformers** for embeddings and **FAISS** for vector similarity search.  

Didnt used LanagChain, Langchain had lot of abstractions that made this alot simpler and efficient. But I didnt wanted it to be simple. So, went the traditional way.

---
<img width="949" height="542" alt="Screenshot 2025-09-17 at 11 05 01 PM" src="https://github.com/user-attachments/assets/aa81bb90-07ce-4696-a380-c14a6548a2d9" />

## ✨ Features
- **PDF Reader** – Upload and parse text from PDF files.  
- **Text Chunking** – Splits large documents into smaller sentence chunks for better embedding and retrieval.  
- **Embeddings** – Uses `all-MiniLM-L6-v2` from Sentence Transformers to generate semantic embeddings.  
- **Vector Search** – Leverages **FAISS** to store and query embeddings efficiently.  
- **Question Answering** – Ask a question like *"What is the relevance of Blockchain?"* and retrieve the most relevant chunks.  

---

## ⚡ Example

```bash
python main.py
```
Sample output:

```bash
Total chunks: 42
2
[[0.23 0.45]] [[12 33]]
['Blockchain enables secure and transparent transactions ...'] distance: 0.23
['Distributed ledgers provide ...'] distance: 0.45
```
## 🛠️ How It Works
1. Extract text from PDFs using PyPDF2.
2. Tokenize sentences using NLTK.
3. Split into chunks (approx. 100 words each).
4. Encode sentences into embeddings using Sentence Transformers.
5. Build a FAISS index to store embeddings.
6. Search queries against the index to find relevant chunks.

##📦 Prerequisites
- Python 3.9+ (tested on 3.11 / 3.13)
- Virtual Environment (venv) recommended
Install dependencies:
```bash
pip install PyPDF2 nltk sentence-transformers faiss-cpu numpy 
```




