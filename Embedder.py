# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:27:50 2024

@author: MANISH
"""

# embedder.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from nltk.tokenize import sent_tokenize

def embed_and_store(text_file, model_name='all-MiniLM-L6-v2'):
    with open(text_file, 'r') as file:
        content = file.read()
    
    chunks = sent_tokenize(content)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))
    
    with open('chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    
    faiss.write_index(index, 'faiss_index.bin')

if __name__ == "__main__":
    embed_and_store("luke_skywalker.txt")
