import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once
emodel = SentenceTransformer('./bge-large-en-v1.5')

def embed_texts(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = emodel.encode(batch, normalize_embeddings=True)
        embeddings.append(emb)
    return np.vstack(embeddings).tolist()
