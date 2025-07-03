from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
text = "HELLO Atchaya"
embedding = model.encode(text)
print(embedding)  # This is a NumPy array of shape (384,)
print(f"Embedding shape: {embedding.shape}")  # Should print (384,)
print(f"Embedding type: {type(embedding)}")  # Should print <class '
print(len(embedding))  # Should print 384, the dimensionality of the embedding