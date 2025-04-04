from sentence_transformers import SentenceTransformer
import numpy as np
np.set_printoptions(suppress=True, precision=5)
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer("BAAI/BGE-Base-EN-v1.5", device='cuda')

# MAKE THEM
def make_ze_numbers(stuff):
    sentences = [stuff]
    embedding = model.encode(sentences)
    return embedding[0]

e1 = make_ze_numbers("what kind of laptops is randerzander interested in?")
#e2 = make_ze_numbers("Yeah but most windows laptops are kinda shit...")
e2 = make_ze_numbers("Yeah but most windows laptops are kind of shitty 😄")

cosine_similarity = np.dot(e1, e2)
print(cosine_similarity)
