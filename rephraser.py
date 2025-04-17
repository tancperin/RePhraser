from sentence_transformers import SentenceTransformer, util
import torch

def __cosine_similarity(sentences):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    return cosine_matrix
