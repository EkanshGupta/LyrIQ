import faiss
import numpy as np
from annoy import AnnoyIndex
import os 

class getNN:
    def __init__(self, embeddings=None, faiss_path='faiss.index', annoy_path='annoy.index'):
        self.faiss_path = faiss_path
        self.annoy_path = annoy_path
        if embeddings is not None:
            embeddings = embeddings.astype('float32')
            self.embeddings = embeddings
            self.D = embeddings.shape[1]
            self.N = embeddings.shape[0]

            if os.path.exists(faiss_path):
                self.index_faiss = faiss.read_index(faiss_path)
            else:
                self.index_faiss = faiss.IndexFlatL2(self.D)
                self.index_faiss.add(embeddings)
                faiss.write_index(self.index_faiss, faiss_path)

            self.index_annoy = AnnoyIndex(self.D, 'euclidean')
            if os.path.exists(annoy_path):
                self.index_annoy.load(annoy_path)
            else:
                for i in range(self.N):
                    self.index_annoy.add_item(i, embeddings[i])
                self.index_annoy.build(n_trees=10)
                self.index_annoy.save(annoy_path)
        else:
            raise ValueError("Embeddings must be provided to build or load indices.")

    def getNN_faiss(self, query, k=5):
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        distances_faiss, indices_faiss = self.index_faiss.search(query, k)
        return indices_faiss[0]

    def getNN_annoy(self, query, k=5):
        return self.index_annoy.get_nns_by_vector(query, k)


        