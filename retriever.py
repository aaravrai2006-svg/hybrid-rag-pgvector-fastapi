import numpy as np
from typing import List, Dict, Any

class HybridRetriever:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.corpus: List[Dict[str, Any]] = []

    def reciprocal_rank_fusion(self, dense_ranks: List[str], sparse_ranks: List[str]) -> List[tuple[str, float]]:
        scores = {}
        for rank, doc_id in enumerate(dense_ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))
        for rank, doc_id in enumerate(sparse_ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    retriever = HybridRetriever()
    fused = retriever.reciprocal_rank_fusion(["doc1", "doc2", "doc3"], ["doc2", "doc4", "doc1"])
    print("Fused Ranking Scores:", fused)
