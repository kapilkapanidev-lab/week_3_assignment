#!/usr/bin/env python3
"""
Retrieval Pipeline for SIGGRAPH 2025 Papers.

Implements hybrid search:
1. Semantic search (embeddings via OpenRouter + Qdrant Cloud)
2. Keyword search (BM25 - runs locally)
3. Reranking (Cohere API - optional)

Usage:
    from retrieval_pipeline import RetrievalPipeline
    
    pipeline = RetrievalPipeline()
    results = pipeline.retrieve("3D Gaussian Splatting", top_k=5)
"""

import json
import os
import re
import requests
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
load_dotenv()

# Must match the collection name used in upload_to_qdrant.py
COLLECTION_NAME = "siggraph2025_papers"


@dataclass
class RetrievalResult:
    """
    Represents a single search result.
    The api_server.py expects these exact fields - do not change!
    """
    chunk_id: str
    paper_id: str
    title: str
    authors: str
    text: str
    score: float
    chunk_type: str = ""
    chunk_section: str = ""
    pdf_url: Optional[str] = None
    github_link: Optional[str] = None
    video_link: Optional[str] = None
    acm_url: Optional[str] = None
    abstract_url: Optional[str] = None


@dataclass
class RetrievalPipelineConfig:
    """Configuration for the retrieval pipeline."""
    qdrant_url: str
    qdrant_api_key: str
    openrouter_api_key: str
    embedding_model: str = "baai/bge-large-en-v1.5"
    chunks_path: str = "./chunks.json"
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    use_reranker: bool = True
    cohere_api_key: Optional[str] = None


class OpenRouterEmbedder:
    """
    Generate embeddings using OpenRouter API.
    Used to embed user queries for semantic search.
    """
    
    def __init__(self, api_key: str, model: str = "baai/bge-large-en-v1.5"):
        """
        Initialize the embedder.
        """

        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        TODO:
        1. Build headers dict:
           - "Authorization": f"Bearer {self.api_key}"
           - "Content-Type": "application/json"

        """
        headers = {
          "Authorization": f"Bearer {self.api_key}",
          "Content-Type": "application/json"
        }

        """
        
        2. Build payload dict:
           - "model": self.model
           - "input": text

        """
        payload = {
            "model": self.model,
            "input": text
        }

        """
        
        3. Make POST request to f"{self.base_url}/embeddings"

        """

        response = requests.post(f"{self.base_url}/embeddings", headers=headers, json=payload)

        """
        
        4. Check response status code, raise error if not 200

        """

        if response.status_code != 200:
          raise ValueError(f"OpenRouter API error: {response.status_code} - {response.text}")

        """
        
        5. Parse response JSON


        """

        """
        
        6. Extract embedding: embedding = response_data["data"][0]["embedding"]

        """
        response_data = response.json()
        embedding = response_data["data"][0]["embedding"]
        
        """
        7. Convert to numpy array and return:
           return np.array(embedding, dtype=np.float32)
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as numpy array
        """


        return np.array(embedding, dtype=np.float32)


class BM25Index:
    """
    BM25 index for keyword search.
    This runs entirely locally - no API calls needed!
    BM25 is good at finding exact keyword matches that semantic search might miss.
    """
    
    def __init__(self, chunks: list[dict]):
        """
        Build BM25 index from chunks.
        
        TODO:
        1. Store the chunks: self.chunks = chunks

        """
        self.chunks = chunks
        
        """
        2. Create a lookup dict for quick access by chunk_id:
           self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}

           """
        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}

        """
        
        3. Tokenize all documents (convert each chunk's text to list of words):
           self.tokenized_docs = [self._tokenize(c["text"]) for c in chunks]

        """
        self.tokenized_docs = [self._tokenize(c["text"]) for c in chunks]
        
        """
        
        4. Create the BM25 index:
           self.bm25 = BM25Okapi(self.tokenized_docs)

        """

        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization: lowercase and extract alphanumeric words.
        
        TODO:
        1. Convert text to lowercase
        """
        text = text.lower()
        
        """

        2. Use regex to find all words (alphanumeric sequences)

        """

        tokens = re.findall(r'\w+', text)
        

        """
        
        3. Return the list of tokens

        """
        return tokens
    
    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """
        Search for query and return top-k results.
        
        TODO:
        1. Tokenize the query

        """
        
        tokenized_query =  self._tokenize(query)

        """
        
        2. Get BM25 scores for all documents

        """

        bm25_scores = self.bm25.get_scores(tokenized_query)

        """
         
        3. Get indices of top-k highest scores (hint: use np.argsort)

        """
        top_k_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        """

        
        4. Build result list with only non-zero scores (hint: use list comprehension)
           
        """
        results = [(idx, bm25_scores[idx]) for idx in top_k_indices if bm25_scores[idx] > 0]
        
        """
        
        5. Return results

        """

        return results
        
        """
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of (chunk_index, score) tuples, sorted by score descending
        """


class RetrievalPipeline:
    """
    Main retrieval pipeline combining semantic search + BM25 + reranking.
    This is what api_server.py uses to find relevant chunks.
    """
    
    def __init__(self, config: Optional[RetrievalPipelineConfig] = None):
        """
        Initialize all components of the retrieval pipeline.
        
        TODO:
        1. If config is None, create one from environment variables:
           config = RetrievalPipelineConfig(
               qdrant_url=os.getenv("QDRANT_URL"),
               qdrant_api_key=os.getenv("QDRANT_API_KEY"),
               openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
               cohere_api_key=os.getenv("COHERE_API_KEY"),
               chunks_path=os.getenv("CHUNKS_PATH", "./chunks.json"),
           )

           """
        if config is None:
            config = RetrievalPipelineConfig(
                qdrant_url=os.getenv("QDRANT_URL"),
                qdrant_api_key=os.getenv("QDRANT_API_KEY"),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                chunks_path=os.getenv("CHUNKS_PATH", "./chunks.json"),
            )

            """
      
        2. Validate required fields - raise ValueError if missing:
           - config.qdrant_url
           - config.qdrant_api_key
           - config.openrouter_api_key

           """

        if not config.qdrant_url or not config.qdrant_api_key or not config.openrouter_api_key:
            raise ValueError("Missing required configuration for RetrievalPipeline (check your .env file)")

        
        """


        3. Initialize Qdrant client:

        """
        self.qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)

        """  
        4. Initialize the embedder:
        """
        self.embedder = OpenRouterEmbedder(api_key=config.openrouter_api_key, model=config.embedding_model)

        """  

        5. Load chunks from JSON file:

        """

        with open(config.chunks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.chunks = data["chunks"]

        """       
        
        6. Build BM25 index:
           self.bm25_index = BM25Index(self.chunks)
           print("BM25 index built")

        """

        self.bm25_index = BM25Index(self.chunks)
        print("BM25 index built")

        """
        
        7. Store the config:
           self.config = config
        
        Args:
            config: Optional configuration. If None, loads from environment variables.
        """
        self.config = config
        
    
    def semantic_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform semantic search using Qdrant.
        
        TODO:
        1. Embed the query:
           query_embedding = self.embedder.embed_query(query)

        """
        query_embedding_vector = self.embedder.embed_query(query)

        """
        
        2. Search Qdrant:
           results = self.qdrant.query_points(
               collection_name=COLLECTION_NAME,
               query=query_embedding.tolist(),
               limit=top_k,
               with_payload=True
           ).points

        """
        results = self.qdrant.query_points(
          collection_name="siggraph2025_papers", 
          query=query_embedding_vector.tolist(), 
          limit=top_k, with_payload=True
          ).points

        """

       
        
        3. Convert to list of dicts:
           return [
               {
                   "chunk_id": r.payload["chunk_id"],
                   "score": r.score,
                   "payload": r.payload
               }
               for r in results
           ]
        
        Args:
            query: Search query
            top_k: Number of results to return


        """
        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "score": r.score,
                "payload": r.payload
            }
            for r in results
        ]
    
    def bm25_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform BM25 keyword search.
        
        TODO:
        1. Call BM25 search:
           results = self.bm25_index.search(query, top_k)
        
        """

        results = self.bm25_index.search(query,top_k)



        """
        2. Convert to list of dicts (same format as semantic_search):
           return [
               {
                   "chunk_id": self.chunks[idx]["chunk_id"],
                   "score": score,
                   "payload": self.chunks[idx]
               }
               for idx, score in results
           ]
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        """

        return [
               {
                   "chunk_id": self.chunks[idx]["chunk_id"],
                   "score": score,
                   "payload": self.chunks[idx]
               }
               for idx, score in results
           ]        
    
    def hybrid_search(self, query: str, semantic_top_k: int = 30, bm25_top_k: int = 30) -> list[dict]:
        """
        Combine semantic and BM25 results using weighted scoring.
        
        TODO:
        1. Get results from both search methods:
           semantic_results = self.semantic_search(query, semantic_top_k)
           bm25_results = self.bm25_search(query, bm25_top_k)
        
        """
        semantic_results = self.semantic_search(query, semantic_top_k)
        bm25_results = self.bm25_search(query)

        """
        2. Normalize semantic scores (divide by max score):
           if semantic_results:
               max_semantic = max(r["score"] for r in semantic_results)
               for r in semantic_results:
                   r["normalized_score"] = r["score"] / max_semantic if max_semantic > 0 else 0
        
        """

        if semantic_results:
            max_semantic = max(r["score"] for r in semantic_results)
            for r in semantic_results:
                r["normalized_score"] = r["score"] / max_semantic if max_semantic > 0 else 0

        """
        
        3. Normalize BM25 scores the same way


        """
        
        if bm25_results:
            max_bm25 = max(r["score"] for r in bm25_results)
            for r in bm25_results:
                r["normalized_score"] = r["score"] / max_bm25 if max_bm25 > 0 else 0

        """
        
        4. Combine results into a single dict keyed by chunk_id:
           combined = {}
           
           For semantic results:
           - Add to combined with semantic_score and initial combined_score
           
           For BM25 results:
           - If chunk_id already in combined, add bm25_score and update combined_score
           - If new, add with just bm25_score
           
           Combined score formula:
           combined_score = (semantic_weight * semantic_score) + (bm25_weight * bm25_score)

        """

        for r in semantic_results:
            cid = r["chunk_id"]
            combined[cid] = {
            "chunk_id": cid,
            "payload": r["payload"],
            "semantic_score": r["normalized_score"],
            "bm25_score": 0.0,
            "combined_score": r["normalized_score"] * self.config.semantic_weight
        }
        for r in bm25_results:
            cid = r["chunk_id"]
            if cid in combined:
                # Already found in semantic search? Add the extra points!
                combined[cid]["bm25_score"] = r["normalized_score"]
                combined[cid]["combined_score"] += r["normalized_score"] * self.config.bm25_weight
            else:
                # New result found only by keywords
                combined[cid] = {
            "chunk_id": cid,
            "payload": r["payload"],
            "semantic_score": 0.0,
            "bm25_score": r["normalized_score"],
            "combined_score": r["normalized_score"] * self.config.bm25_weight
        }

        """
        
        5. Sort by combined_score descending:
           results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)

        """

        results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)

        """

        
        6. Return the sorted list
        
        Args:
            query: Search query
            semantic_top_k: Max results from semantic search
            bm25_top_k: Max results from BM25 search
            
        Returns:
            Combined and sorted list of results
        """
        return results



    
    def rerank(self, query: str, results: list[dict], top_k: int = 10) -> list[dict]:
        """
        Rerank results using Cohere API (optional but improves quality).
        
        TODO:
        1. If no Cohere API key or no results, return results[:top_k]
        
        2. Extract texts for reranking:
           texts = [r["payload"]["text"] for r in results]
        
        3. Call Cohere Rerank API:
           - URL: "https://api.cohere.ai/v1/rerank"
           - Headers: {"Authorization": f"Bearer {self.config.cohere_api_key}"}
           - Body: {
               "model": "rerank-english-v3.0",
               "query": query,
               "documents": texts,
               "top_n": top_k
           }
        
        4. Parse response and reorder results based on Cohere's ranking:
           - response.results contains items with index and relevance_score
           - Reorder your results list to match Cohere's order
           - Update each result's score with the rerank_score
        
        5. Return reranked results
        
        Note: If Cohere API fails, catch the error and fall back to returning results[:top_k]
        
        Args:
            query: Original query
            results: Results from hybrid_search
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked list of results
        """
        pass
    
    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        """
        Full retrieval pipeline - THIS IS WHAT api_server.py CALLS!
        
        TODO:
        1. Run hybrid search to get candidates:
           candidates = self.hybrid_search(query)
        

        """
        candidates = self.hybrid.search(query)
        """



        2. Rerank if enabled:
           if self.config.use_reranker:
               reranked = self.rerank(query, candidates, top_k=min(top_k * 2, len(candidates)))
           else:
               reranked = candidates


        """
        if self.config.use_reranker:
            reranked = self.rerank(query, candidates, top_k=min(top_k * 2, len(candidates)))
        else:
            reranked = candidates

        """
        
        3. Take top_k results:
           final = reranked[:top_k]

        """
        final = reranked[:top_k]
        """    
        
        4. Convert to RetrievalResult objects:
           return [
               RetrievalResult(
                   chunk_id=r["payload"]["chunk_id"],
                   paper_id=r["payload"]["paper_id"],
                   title=r["payload"]["title"],
                   authors=r["payload"]["authors"],
                   text=r["payload"]["text"],
                   score=r.get("rerank_score", r.get("combined_score", r.get("score", 0))),
                   chunk_type=r["payload"].get("chunk_type", ""),
                   chunk_section=r["payload"].get("chunk_section", ""),
                   pdf_url=r["payload"].get("pdf_url"),
                   github_link=r["payload"].get("github_link"),
                   video_link=r["payload"].get("video_link"),
                   acm_url=r["payload"].get("acm_url"),
                   abstract_url=r["payload"].get("abstract_url"),
               )
               for r in final
           ]
        
        Args:
            query: User's search query
            top_k: Number of results to return


            
        Returns:
            List of RetrievalResult objects ready for RAG generation
        """
        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                paper_id=r["payload"]["paper_id"],
                title=r["payload"]["title"],
                authors=r["payload"]["authors"],
                text=r["payload"]["text"],
                score=r.get("combined_score", r.get("score", 0.0)),
                acm_url=r["payload"].get("acm_url"),
                abstract_url=r["payload"].get("abstract_url")
            )
            for r in final
        ]


# For testing this file directly
if __name__ == "__main__":
    import sys
    
    query = sys.argv[1] if len(sys.argv) > 1 else "3D Gaussian Splatting"
    
    print(f"Testing retrieval pipeline with query: '{query}'")
    print("=" * 60)
    
    pipeline = RetrievalPipeline()
    results = pipeline.retrieve(query, top_k=5)
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.4f}] {r.title[:60]}...")
        print(f"   Paper ID: {r.paper_id}")
        print(f"   Text preview: {r.text[:100]}...")
        print()






