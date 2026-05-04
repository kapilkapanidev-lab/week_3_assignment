#!/usr/bin/env python3
"""
FastAPI Server for SIGGRAPH 2025 RAG Pipeline.

Provides REST API for the RAG pipeline.
Frontend is served from the /frontend folder.

Requirements:
    pip install fastapi uvicorn

Usage:
    python api_server.py
"""

import asyncio
import time
import os
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_generate import RAGGenerator, GenerationConfig, SYSTEM_PROMPT


# Global instances
rag_generator: Optional[RAGGenerator] = None

# Paths
BASE_DIR = Path(__file__).parent



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on startup."""
    global rag_generator
    
    print("\n" + "="*60)
    print("🚀 Initializing RAG pipeline...")
    print("="*60)
    
    config = GenerationConfig(
        llm_provider="openai",
        retrieval_top_k=8,
        refine_query=True,
        use_reranker=True
    )
    
    rag_generator = RAGGenerator(config)
    print("\n✅ RAG pipeline ready!")
    print("="*60 + "\n")
    
    yield
    
    print("\n👋 Shutting down...")


app = FastAPI(
    title="SIGGRAPH 2025 RAG API",
    description="RAG pipeline for SIGGRAPH 2025 papers",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 8
    refine_query: Optional[bool] = True
    use_reranker: Optional[bool] = True
    temperature: Optional[float] = 0.3


class QueryResponse(BaseModel):
    query: str
    refined_query: Optional[str] = None
    answer: str
    sources: list[dict] = []
    processing_time: float


# API Endpoints
@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "rag_initialized": rag_generator is not None,
        "timestamp": time.time()
    }


@app.get("/api/info")
async def api_info():
    """API information."""
    return {
        "service": "SIGGRAPH 2025 RAG API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Frontend UI",
            "GET /health": "Health check",
            "POST /api/query": "Query endpoint"
        }
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG pipeline.
    Returns answer with citations and sources.
    """
    if not rag_generator:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    start_time = time.time()
    
    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: rag_generator.generate(
                request.query,
                top_k=request.top_k,
                return_sources=True
            )
        )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=response["query"],
            refined_query=response.get("refined_query"),
            answer=response["answer"],
            sources=response.get("sources", []),
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_rag_response(
    query: str,
    top_k: int = 8,
    refine_query: bool = True,
    use_reranker: bool = True
) -> AsyncGenerator[str, None]:
    """
    Stream RAG response with real-time progress updates.
    Uses Server-Sent Events (SSE) format.
    """
    def emit(event_type: str, data: dict) -> str:
        """Format SSE event."""
        return f"data: {json.dumps({'type': event_type, **data})}\n\n"
    
    start_time = time.time()
    loop = asyncio.get_event_loop()
    
    try:
        # Stage 1: Refining query
        refined = query
        if refine_query:
            yield emit("progress", {"message": "Refining query...", "stage": "refining"})
            
            refined = await loop.run_in_executor(
                None,
                lambda: rag_generator.refine_query(query)
            )
            
            if refined != query:
                yield emit("progress", {
                    "message": f"Refined: {refined}",
                    "stage": "refined",
                    "original": query,
                    "refined": refined
                })
        
        # Stage 2: Searching
        yield emit("progress", {"message": "Searching 11,008 paper chunks...", "stage": "searching"})
        
        results = await loop.run_in_executor(
            None,
            lambda: rag_generator.retrieval.retrieve(refined, top_k=top_k)
        )
        
        if not results:
            yield emit("error", {"message": "No relevant papers found"})
            return
        
        num_papers = len(set(r.paper_id for r in results))
        yield emit("progress", {
            "message": f"Found {len(results)} sources from {num_papers} papers",
            "stage": "found",
            "num_chunks": len(results),
            "num_papers": num_papers
        })
        
        # Stage 3: Generating
        yield emit("progress", {"message": "Generating answer...", "stage": "generating"})
        
        # Format context and build sources
        context = rag_generator._format_context(results)
        sources_metadata = rag_generator._build_sources_metadata(results)
        
        # Build prompt
        user_message = f"""Based on the following research paper excerpts, answer this question:

Question: {query}

Research Paper Excerpts:
{context}

IMPORTANT: You have been provided with {len(results)} paper excerpts. Make sure to:
1. Review ALL provided papers and cite every one that is relevant to the question
2. Use inline citations [Paper Title] for all claims
3. For "which paper" or "what papers" questions, list ALL relevant papers
4. Do NOT include a References section - only use inline citations"""

        # Stream from OpenRouter
        import requests
        
        headers = {
            "Authorization": f"Bearer {rag_generator.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": f"openai/{rag_generator.config.llm_model}",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "temperature": rag_generator.config.temperature,
            "max_tokens": rag_generator.config.max_tokens,
            "stream": True
        }
        
        response = requests.post(
            f"{rag_generator.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
            stream=True
        )
        
        answer_chunks = []
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk["choices"][0]["delta"].get("content"):
                            content = chunk["choices"][0]["delta"]["content"]
                            answer_chunks.append(content)
                            yield emit("chunk", {"content": content})
                    except json.JSONDecodeError:
                        continue
        
        answer = "".join(answer_chunks)
        
        # Stage 4: Complete
        processing_time = time.time() - start_time
        yield emit("complete", {
            "answer": answer,
            "sources": list(sources_metadata.values()),
            "refined_query": refined if refined != query else None,
            "processing_time": processing_time
        })
    
    except Exception as e:
        yield emit("error", {"message": str(e)})


@app.get("/api/stream")
async def stream_query(
    query: str = Query(..., description="Your question"),
    top_k: int = Query(8, description="Number of sources"),
    refine_query: bool = Query(True, description="Refine query"),
    use_reranker: bool = Query(True, description="Use reranker")
):
    """
    Stream query results with real-time progress updates.
    Returns Server-Sent Events (SSE).
    """
    if not rag_generator:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    return StreamingResponse(
        stream_rag_response(query, top_k, refine_query, use_reranker),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for real-time RAG streaming with detailed progress updates.
    """
    await websocket.accept()
    
    try:
        # Receive query parameters
        data = await websocket.receive_json()
        query = data.get("query", "")
        top_k = data.get("top_k", 8)
        refine_query_flag = data.get("refine_query", True)
        use_reranker = data.get("use_reranker", True)
        
        if not query:
            await websocket.send_json({"type": "error", "message": "Query is required"})
            await websocket.close()
            return
        
        if not rag_generator:
            await websocket.send_json({"type": "error", "message": "RAG pipeline not initialized"})
            await websocket.close()
            return
        
        start_time = time.time()
        loop = asyncio.get_event_loop()
        
        # Get total chunks count
        total_chunks = len(rag_generator.retrieval.chunks)
        
        # Stage 1: Query Refinement
        refined = query
        if refine_query_flag:
            await websocket.send_json({
                "type": "progress",
                "stage": "refining",
                "message": "Refining your query..."
            })
            
            refined = await loop.run_in_executor(
                None,
                lambda: rag_generator.refine_query(query)
            )
            
            if refined != query:
                await websocket.send_json({
                    "type": "progress",
                    "stage": "refined",
                    "message": f"Refined: {refined}",
                    "original": query,
                    "refined": refined
                })
        
        # Stage 2: Searching with detailed progress
        await websocket.send_json({
            "type": "progress",
            "stage": "searching",
            "message": f"Searching {total_chunks:,} document chunks (semantic + keyword search)..."
        })
        
        # Use the full retrieve method which handles everything properly
        results = await loop.run_in_executor(
            None,
            lambda: rag_generator.retrieval.retrieve(
                refined, 
                top_k=top_k,
                use_hybrid=True,
                use_reranker=use_reranker
            )
        )
        
        if not results:
            await websocket.send_json({"type": "error", "message": "No relevant papers found"})
            await websocket.close()
            return
        
        num_papers = len(set(r.paper_id for r in results))
        await websocket.send_json({
            "type": "progress",
            "stage": "found",
            "message": f"Selected {len(results)} best sources from {num_papers} papers",
            "num_chunks": len(results),
            "num_papers": num_papers
        })
        
        # Stage 6: Generating Answer
        await websocket.send_json({
            "type": "progress",
            "stage": "generating",
            "message": "Generating answer with GPT-4..."
        })
        
        # Format context and build sources
        context = rag_generator._format_context(results)
        sources_metadata = rag_generator._build_sources_metadata(results)
        
        # Build prompt
        user_message = f"""Based on the following research paper excerpts, answer this question:

Question: {query}

Research Paper Excerpts:
{context}

IMPORTANT: You have been provided with {len(results)} paper excerpts. Make sure to:
1. Review ALL provided papers and cite every one that is relevant to the question
2. Use inline citations [Paper Title] for all claims
3. For "which paper" or "what papers" questions, list ALL relevant papers
4. Do NOT include a References section - only use inline citations"""

        # Stream from OpenRouter
        import requests
        
        headers = {
            "Authorization": f"Bearer {rag_generator.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": f"openai/{rag_generator.config.llm_model}",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "temperature": rag_generator.config.temperature,
            "max_tokens": rag_generator.config.max_tokens,
            "stream": True
        }
        
        response = requests.post(
            f"{rag_generator.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
            stream=True
        )
        
        answer_chunks = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        if chunk["choices"][0]["delta"].get("content"):
                            content = chunk["choices"][0]["delta"]["content"]
                            answer_chunks.append(content)
                            await websocket.send_json({
                                "type": "chunk",
                                "content": content
                            })
                    except json.JSONDecodeError:
                        continue
        
        answer = "".join(answer_chunks)
        
        # Stage 7: Complete
        processing_time = time.time() - start_time
        await websocket.send_json({
            "type": "complete",
            "answer": answer,
            "sources": list(sources_metadata.values()),
            "refined_query": refined if refined != query else None,
            "processing_time": processing_time
        })
        
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (for deployment) or use 8082 for local dev
    port = int(os.getenv("PORT", 8082))
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║           SIGGRAPH 2025 RAG API Server                        ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Starting server on http://0.0.0.0:{port}                     ║
║                                                                ║
║  Endpoints:                                                    ║
║    • GET  /              - Frontend UI                         ║
║    • GET  /docs          - API Documentation                   ║
║    • POST /api/query     - Query endpoint                      ║
║    • WS  /ws/query       - WebSocket streaming                 ║
║                                                                ║
║  Frontend files: ./frontend/                                   ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
