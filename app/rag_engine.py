"""
RAG Engine - Core RAG functionality using Gemini and ChromaDB
With GCP integration and evaluation tracking
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from app.evals import EvalTracker, Timer

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"


class RAGEngine:
    def __init__(self, collection_name: str = "documents"):
        # Initialize Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "paste-your-gemini-api-key-here":
            raise ValueError("Please set your GOOGLE_API_KEY in the .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.embedding_model = 'models/text-embedding-004'

        # Initialize ChromaDB (vector store)
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize evaluation tracker
        self.eval_tracker = EvalTracker(storage_path="./evals")
        
        # Try to initialize GCP Storage (optional)
        self.gcp_storage = None
        try:
            from app.gcp_storage import GCPStorageManager
            self.gcp_storage = GCPStorageManager()
        except Exception as e:
            print(f"WARNING: GCP Storage not available: {e}")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini's embedding model."""
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def get_query_embedding(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

    def add_documents(
        self, 
        chunks: List[str], 
        metadata: Optional[List[dict]] = None,
        gcs_uri: Optional[str] = None
    ) -> int:
        """Add document chunks to the vector store."""
        if not chunks:
            return 0

        # Generate embeddings for all chunks
        embeddings = []
        for chunk in chunks:
            embedding = self.get_embedding(chunk)
            embeddings.append(embedding)

        # Generate unique IDs
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(chunks))]

        # Add GCS URI to metadata if available
        if metadata is None:
            metadata = [{"source": "uploaded"}] * len(chunks)
        
        if gcs_uri:
            for m in metadata:
                m["gcs_uri"] = gcs_uri

        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids,
            metadatas=metadata
        )

        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> Tuple[List[str], List[float]]:
        """Search for relevant documents. Returns (documents, distances)."""
        if self.collection.count() == 0:
            return [], []
            
        # Get query embedding
        query_embedding = self.get_query_embedding(query)

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "distances"]
        )

        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results.get('distances') else []
        
        return documents, distances

    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using Gemini with retrieved context."""
        context_text = "\n\n---\n\n".join(context)

        prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
If the context doesn't contain relevant information to answer the question, say so honestly.
Be concise but thorough in your answers.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

        response = self.model.generate_content(prompt)
        return response.text

    def query(self, question: str, n_results: int = 5) -> dict:
        """Complete RAG pipeline with evaluation tracking."""
        
        # Step 1: Retrieve relevant documents (with timing)
        with Timer() as retrieval_timer:
            relevant_docs, distances = self.search(question, n_results)

        if not relevant_docs:
            return {
                "answer": "I don't have any documents indexed yet. Please upload some documents first!",
                "sources": [],
                "num_sources": 0,
                "query_id": None,
                "metrics": {
                    "retrieval_time_ms": round(retrieval_timer.elapsed * 1000, 2),
                    "generation_time_ms": 0
                }
            }

        # Step 2: Generate response with context (with timing)
        with Timer() as generation_timer:
            answer = self.generate_response(question, relevant_docs)

        # Step 3: Log evaluation
        query_id = self.eval_tracker.log_query(
            question=question,
            answer=answer,
            sources=relevant_docs,
            retrieval_time=retrieval_timer.elapsed,
            generation_time=generation_timer.elapsed
        )
        
        # Calculate relevance scores (lower distance = more relevant)
        relevance_scores = []
        if distances:
            # Convert distances to similarity scores (1 - distance for cosine)
            relevance_scores = [round((1 - d) * 100, 1) for d in distances[:3]]

        return {
            "answer": answer,
            "sources": relevant_docs[:3],
            "num_sources": len(relevant_docs),
            "query_id": query_id,
            "relevance_scores": relevance_scores,
            "metrics": {
                "retrieval_time_ms": round(retrieval_timer.elapsed * 1000, 2),
                "generation_time_ms": round(generation_timer.elapsed * 1000, 2),
                "total_time_ms": round((retrieval_timer.elapsed + generation_timer.elapsed) * 1000, 2)
            }
        }
    
    def add_feedback(self, query_id: str, feedback: str, comment: Optional[str] = None) -> bool:
        """Add user feedback for a query."""
        return self.eval_tracker.add_feedback(query_id, feedback, comment)
    
    def get_eval_metrics(self) -> Dict:
        """Get evaluation metrics."""
        return self.eval_tracker.get_metrics()
    
    def get_recent_evals(self, limit: int = 10) -> List[Dict]:
        """Get recent evaluations."""
        return self.eval_tracker.get_recent_evals(limit)

    def get_collection_stats(self) -> dict:
        """Get statistics about the document collection."""
        stats = {
            "total_documents": self.collection.count()
        }
        
        # Add eval metrics
        eval_metrics = self.get_eval_metrics()
        if eval_metrics:
            stats["eval_metrics"] = eval_metrics
            
        return stats
    
    def clear_collection(self) -> dict:
        """Clear all documents from the collection."""
        self.chroma_client.delete_collection("documents")
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        return {"message": "Collection cleared", "total_documents": 0}
