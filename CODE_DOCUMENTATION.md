# RAG Pipeline Code Documentation

This document provides a comprehensive explanation of the RAG (Retrieval-Augmented Generation) pipeline codebase, from entry points to APIs and all core components.

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Entry Point](#entry-point)
- [Core Components](#core-components)
- [API Endpoints](#api-endpoints)
- [Data Flow](#data-flow)
- [Configuration](#configuration)
- [Deployment](#deployment)

## 🏗️ Architecture Overview

The RAG pipeline consists of several interconnected components that work together to provide document indexing, retrieval, and question-answering capabilities:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │    Flask API     │    │  RAG Engine     │
│   (templates/)  │◄──►│   (main.py)      │◄──►│  (rag_engine.py)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Document       │    │   Vector Store   │
│   Processor     │    │   Upload/Process │    │   (ChromaDB)     │
│(document_proc..)│    │   (main.py)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Cloud Storage │    │   Evaluation     │    │   AI Models      │
│   (GCP)         │    │   Tracker        │    │   (Gemini)       │
│(gcp_storage.py) │    │ (evals.py)       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Technologies

- **Flask**: Web framework for REST API and UI serving
- **ChromaDB**: Vector database for document embeddings
- **Google Gemini**: AI model for embeddings and text generation
- **Google Cloud Storage**: Document persistence (optional)
- **LangChain**: Text splitting utilities
- **PyPDF2**: PDF text extraction

## 🚀 Entry Point

### `app/main.py`

The main entry point is `app/main.py`, a Flask application that serves both the web interface and REST API.

#### Key Features:

1. **Lazy Loading Pattern**: Components are initialized only when first accessed
2. **Error Handling**: Comprehensive error handling with detailed logging
3. **Configuration**: Environment-based configuration via `.env` file

#### Main Components:

```python
# Lazy-loaded components
_rag_engine = None
_doc_processor = None
_gcp_storage = None

def get_rag_engine():
    """Lazy load RAG engine."""
    global _rag_engine
    if _rag_engine is None:
        from app.rag_engine import RAGEngine
        _rag_engine = RAGEngine()
    return _rag_engine
```

#### Application Startup:

```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
```

## 🔧 Core Components

### 1. RAG Engine (`app/rag_engine.py`)

The core component orchestrating the RAG pipeline.

#### Key Classes:

#### `RAGEngine`

**Initialization:**
```python
def __init__(self, collection_name: str = "documents"):
    # Initialize Gemini AI
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    self.model = genai.GenerativeModel('gemini-2.0-flash')
    self.embedding_model = 'models/text-embedding-004'

    # Initialize ChromaDB
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    os.makedirs(persist_dir, exist_ok=True)
    self.chroma_client = chromadb.PersistentClient(path=persist_dir)

    # Create collection with cosine similarity
    self.collection = self.chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Initialize evaluation tracker
    self.eval_tracker = EvalTracker(storage_path="./evals")

    # Try GCP Storage (optional)
    self.gcp_storage = None
    try:
        from app.gcp_storage import GCPStorageManager
        self.gcp_storage = GCPStorageManager()
    except Exception as e:
        print(f"WARNING: GCP Storage not available: {e}")
```

#### Key Methods:

**Document Addition:**
```python
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
```

**Document Search:**
```python
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
```

**Response Generation:**
```python
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
```

**Complete Query Pipeline:**
```python
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
```

### 2. Document Processor (`app/document_processor.py`)

Handles document loading and text chunking.

#### Key Methods:

**PDF Loading:**
```python
def load_pdf(self, file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
```

**Text Chunking:**
```python
def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
    self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

def chunk_text(self, text: str) -> List[str]:
    """Split a single text into smaller chunks."""
    return self.text_splitter.split_text(text)
```

### 3. GCP Storage Manager (`app/gcp_storage.py`)

Manages document storage in Google Cloud Storage.

#### Key Methods:

**Bucket Management:**
```python
def _get_or_create_bucket(self):
    """Get existing bucket or create new one."""
    try:
        bucket = self.client.get_bucket(self.bucket_name)
        print(f"SUCCESS: Using existing bucket: {self.bucket_name}")
        return bucket
    except Exception:
        # Bucket doesn't exist, create it
        try:
            bucket = self.client.create_bucket(
                self.bucket_name,
                location=os.getenv("GCP_REGION", "us-central1")
            )
            print(f"SUCCESS: Created new bucket: {self.bucket_name}")
            return bucket
        except Exception as e:
            print(f"WARNING: Could not create bucket: {e}")
            print("Using local storage fallback")
            return None
```

**Document Upload:**
```python
def upload_document(self, file_path: str, original_filename: str) -> Optional[str]:
    """Upload a document to Cloud Storage."""
    if not self.bucket:
        return None

    # Generate unique blob name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob_name = f"documents/{timestamp}_{original_filename}"

    blob = self.bucket.blob(blob_name)
    blob.upload_from_filename(file_path)

    gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
    print(f"SUCCESS: Uploaded to: {gcs_uri}")
    return gcs_uri
```

### 4. Evaluation Tracker (`app/evals.py`)

Tracks performance metrics and user feedback.

#### Key Classes:

**QueryEval Dataclass:**
```python
@dataclass
class QueryEval:
    """Evaluation data for a single query."""
    query_id: str
    timestamp: str
    question: str
    answer: str
    num_sources: int
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    sources_preview: List[str]
    feedback: Optional[str] = None  # "positive", "negative", or None
    feedback_comment: Optional[str] = None
```

**EvalTracker Class:**
```python
def log_query(
    self,
    question: str,
    answer: str,
    sources: List[str],
    retrieval_time: float,
    generation_time: float
) -> str:
    """Log a query evaluation."""
    query_id = f"q_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    eval_entry = QueryEval(
        query_id=query_id,
        timestamp=datetime.now().isoformat(),
        question=question,
        answer=answer[:500] + "..." if len(answer) > 500 else answer,
        num_sources=len(sources),
        retrieval_time_ms=round(retrieval_time * 1000, 2),
        generation_time_ms=round(generation_time * 1000, 2),
        total_time_ms=round((retrieval_time + generation_time) * 1000, 2),
        sources_preview=[s[:100] + "..." if len(s) > 100 else s for s in sources[:3]]
    )

    self.evals.append(eval_entry)
    self._save_evals()

    return query_id
```

## 🌐 API Endpoints

### Web Interface

#### `GET /`
Serves the main web interface with tabs for Query, Upload, and Evaluations.

### Health Check

#### `GET /health`
```json
{
  "status": "healthy",
  "service": "RAG Pipeline"
}
```

### Document Management

#### `POST /api/upload`
Upload and process documents (PDF/TXT).

**Request:** Multipart form data with `file` field
**Response:**
```json
{
  "message": "Successfully processed document.pdf",
  "chunks_added": 15,
  "total_documents": 15,
  "gcs_uri": "gs://project-rag-docs/documents/20241201_143000_document.pdf"
}
```

#### `GET /api/documents`
List documents stored in GCP (if configured).

**Response:**
```json
{
  "documents": [
    {
      "name": "documents/20241201_143000_document.pdf",
      "size": 245760,
      "updated": "2024-12-01T14:30:00Z"
    }
  ],
  "count": 1
}
```

### Query Processing

#### `POST /api/query`
Ask questions against indexed documents.

**Request:**
```json
{
  "question": "What is machine learning?",
  "n_results": 5
}
```

**Response:**
```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "sources": ["Machine learning is...", "There are several types..."],
  "num_sources": 5,
  "query_id": "q_20241201_143000_123456",
  "relevance_scores": [95.2, 87.1, 82.5],
  "metrics": {
    "retrieval_time_ms": 45.67,
    "generation_time_ms": 123.89,
    "total_time_ms": 169.56
  }
}
```

### Feedback System

#### `POST /api/feedback`
Submit user feedback for query responses.

**Request:**
```json
{
  "query_id": "q_20241201_143000_123456",
  "feedback": "positive",
  "comment": "Great answer!"
}
```

### Analytics & Metrics

#### `GET /api/stats`
Get collection statistics and evaluation metrics.

**Response:**
```json
{
  "total_documents": 25,
  "eval_metrics": {
    "total_queries": 150,
    "avg_retrieval_time_ms": 42.3,
    "avg_generation_time_ms": 118.7,
    "avg_total_time_ms": 161.0,
    "positive_feedback_rate": 0.85,
    "recent_feedback_trend": [0.8, 0.82, 0.85]
  }
}
```

#### `GET /api/metrics`
Get evaluation metrics only.

#### `GET /api/evals`
Get recent query evaluations.

**Query Parameters:**
- `limit` (int): Number of evaluations to return (default: 10)

### Administration

#### `POST /api/clear`
Clear all documents from the collection.

**Response:**
```json
{
  "message": "Collection cleared",
  "total_documents": 0
}
```

## 🔄 Data Flow

### Document Upload Flow

1. **File Upload** (`POST /api/upload`)
   - Validate file type (PDF/TXT)
   - Save temporarily to disk

2. **Document Processing** (`DocumentProcessor`)
   - Extract text from PDF/TXT
   - Split into chunks (1000 chars, 200 overlap)

3. **Embedding Generation** (`RAGEngine.get_embedding()`)
   - Generate embeddings using Gemini
   - Different models for documents vs queries

4. **Vector Storage** (`RAGEngine.add_documents()`)
   - Store embeddings in ChromaDB
   - Add metadata (source, GCS URI if available)

5. **Cloud Storage** (`GCPStorageManager`) *[Optional]*
   - Upload original document to GCS
   - Return GCS URI for metadata

### Query Processing Flow

1. **Query Reception** (`POST /api/query`)
   - Validate question input
   - Set retrieval parameters

2. **Document Retrieval** (`RAGEngine.search()`)
   - Generate query embedding
   - Search ChromaDB for similar documents
   - Return top-k results with distances

3. **Response Generation** (`RAGEngine.generate_response()`)
   - Combine retrieved documents as context
   - Generate prompt with context + question
   - Call Gemini for final answer

4. **Evaluation Logging** (`EvalTracker.log_query()`)
   - Record query, response, timing
   - Store for performance analysis

5. **Response Assembly**
   - Format answer with sources
   - Include relevance scores
   - Add performance metrics

## ⚙️ Configuration

### Environment Variables

#### Required:
- `GOOGLE_API_KEY`: Gemini API key for AI models

#### Optional (GCP):
- `GOOGLE_CLOUD_PROJECT`: GCP project ID
- `GCP_REGION`: GCP region (default: us-central1)
- `GCS_BUCKET_NAME`: Cloud Storage bucket name

#### Optional (App):
- `CHROMA_PERSIST_DIR`: Vector DB storage path (default: ./chroma_db)
- `PORT`: Server port (default: 8080)
- `FLASK_ENV`: Set to 'development' for debug mode

### Example `.env` file:
```env
# Gemini API Configuration
GOOGLE_API_KEY=your_gemini_api_key_here

# GCP Configuration (optional - leave empty for local-only mode)
GOOGLE_CLOUD_PROJECT=your-project-id
GCP_REGION=us-central1
GCS_BUCKET_NAME=your-project-id-rag-docs

# Application Settings
CHROMA_PERSIST_DIR=./chroma_db
```

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env  # Edit with your keys

# Run the application
python -m app.main
```

### Docker Deployment
```bash
# Build image
docker build -t rag-pipeline .

# Run container
docker run -p 8080:8080 --env-file .env rag-pipeline
```

### Cloud Run (GCP)
```bash
# Build and deploy
gcloud run deploy rag-pipeline \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars GOOGLE_API_KEY=your-key \
  --allow-unauthenticated
```

### CI/CD Pipeline
Uses Cloud Build with `cloudbuild.yaml`:
- Build Docker image
- Push to Artifact Registry
- Deploy to Cloud Run
- Trigger on GitHub pushes

## 🔧 Key Design Patterns

### 1. Lazy Loading
Components are initialized only when first accessed, improving startup time and allowing graceful degradation when optional services are unavailable.

### 2. Error Resilience
- GCP services are optional - app works locally without them
- Comprehensive error handling with user-friendly messages
- Fallback mechanisms for service failures

### 3. Performance Monitoring
- Built-in timing for all operations
- Query evaluation tracking
- User feedback collection

### 4. Separation of Concerns
- Document processing separate from RAG logic
- Storage abstraction (local vs cloud)
- Evaluation tracking as independent module

### 5. Configuration Management
- Environment-based configuration
- Sensible defaults
- Clear error messages for missing required settings

## 📊 Monitoring & Metrics

The system tracks:
- **Query Performance**: Retrieval and generation times
- **User Feedback**: Positive/negative ratings
- **System Usage**: Query volume, document count
- **Error Rates**: API failures, processing errors

All metrics are accessible via `/api/metrics` and `/api/stats` endpoints.

## 🔒 Security Considerations

- API keys stored securely in environment variables
- File upload validation (type, size limits)
- CORS headers for web interface
- Input sanitization for queries
- Secure cloud storage with proper IAM

## 🚀 Future Enhancements

- **Multi-modal support**: Images, audio documents
- **Advanced chunking**: Semantic chunking strategies
- **Model fine-tuning**: Custom embeddings for domain-specific content
- **Caching**: Response caching for frequently asked questions
- **Batch processing**: Bulk document upload and processing
- **Analytics dashboard**: Rich visualizations of usage metrics

---

**This documentation provides a complete understanding of the RAG pipeline architecture, from the entry point through all core components and API endpoints. The modular design allows for easy extension and deployment across different environments.**
