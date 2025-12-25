"""
Main Flask Application - RAG Pipeline API
With GCP integration and evaluation tracking
"""
import os
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

app = Flask(__name__, template_folder='../templates')

# Lazy initialization
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


def get_doc_processor():
    """Lazy load document processor."""
    global _doc_processor
    if _doc_processor is None:
        from app.document_processor import DocumentProcessor
        _doc_processor = DocumentProcessor()
    return _doc_processor


def get_gcp_storage():
    """Lazy load GCP storage."""
    global _gcp_storage
    if _gcp_storage is None:
        try:
            from app.gcp_storage import GCPStorageManager
            _gcp_storage = GCPStorageManager()
        except Exception as e:
            print(f"WARNING: GCP Storage not available: {e}")
            _gcp_storage = None
    return _gcp_storage


@app.route('/')
def home():
    """Serve the main page."""
    try:
        rag_engine = get_rag_engine()
        stats = rag_engine.get_collection_stats()
    except Exception as e:
        stats = {"total_documents": 0, "error": str(e)}
    return render_template('index.html', stats=stats)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "RAG Pipeline"})


@app.route('/api/upload', methods=['POST'])
def upload_documents():
    """Upload and process documents."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Check file extension
    allowed_extensions = {'.pdf', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type. Use: {', '.join(allowed_extensions)}"}), 400

    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        doc_processor = get_doc_processor()
        rag_engine = get_rag_engine()
        gcp_storage = get_gcp_storage()
        
        # Upload to GCS if available
        gcs_uri = None
        if gcp_storage:
            gcs_uri = gcp_storage.upload_document(temp_path, file.filename)

        # Process the document
        if file_ext == '.pdf':
            text = doc_processor.load_pdf(temp_path)
        else:  # .txt
            text = doc_processor.load_text_file(temp_path)

        if not text.strip():
            return jsonify({"error": "Could not extract text from the document"}), 400

        # Chunk the document
        chunks = doc_processor.chunk_text(text)

        if not chunks:
            return jsonify({"error": "No content to index"}), 400

        # Add to vector store
        num_chunks = rag_engine.add_documents(
            chunks, 
            metadata=[{"source": file.filename}] * len(chunks),
            gcs_uri=gcs_uri
        )

        response = {
            "message": f"Successfully processed {file.filename}",
            "chunks_added": num_chunks,
            "total_documents": rag_engine.get_collection_stats()["total_documents"]
        }
        
        if gcs_uri:
            response["gcs_uri"] = gcs_uri

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        # Cleanup temp file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/query', methods=['POST'])
def query():
    """Query the RAG system."""
    data = request.get_json()

    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data['question'].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    n_results = min(data.get('n_results', 5), 10)

    try:
        rag_engine = get_rag_engine()
        result = rag_engine.query(question, n_results)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Query failed: {str(e)}"}), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Submit feedback for a query."""
    data = request.get_json()
    
    if not data or 'query_id' not in data or 'feedback' not in data:
        return jsonify({"error": "Missing query_id or feedback"}), 400
    
    query_id = data['query_id']
    feedback_type = data['feedback']  # "positive" or "negative"
    comment = data.get('comment')
    
    if feedback_type not in ['positive', 'negative']:
        return jsonify({"error": "Feedback must be 'positive' or 'negative'"}), 400
    
    try:
        rag_engine = get_rag_engine()
        success = rag_engine.add_feedback(query_id, feedback_type, comment)
        
        if success:
            return jsonify({"message": "Feedback recorded", "query_id": query_id})
        else:
            return jsonify({"error": "Query not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics')
def metrics():
    """Get evaluation metrics."""
    try:
        rag_engine = get_rag_engine()
        return jsonify(rag_engine.get_eval_metrics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/evals')
def evals():
    """Get recent evaluations."""
    limit = request.args.get('limit', 10, type=int)
    try:
        rag_engine = get_rag_engine()
        return jsonify({
            "evals": rag_engine.get_recent_evals(limit),
            "metrics": rag_engine.get_eval_metrics()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats')
def stats():
    """Get collection statistics."""
    try:
        rag_engine = get_rag_engine()
        return jsonify(rag_engine.get_collection_stats())
    except Exception as e:
        return jsonify({"error": str(e), "total_documents": 0})


@app.route('/api/documents')
def list_documents():
    """List documents stored in GCS."""
    try:
        gcp_storage = get_gcp_storage()
        if gcp_storage:
            docs = gcp_storage.list_documents()
            return jsonify({
                "documents": docs,
                "count": len(docs)
            })
        else:
            return jsonify({"documents": [], "count": 0, "message": "GCS not configured"})
    except Exception as e:
        return jsonify({"error": str(e), "documents": [], "count": 0}), 500


@app.route('/api/clear', methods=['POST'])
def clear():
    """Clear all documents from the collection."""
    try:
        rag_engine = get_rag_engine()
        result = rag_engine.clear_collection()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
