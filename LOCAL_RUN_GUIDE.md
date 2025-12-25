# Local Development Guide - RAG Pipeline

This guide provides step-by-step instructions to run the RAG pipeline locally on your machine.

## Prerequisites

- **Python 3.11+** installed
- **Windows Subsystem for Linux 2 (WSL2)** with Ubuntu (if on Windows)
- **Docker Desktop** (optional, for containerized testing)
- **Git** for version control

## 1. Environment Setup

### Clone or Setup Project
```bash
cd your-projects-directory
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline
```

### Create Virtual Environment
```bash
python -m venv venv
```

### Activate Virtual Environment
**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```
**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```
**Linux/Mac:**
```bash
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Environment Configuration

### Create .env File
Create a `.env` file in the project root with the following content:

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

### Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Replace `your_gemini_api_key_here` in `.env`

## 3. Run the Application

### Start the Flask Server
```bash
python -m app.main
```

### Verify Server is Running
Open a new terminal and run:
```bash
curl http://127.0.0.1:8080/health
```

Expected response:
```json
{
  "service": "RAG Pipeline",
  "status": "healthy"
}
```

## 4. Access the Web Interface

Open your browser and navigate to:
```
http://127.0.0.1:8080
```

## 5. Test the Application

### Upload Documents
1. Click "Choose File" in the Upload section
2. Select a PDF or text file
3. Click "Upload & Index"
4. Wait for processing to complete

### Ask Questions
1. Switch to the "Query" tab
2. Type your question in the text area
3. Click "Ask Question"
4. View the AI-generated response with sources

### View Statistics
- Check the dashboard for document count, chunk count, and query metrics
- View recent evaluations in the "Evals" tab

## 6. Troubleshooting

### Common Issues

**ModuleNotFoundError:**
```bash
pip install -r requirements.txt
```

**Unicode Encoding Errors:**
- Ensure you're using Python 3.11+
- If running on Windows, use WSL2 or ensure proper encoding settings

**Port Already in Use:**
```bash
# Find process using port 8080
netstat -ano | findstr :8080
# Kill the process (replace PID)
taskkill /PID <PID> /F
```

**Gemini API Errors:**
- Verify your API key in `.env`
- Check API quota limits
- Ensure internet connection

## 7. Development Mode

### Auto-Reload (Development)
For automatic server restart on code changes:
```bash
pip install watchdog
python -m app.main  # Server will auto-reload
```

### Run with Debug Logging
Set environment variable for detailed logs:
```bash
export FLASK_ENV=development
python -m app.main
```

## 8. Data Persistence

### Local Data Storage
- **Documents**: Stored in `./chroma_db/` directory
- **Uploaded files**: Temporarily stored in memory (GCP mode stores in Cloud Storage)
- **Evaluation data**: Stored in `./evals/` directory

### Reset Application Data
To clear all stored data:
```bash
# Stop the server first
rm -rf chroma_db/
rm -rf evals/
```

## 9. Docker Testing (Optional)

### Build Docker Image
```bash
docker build -t rag-pipeline-local .
```

### Run in Docker
```bash
docker run -p 8080:8080 --env-file .env rag-pipeline-local
```

## 10. API Endpoints

The application provides REST API endpoints:

- `GET /` - Web interface
- `GET /health` - Health check
- `POST /api/upload` - Upload documents
- `POST /api/query` - Ask questions
- `POST /api/feedback` - Submit feedback
- `GET /api/stats` - Get statistics
- `GET /api/evals` - Get evaluations
- `GET /api/documents` - List documents
- `POST /api/clear` - Clear all data

## 11. Next Steps

Once local development is working, you can:

1. **Deploy to GCP**: Follow the `SETUP_GUIDE.md` for cloud deployment
2. **Add Features**: Extend the pipeline with new capabilities
3. **Integrate Services**: Add more GCP services as needed

## Support

If you encounter issues:
1. Check the terminal output for error messages
2. Verify all prerequisites are installed
3. Ensure `.env` file is properly configured
4. Test with the health endpoint first

---

**Happy coding! 🚀**
