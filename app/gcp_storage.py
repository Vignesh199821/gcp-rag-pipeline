"""
GCP Cloud Storage - Document storage and management
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class GCPStorageManager:
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.bucket_name = os.getenv("GCS_BUCKET_NAME", f"{self.project_id}-rag-docs")
        
        # Initialize client
        self.client = storage.Client(project=self.project_id)
        self.bucket = self._get_or_create_bucket()
    
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
    
    def upload_document(self, file_path: str, original_filename: str) -> Optional[str]:
        """Upload a document to Cloud Storage."""
        if not self.bucket:
            return None
            
        try:
            # Create unique blob name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"documents/{timestamp}_{original_filename}"
            
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            
            # Make it accessible
            gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
            print(f"SUCCESS: Uploaded to: {gcs_uri}")
            
            return gcs_uri
        except Exception as e:
            print(f"ERROR: Upload failed: {e}")
            return None
    
    def upload_document_bytes(self, content: bytes, filename: str, content_type: str = "application/octet-stream") -> Optional[str]:
        """Upload document from bytes."""
        if not self.bucket:
            return None
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"documents/{timestamp}_{filename}"
            
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(content, content_type=content_type)
            
            gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
            print(f"SUCCESS: Uploaded to: {gcs_uri}")
            
            return gcs_uri
        except Exception as e:
            print(f"ERROR: Upload failed: {e}")
            return None
    
    def list_documents(self) -> List[Dict]:
        """List all documents in the bucket."""
        if not self.bucket:
            return []
            
        documents = []
        blobs = self.bucket.list_blobs(prefix="documents/")
        
        for blob in blobs:
            documents.append({
                "name": blob.name,
                "size": blob.size,
                "created": blob.time_created.isoformat() if blob.time_created else None,
                "uri": f"gs://{self.bucket_name}/{blob.name}"
            })
        
        return documents
    
    def download_document(self, blob_name: str) -> Optional[bytes]:
        """Download a document from Cloud Storage."""
        if not self.bucket:
            return None
            
        try:
            blob = self.bucket.blob(blob_name)
            return blob.download_as_bytes()
        except Exception as e:
            print(f"ERROR: Download failed: {e}")
            return None
    
    def delete_document(self, blob_name: str) -> bool:
        """Delete a document from Cloud Storage."""
        if not self.bucket:
            return False
            
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception as e:
            print(f"ERROR: Delete failed: {e}")
            return False
    
    def save_metadata(self, doc_id: str, metadata: Dict):
        """Save document metadata to Cloud Storage."""
        if not self.bucket:
            return
            
        try:
            blob_name = f"metadata/{doc_id}.json"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(metadata, indent=2),
                content_type="application/json"
            )
        except Exception as e:
            print(f"ERROR: Metadata save failed: {e}")
    
    def get_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get document metadata from Cloud Storage."""
        if not self.bucket:
            return None
            
        try:
            blob_name = f"metadata/{doc_id}.json"
            blob = self.bucket.blob(blob_name)
            content = blob.download_as_string()
            return json.loads(content)
        except Exception:
            return None

