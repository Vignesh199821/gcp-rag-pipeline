# GCP RAG Pipeline Setup Guide - Complete Step-by-Step Instructions

## Overview
This guide provides every single step needed to set up a complete RAG (Retrieval-Augmented Generation) pipeline on Google Cloud Platform from scratch. No prior GCP knowledge required.

## Prerequisites
- A Google account (gmail)
- Windows 10/11 (version 2004 or higher, build 19041 or higher)
- Basic command line knowledge
- Python knowledge (but we'll guide through everything)

**Note**: Complete the WSL2 setup section below before proceeding with GCP setup.

---

## Pre-Setup: Install Windows Subsystem for Linux (WSL2)

### Why WSL2?
WSL2 enables you to run a Linux environment directly on Windows, which is essential for:
- Running Docker containers properly
- Using Linux-based development tools
- Ensuring compatibility with cloud deployment environments
- Running bash scripts and Linux commands

### Step 1: Enable WSL and Virtual Machine Platform
1. Open PowerShell as Administrator (right-click Start → Terminal (Admin))
2. Run this command to enable WSL:
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   ```
3. Enable Virtual Machine Platform:
   ```powershell
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```
4. **Restart your computer** (required for changes to take effect)

### Step 2: Download and Install WSL2 Kernel
1. Download the WSL2 Linux kernel update package from: https://aka.ms/wsl2kernel
2. Run the downloaded .msi file and follow the installation wizard
3. Click "Next" through all prompts, then "Finish"

### Step 3: Set WSL2 as Default Version
1. Open PowerShell as Administrator again
2. Set WSL2 as the default version:
   ```powershell
   wsl --set-default-version 2
   ```

### Step 4: Install Ubuntu Distribution
1. Open Microsoft Store from Start menu
2. Search for "Ubuntu" (or "Ubuntu 22.04 LTS")
3. Click "Get" to download and install
4. After installation, launch Ubuntu from Start menu
5. Follow the setup prompts:
   - Create a username (lowercase, no spaces)
   - Create a password
   - Confirm password

### Step 5: Update Ubuntu
1. In the Ubuntu terminal, update packages:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
2. Install essential tools:
   ```bash
   sudo apt install -y curl wget git
   ```

### Step 6: Verify WSL2 Installation
1. Open PowerShell and check WSL status:
   ```powershell
   wsl --list --verbose
   ```
   You should see your Ubuntu distribution with VERSION 2 and STATE Running

### Step 7: Configure Windows Terminal (Optional but Recommended)
1. Install Windows Terminal from Microsoft Store if not already installed
2. Open Windows Terminal
3. Click the dropdown arrow (∨) → Settings
4. Under "Profiles", you should see "Ubuntu" listed
5. Set Ubuntu as default if desired

### Common WSL Issues and Solutions

#### Issue: "WSL 2 requires an update to its kernel component"
**Solution**: Install the WSL2 kernel update package from https://aka.ms/wsl2kernel

#### Issue: "The virtual machine could not be started"
**Solution**: Enable virtualization in BIOS and ensure Hyper-V is enabled:
```powershell
bcdedit /set hypervisorlaunchtype auto
```

#### Issue: Ubuntu won't start or shows error
**Solution**: Reset WSL and reinstall:
```powershell
wsl --shutdown
wsl --unregister Ubuntu
# Then reinstall Ubuntu from Microsoft Store
```

#### Issue: DNS/Network issues in WSL
**Solution**: Create/edit `/etc/resolv.conf` in WSL:
```bash
sudo nano /etc/resolv.conf
# Add: nameserver 8.8.8.8
```

#### Issue: Permission denied when running Docker commands
**Solution**: Ensure you're in the docker group and WSL integration is enabled in Docker Desktop

### WSL File System Navigation
- Windows files accessible at: `/mnt/c/` (C drive), `/mnt/d/` (D drive), etc.
- WSL files accessible in Windows at: `\\wsl$\Ubuntu\` (in File Explorer)
- Your project will be at: `/mnt/c/Users/YourUsername/Desktop/projects_3/gcp_demo_project`

---

## Phase 1: Initial GCP Project Setup

---

## Phase 1: Initial GCP Project Setup

### Step 1: Create Google Cloud Project
1. Open your web browser and navigate to: https://console.cloud.google.com/
2. Sign in with your Google account (gmail)
3. On the Google Cloud Console homepage, look for the project selector at the top of the page (next to "Google Cloud" logo)
4. Click on the project dropdown menu
5. Click the blue "NEW PROJECT" button
6. In the "New Project" dialog box:
   - Project name: `rag-q-results` (or your choice)
   - Organization: Leave as "No organization" (if you see this option)
   - Location: Leave as default
7. Click the blue "CREATE" button
8. Wait for project creation notification (takes 30-60 seconds)
9. Make sure your new project is selected in the project dropdown at the top

### Step 2: Enable Billing
1. In the GCP Console, use the search bar at the top (magnifying glass icon) and type "Billing"
2. Click on "Billing" from the search results
3. On the Billing page, click the blue "LINK A BILLING ACCOUNT" button
4. If you don't have a billing account, click "CREATE BILLING ACCOUNT"
5. In the "Create billing account" form:
   - Account type: Select "Individual"
   - Country: Select your country from the dropdown
   - Name: Enter your full name
   - Address: Fill in your complete address
6. Under "Payment method", click "Add credit or debit card"
7. Fill in your card details and billing address
8. Click the blue "START MY FREE TRIAL" button
9. Review and accept the terms and conditions
10. The billing account will be automatically linked to your project

### Step 3: Enable Required APIs
1. In the GCP Console, click the hamburger menu (☰) in the top-left corner
2. Scroll down and click on "APIs & Services" → then click "Library"
3. In the API Library page, use the search box to find each API
4. For each API, click on it, then click the blue "ENABLE" button
5. Enable these APIs one by one (wait 2-3 minutes between each):
   - **Vertex AI API** (search: "Vertex AI API")
   - **Cloud Storage API** (search: "Cloud Storage")
   - **Cloud Run API** (search: "Cloud Run")
   - **Cloud Build API** (search: "Cloud Build")
   - **Artifact Registry API** (search: "Artifact Registry")
   - **Secret Manager API** (search: "Secret Manager")
   - **Compute Engine API** (search: "Compute Engine")

**Important**: Wait 2-3 minutes after enabling each API before proceeding to the next one.

---

## Phase 2: Install and Configure Development Tools

### Step 4: Install Google Cloud SDK (gcloud CLI)
1. Download from: https://cloud.google.com/sdk/docs/install#windows
2. Run the installer
3. During installation, choose:
   - Install location: Default
   - Start Google Cloud SDK shell: Yes
   - Run 'gcloud init': Yes

### Step 5: Initialize gcloud CLI
1. Open Command Prompt as Administrator
2. Run: `gcloud init`
3. Choose: "Log in with a new account"
4. Browser opens - sign in with your Google account
5. Choose your project: `rag-q-results`
6. Choose default region: `us-central1`
7. Choose default zone: `us-central1-c`

### Step 6: Install Docker Desktop
1. Download from: https://www.docker.com/products/docker-desktop/
2. Run installer
3. During installation:
   - Enable WSL2 backend
   - Start Docker Desktop when setup completes
4. After installation, start Docker Desktop
5. Sign in with Docker Hub account (create one if needed)

### Step 7: Configure Docker with WSL2
1. Open Docker Desktop (installed in Step 6)
2. Click the gear/settings icon in Docker Desktop
3. Go to "Resources" tab → "WSL Integration"
4. Enable the option "Enable integration with my default WSL distro"
5. Check the box next to your Ubuntu distribution
6. Click "Apply & Restart"
7. Verify WSL integration:
   - Open Ubuntu terminal
   - Run: `docker --version` (should work without errors)

---

## Phase 3: Set Up Project Structure

### Step 8: Create Project Directory
1. Open Command Prompt
2. Navigate to desktop: `cd Desktop`
3. Create folder: `mkdir projects_3`
4. Navigate: `cd projects_3`
5. Create project folder: `mkdir gcp_demo_project`
6. Navigate: `cd gcp_demo_project`

### Step 9: Create Python Virtual Environment
1. Install Python if not installed (download from python.org)
2. In Command Prompt: `python --version` (should be 3.11+)
3. Create venv: `python -m venv venv`
4. Activate venv: `venv\Scripts\activate`
5. Upgrade pip: `python -m pip install --upgrade pip`

---

## Phase 4: Set Up GCP Authentication and Permissions

### IAM Roles Overview
Your RAG pipeline needs specific IAM roles to function properly. Each service account requires different permissions:

**Cloud Build Service Account (cloud-build-sa)**: Handles the CI/CD deployment process
- `Cloud Build Service Account` (basic): Allows Cloud Build to run
- `Artifact Registry Writer`: Upload Docker images to container registry
- `Cloud Run Admin`: Deploy and manage Cloud Run services
- `Storage Admin`: Upload documents to Cloud Storage buckets
- `Secret Manager Secret Accessor`: Access API keys during build
- `Vertex AI User`: Use Gemini AI services
- `IAM Service Account User`: Act as other service accounts

**Cloud Run Service Account (compute service account)**: Runs your application
- `Secret Manager Secret Accessor`: Access Gemini API key at runtime
- `Storage Admin`: Read/write documents in Cloud Storage
- `Vertex AI User`: Use Gemini AI for embeddings and generation

Without these roles, you'll encounter permission errors during deployment or runtime.

### Step 10: Set Up Application Default Credentials
1. In Command Prompt (with venv active): `gcloud auth application-default login`
2. Browser opens - sign in with Google account
3. Grant permissions when prompted

### Step 11: Create Service Account for Cloud Build
1. In the GCP Console, click the hamburger menu (☰) in the top-left corner
2. Scroll down to "IAM & Admin" and click on it
3. In the left sidebar, click on "Service Accounts"
4. On the Service Accounts page, click the blue "+ CREATE SERVICE ACCOUNT" button
5. In the "Create service account" dialog:
   - Service account name: `cloud-build-sa`
   - Service account ID: (auto-filled)
   - Description: `Service account for Cloud Build deployments`
6. Click "CREATE AND CONTINUE"
7. On the "Grant this service account access to project" page:
   - Role: Search for and select "Cloud Build Service Account"
8. Click "DONE"

### Step 12: Grant IAM Permissions to Cloud Build Service Account

**Modern GCP Note**: Cloud Build no longer requires downloading service account JSON keys. You can use either:
- Your custom `cloud-build-sa` service account (recommended for fine-grained control)
- Cloud Build's built-in service account: `[PROJECT_NUMBER]@cloudbuild.gserviceaccount.com`

We'll use the custom service account approach for clarity:
1. In the GCP Console, click the hamburger menu (☰) in the top-left corner
2. Go to "IAM & Admin" → "IAM"
3. On the IAM page, scroll through the list or use the search box to find: `cloud-build-sa@rag-q-results.iam.gserviceaccount.com`
4. Click the pencil/edit icon on the right side of that row
5. In the "Edit permissions" panel that opens:
   - Click "ADD ANOTHER ROLE"
   - Search for and select each role, then click "ADD":

     **`Artifact Registry Writer`** (needed to push Docker images to Artifact Registry)
     **`Cloud Run Admin`** (needed to deploy and manage Cloud Run services)
     **`Secret Manager Secret Accessor`** (needed to access API keys from Secret Manager)
     **`Storage Admin`** (needed to upload and manage files in Cloud Storage)
     **`Vertex AI User`** (needed to use Gemini AI for embeddings and generation)
     **`IAM Service Account User`** (needed to act as other service accounts)

6. **Important**: Without these roles, you'll get permission errors during deployment:
   - "Permission 'artifactregistry.repositories.uploadArtifacts' denied" → Add Artifact Registry Writer
   - "Permission 'run.services.get' denied" → Add Cloud Run Admin + IAM Service Account User
   - "Permission denied on secret" → Add Secret Manager Secret Accessor
7. Click "SAVE"

### Step 13: Get Project Number
1. In GCP Console, go to "IAM & Admin" → "Settings"
2. Copy the "Project number" (something like 963953387804)

---

## Phase 5: Set Up Cloud Storage

### Step 14: Create Cloud Storage Bucket
1. In the GCP Console, click the hamburger menu (☰) in the top-left corner
2. Scroll down to "Storage" section and click on "Cloud Storage" → "Buckets"
3. On the Cloud Storage Buckets page, click the blue "+ CREATE" button
4. In the "Create bucket" form:
   - Name your bucket: `rag-q-results-rag-docs` (must be globally unique - add numbers if needed)
   - Location type: Select "Region"
   - Location: Choose `us-central1` from the dropdown
   - Storage class: Leave as "Standard"
   - Access control: Leave as "Uniform"
   - Protection: Leave as "None"
5. Click "CREATE"

---

## Phase 6: Set Up Secret Manager

### Step 15: Create Gemini API Key Secret
1. First, get your Gemini API key:
   - Go to: https://aistudio.google.com/app/apikey
   - Sign in and create a new API key
   - Copy the API key (it starts with "AIzaSy...")
2. Back in GCP Console, click the hamburger menu (☰) in the top-left corner
3. Scroll down to "Security" section and click on "Secret Manager"
4. On the Secret Manager page, click the blue "+ CREATE SECRET" button
5. In the "Create secret" form:
   - Name: `gemini-api-key`
   - Secret value: Paste your Gemini API key here
   - Regions: Click "Select regions" and choose `us-central1`
6. Click "CREATE SECRET"

### Step 16: Grant Secret Access to Cloud Run Service Account
1. In Secret Manager, find and click on your secret `gemini-api-key` from the list
2. In the secret details page, click on the "Permissions" tab
3. Click the "+ ADD" button (next to "Members")
4. In the "Add members to 'gemini-api-key'" dialog:
   - New members: Enter `963953387804-compute@developer.gserviceaccount.com` (replace with your project number from Step 14)
   - Role: Search for and select "Secret Manager Secret Accessor"
5. **Critical**: This step is required because Cloud Run uses the default compute service account to run your application. Without this permission, your app will fail with: "Permission denied on secret: projects/xxx/secrets/gemini-api-key/versions/latest"
6. Click "SAVE"

---

## Phase 7: Set Up Artifact Registry

### Step 17: Create Artifact Registry Repository
1. In the GCP Console, click the hamburger menu (☰) in the top-left corner
2. Scroll down to "CI/CD" section and click on "Artifact Registry"
3. On the Artifact Registry page, click the blue "+ CREATE REPOSITORY" button
4. In the "Create repository" form:
   - Name: `rag-repo`
   - Format: Select "Docker"
   - Location: Choose `us-central1` from the dropdown
   - Description: `Docker images for RAG pipeline`
5. Click "CREATE"

---

## Phase 8: Set Up Cloud Build Trigger

### Step 18: Enable Cloud Build API (if not already done)
1. Go to "APIs & Services" → "Library"
2. Search "Cloud Build API"
3. Enable it

### Step 19: Create Cloud Build Trigger
1. In the GCP Console, click the hamburger menu (☰) in the top-left corner
2. Go to "CI/CD" section and click on "Cloud Build" → "Triggers"
3. On the Triggers page, click the blue "+ CREATE TRIGGER" button
4. In the "Create trigger" form:
   - Name: `rag-pipeline-deploy`
   - Event: Select "Manual invocation"
   - Source: If you created a GitHub repo in Step 25, select "Connect to GitHub", otherwise select "Upload source code"
   - Repository: Select your GitHub repo or upload local files
   - Branch: `main` (or your default branch)
   - Configuration: Select "Cloud Build configuration file"
   - Cloud Build configuration file location: `cloudbuild.yaml`
   - Service account: `cloud-build-sa@rag-q-results.iam.gserviceaccount.com`

**Alternative**: Use Cloud Build's built-in service account (simpler setup):
   - Service account: `[YOUR_PROJECT_NUMBER]@cloudbuild.gserviceaccount.com`
   - Make sure this built-in account has the required IAM roles (Artifact Registry Writer, Cloud Run Admin, Storage Admin, Secret Manager Secret Accessor, Vertex AI User)

5. Click "CREATE"

---

## Phase 9: Local Development Setup

### Step 20: Create .env File
1. In your project folder, create file `.env`
2. Add these lines:
```
GOOGLE_API_KEY=your-gemini-api-key-here
GOOGLE_CLOUD_PROJECT=rag-q-results
GCS_BUCKET_NAME=rag-q-results-rag-docs
GCP_REGION=us-central1
CHROMA_PERSIST_DIR=./chroma_db
```

### Step 21: Install Python Dependencies
1. In Command Prompt (venv active): `pip install -r requirements.txt`

### Step 22: Test Local Development
1. Run: `python -c "from app.main import app; print('Import successful')"`
2. If no errors, local setup is working

---

## Phase 9.5: Set Up Git Version Control

**Note**: This section covers pushing your existing local code to GitHub for CI/CD integration.

### Step 23: Install Git
1. Download Git from: https://git-scm.com/download/win
2. Run the installer
3. During installation, use default settings
4. After installation, open Command Prompt and run: `git --version`

### Step 24: Initialize Git Repository
1. In Command Prompt, navigate to your project: `cd Desktop\projects_3\gcp_demo_project`
2. Initialize Git: `git init`
3. Set up your Git identity:
   - `git config --global user.name "Your Name"`
   - `git config --global user.email "your.email@gmail.com"`

### Step 25: Create GitHub Repository (Optional but Recommended)
1. Go to https://github.com and sign in
2. Click the "+" icon → "New repository"
3. Repository name: `gcp-rag-pipeline` (or your choice)
4. Description: `RAG Pipeline using GCP services`
5. Keep it Public or Private (your choice)
6. **DO NOT** initialize with README (we already have one)
7. Click "Create repository"


### Step 26: Connect Local Repository to GitHub
1. Copy the repository URL from GitHub (looks like: https://github.com/yourusername/gcp-rag-pipeline.git)
2. In Command Prompt: `git remote add origin https://github.com/yourusername/gcp-rag-pipeline.git`
3. Verify: `git remote -v`

### Step 27: Stage and Commit Files
1. Add all files: `git add .`
2. Check status: `git status` (should show files staged)
3. Commit: `git commit -m "Initial commit: GCP RAG Pipeline setup"`
4. Push to GitHub: `git push -u origin main`

### Step 28: Verify GitHub Upload
1. Go to your GitHub repository URL
2. You should see all your files uploaded
3. The repository is now ready for Cloud Build integration

### Additional Git Commands for Ongoing Development
- **Check status**: `git status`
- **Add specific files**: `git add filename.py`
- **Add all changes**: `git add .`
- **Commit changes**: `git commit -m "Your commit message"`
- **Push changes**: `git push origin main`
- **Pull latest changes**: `git pull origin main`
- **Create new branch**: `git checkout -b feature-branch-name`
- **Switch branches**: `git checkout main`
- **Merge branches**: `git merge feature-branch-name`

### Common Git Issues and Solutions
- **"Permission denied" when pushing**: Make sure you have the correct repository URL and permissions
- **"fatal: remote origin already exists"**: Run `git remote remove origin` then add again
- **"Your branch is ahead of 'origin/main'"**: Run `git push` to sync changes
- **"error: src refspec main does not match any"**: Your local branch is named "master" but you're pushing to "main". Run:
  ```bash
  git branch -m master main  # Rename local branch to main
  git push -u origin main    # Push the renamed branch
  ```
- **Lost changes**: Use `git log` to see history, `git checkout <commit-hash>` to revert

---

## Phase 10: Build and Deploy

### Step 30: Build Docker Image Locally (Optional)
1. In Command Prompt: `docker build -t rag-pipeline .`
2. Test: `docker run -p 8080:8080 --env-file .env rag-pipeline`
3. Visit http://localhost:8080 to test

### Step 31: Deploy via Cloud Build
1. In Command Prompt: `gcloud builds submit --config=cloudbuild.yaml .`
2. Wait for build to complete (5-10 minutes)
3. Check build logs: `gcloud builds log BUILD_ID` (replace BUILD_ID)

### Step 32: Verify Cloud Run Deployment
1. In the GCP Console, click the hamburger menu (☰) in the top-left corner
2. Scroll down to "Serverless" section and click on "Cloud Run"
3. On the Cloud Run page, find your service named `rag-pipeline`
4. Click on the service name to open its details
5. On the service details page, find the "URL" field (it will look like: https://rag-pipeline-xxxxxx-us-central1.run.app)
6. Click on the URL or copy and paste it into your browser
7. Test the application by uploading a document and asking questions

---

## Phase 11: Monitoring and Troubleshooting

### Step 33: Check Application Logs
1. In the GCP Console, go to "Serverless" → "Cloud Run"
2. Click on your `rag-pipeline` service
3. In the service details page, click on the "Logs" tab
4. Look for any error messages or check that the application started successfully
5. You can filter logs by severity level using the dropdown

### Step 34: Check Cloud Build Logs
1. In the GCP Console, go to "CI/CD" → "Cloud Build" → "Builds"
2. Find your recent build in the list (most recent at the top)
3. Click on the build to open its details
4. Check the build logs for any errors or warnings

### Step 35: Verify Permissions
If you get permission errors:
1. Go to "IAM & Admin" → "IAM"
2. Check that service accounts have correct roles
3. Add missing roles if needed

### Step 36: Test the Application
1. Visit your Cloud Run URL
2. Upload a PDF document
3. Ask a question about the document
4. Check that responses work

---

## Common Issues and Solutions

### Issue: "Permission 'artifactregistry.repositories.uploadArtifacts' denied"
**Solution**: Add "Artifact Registry Writer" role to the cloud-build-sa service account
**When it happens**: During the Docker push step in Cloud Build
**How to fix**: Go to IAM → find cloud-build-sa → Edit → Add "Artifact Registry Writer" role

### Issue: "Permission 'run.services.get' denied"
**Solution**: Add "Cloud Run Admin" and "IAM Service Account User" roles to cloud-build-sa
**When it happens**: During Cloud Run deployment step in Cloud Build
**How to fix**: Go to IAM → find cloud-build-sa → Edit → Add both roles

### Issue: "Permission denied on secret: projects/xxx/secrets/gemini-api-key/versions/latest"
**Solution**: Grant "Secret Manager Secret Accessor" role to the Cloud Run service account
**When it happens**: When the deployed app tries to access the Gemini API key
**How to fix**: Go to Secret Manager → gemini-api-key → Permissions → Add the compute service account (project-number-compute@developer.gserviceaccount.com)

### Issue: "Permission denied" for Cloud Storage
**Solution**: Add "Storage Admin" role to cloud-build-sa
**When it happens**: When the app tries to upload documents to Cloud Storage
**How to fix**: Go to IAM → find cloud-build-sa → Edit → Add "Storage Admin" role

### Issue: Build fails with Docker errors
**Solution**: Ensure Docker Desktop is running and WSL integration is enabled

### Issue: Application won't start
**Solution**: Check Cloud Run logs for Python import errors or missing environment variables

---

## IAM Roles Summary

| Service Account | Role | Purpose | When Needed | Error If Missing |
|---|---|---|---|---|
| **cloud-build-sa** | Cloud Build Service Account | Basic Cloud Build operations | Always | Build fails |
| **cloud-build-sa** | Artifact Registry Writer | Push Docker images | During build | "artifactregistry.repositories.uploadArtifacts denied" |
| **cloud-build-sa** | Cloud Run Admin | Deploy services | During deploy | "run.services.get denied" |
| **cloud-build-sa** | Storage Admin | Upload to GCS | During build/app runtime | GCS upload fails |
| **cloud-build-sa** | Secret Manager Secret Accessor | Access secrets | During build | Secret access denied |
| **cloud-build-sa** | Vertex AI User | Use Gemini AI | During build/app runtime | AI API calls fail |
| **cloud-build-sa** | IAM Service Account User | Act as other accounts | During deploy | Permission escalation errors |
| **Compute SA** | Secret Manager Secret Accessor | Access Gemini key | App runtime | "Permission denied on secret" |
| **Compute SA** | Storage Admin | Read/write GCS | App runtime | Document upload/download fails |
| **Compute SA** | Vertex AI User | Use Gemini AI | App runtime | AI responses fail |

---

## Final Verification Steps

1. **Cloud Storage**: Check that documents are uploaded to your bucket
2. **Cloud Run**: Verify the service is running and accessible
3. **Secret Manager**: Confirm the API key is accessible
4. **Artifact Registry**: Check that Docker images are stored
5. **Cloud Build**: Verify builds complete successfully
6. **Application**: Test upload and query functionality

---

## Cost Monitoring

1. Go to "Billing" → "Budgets & alerts"
2. Create a budget for $10/month
3. Set up email alerts

**Expected costs**:
- Cloud Run: ~$0-5/month (depending on usage)
- Cloud Storage: ~$0.02/GB/month
- Secret Manager: Free
- Cloud Build: Free tier available
- Vertex AI: Pay per API call (~$0.001 per request)

---

## Backup and Security

1. **Environment Variables**: Never commit .env files
2. **Service Account Keys**: Never commit JSON keys to Git
3. **API Keys**: Always use Secret Manager in production
4. **IAM**: Use principle of least privilege
5. **Monitoring**: Set up logging and monitoring alerts

Your RAG pipeline is now fully set up and running on Google Cloud Platform!
