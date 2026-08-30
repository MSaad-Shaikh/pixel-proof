# ?? PixelProof ? Hackathon Public Deployment Guide

Your project is now containerized and deployment-ready!

---

## ??? What We Configured
1. **Dockerfile**: Builds on python:3.11-slim, installs required Linux libraries (libgl1, libglib2.0-0, 	esseract-ocr, curl), PyTorch CPU wheels, and dependencies.
2. **.dockerignore**: Keeps the Docker build context lean by omitting caches, tests, and temporary files.
3. **PORT dynamic binding**: Streamlit binds to --server.port= --server.address=0.0.0.0 so it works seamlessly on Google Cloud Run, Hugging Face Spaces, Render, and Railway.
4. **Health Check**: Configured HEALTHCHECK checking Streamlit's native /_stcore/health endpoint.
5. **.streamlit/config.toml**: Configured dark theme, headless mode, and 25MB maximum upload limit.

---

## ?? Recommended Option 1: Hugging Face Spaces (100% Free, Easiest for Hackathons)
> **Why choose this?** Completely free, no credit card required, instant deployment, provides a clean public URL (e.g., https://huggingface.co/spaces/your-username/pixelproof), and supports Docker directly.

### Steps:
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
3. Fill in:
   - **Space name**: orderguard-ai
   - **Space SDK**: Select **Docker** -> **Blank**.
   - **Space Hardware**: Free (CPU basic - 2 vCPU, 16GB RAM).
4. Click **Create Space**.
5. Push your code:
   - You can upload files directly via the Hugging Face web UI (**Files** -> **Add file** -> **Upload files**), OR push via Git:
     `ash
     git remote add space https://huggingface.co/spaces/<your-username>/pixelproof
     git push space main
     `
6. Hugging Face will automatically detect the Dockerfile, build it, and launch your live public app!

---

## ?? Option 2: Render (Free Web Service)
> **Why choose this?** Free tier, automatic deployment on GitHub pushes, easy custom domain.

### Steps:
1. Push this project to a GitHub repository.
2. Sign up / Log in at [render.com](https://render.com).
3. Click **New +** -> **Web Service**.
4. Connect your GitHub repository.
5. Render will automatically detect 
ender.yaml or you can select:
   - **Environment**: Docker
   - **Plan**: Free
6. Click **Create Web Service**.
7. Render will build the Docker container and give you a public URL (e.g. https://pixelproof.onrender.com).

---

## ?? Option 3: Google Cloud Run (via Web Console)
> **Why choose this?** Highly scalable Google Cloud infrastructure. (Requires a Google Cloud account with billing enabled).

### Steps:
1. Push your repository to GitHub.
2. Go to the [Google Cloud Console - Cloud Run](https://console.cloud.google.com/run).
3. Click **Create Service**.
4. Choose **Continuously deploy from a repository** and click **Set up with Cloud Build**.
5. Select your GitHub repository and branch (main).
6. Under **Build Type**, select **Dockerfile** (Source location: /Dockerfile).
7. Under **Authentication**, select **Allow unauthenticated invocations** (so judges/public can view your app).
8. Under **Container, Networking, Security**:
   - Set **Memory** to 2 GiB or 4 GiB.
   - Set **CPU** to 1 or 2.
   - Container port: 8501 (or leave default $PORT).
9. Click **Create**. Cloud Run will build and provide a live https://...-uc.a.run.app link.
