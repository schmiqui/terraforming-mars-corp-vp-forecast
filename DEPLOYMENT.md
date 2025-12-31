# Deployment Guide

This guide covers multiple deployment options for the Terraforming Mars VP Predictor web app.

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Platforms](#cloud-platforms)
   - [Render](#render-recommended)
   - [Railway](#railway)
   - [Heroku](#heroku)
   - [AWS](#aws)
   - [Google Cloud Platform](#google-cloud-platform)
4. [VPS Deployment](#vps-deployment)

---

## Local Development

For local testing and development:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python web/run_server.py

# Or with uvicorn directly
uvicorn web.backend:app --reload --host 0.0.0.0 --port 8000
```

Access at: `http://localhost:8000`

---

## Docker Deployment

### Option 1: Using Dockerfile (Recommended)

1. **Build the image:**
   ```bash
   docker build -t terraforming-mars-vp .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 terraforming-mars-vp
   ```

3. **Run with volume for model persistence:**
   ```bash
   docker run -p 8000:8000 -v $(pwd)/web/model.pkl:/app/web/model.pkl terraforming-mars-vp
   ```

### Option 2: Using Docker Compose

```bash
docker-compose up -d
```

---

## Cloud Platforms

### Render (Recommended - Free Tier Available)

**Steps:**

1. **Create a Render account** at https://render.com

2. **Create a new Web Service:**
   - Connect your GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn web.backend:app --host 0.0.0.0 --port $PORT`
   - Environment: `Python 3`

3. **Set Environment Variables (if needed):**
   - `PYTHON_VERSION=3.11.9` (must be full version: major.minor.patch)

4. **Deploy!**

**Pros:**
- Free tier available
- Automatic HTTPS
- Easy GitHub integration
- Automatic deployments

**Note:** Model file (`web/model.pkl`) will need to be regenerated on first deploy or stored in persistent storage.

---

### Railway

**Steps:**

1. **Install Railway CLI:**
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Initialize project:**
   ```bash
   railway init
   ```

3. **Deploy:**
   ```bash
   railway up
   ```

**Or use Railway Dashboard:**
- Connect GitHub repo
- Railway auto-detects Python
- Set start command: `uvicorn web.backend:app --host 0.0.0.0 --port $PORT`

**Pros:**
- Simple deployment
- Free tier available
- Good for small projects

---

### Heroku

**Steps:**

1. **Install Heroku CLI** and login:
   ```bash
   heroku login
   ```

2. **Create app:**
   ```bash
   heroku create your-app-name
   ```

3. **Set buildpacks:**
   ```bash
   heroku buildpacks:set heroku/python
   ```

4. **Deploy:**
   ```bash
   git push heroku main
   ```

**Note:** Heroku requires a `Procfile` (see below).

**Pros:**
- Well-established platform
- Good documentation
- Add-ons available

**Cons:**
- No free tier anymore (paid plans only)

---

### AWS

#### Option 1: AWS Elastic Beanstalk (Easiest)

1. **Install EB CLI:**
   ```bash
   pip install awsebcli
   ```

2. **Initialize:**
   ```bash
   eb init -p python-3.11 terraforming-mars-vp
   eb create terraforming-mars-vp-env
   ```

3. **Deploy:**
   ```bash
   eb deploy
   ```

#### Option 2: AWS EC2

1. **Launch EC2 instance** (Ubuntu recommended)
2. **SSH into instance**
3. **Install dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip nginx
   pip3 install -r requirements.txt
   ```

4. **Set up systemd service** (see VPS Deployment section)
5. **Configure Nginx** as reverse proxy

#### Option 3: AWS Lambda + API Gateway

Requires serverless framework or AWS SAM. More complex but cost-effective for low traffic.

---

### Google Cloud Platform

#### Option 1: Cloud Run (Recommended)

1. **Install gcloud CLI**

2. **Build and deploy:**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/terraforming-mars-vp
   gcloud run deploy --image gcr.io/PROJECT-ID/terraforming-mars-vp --platform managed
   ```

#### Option 2: App Engine

1. **Create `app.yaml`:**
   ```yaml
   runtime: python311
   entrypoint: uvicorn web.backend:app --host 0.0.0.0 --port 8080
   ```

2. **Deploy:**
   ```bash
   gcloud app deploy
   ```

---

## VPS Deployment

### Using systemd (Ubuntu/Debian)

1. **Create systemd service file** `/etc/systemd/system/terraforming-mars-vp.service`:
   ```ini
   [Unit]
   Description=Terraforming Mars VP Predictor
   After=network.target

   [Service]
   User=www-data
   WorkingDirectory=/path/to/terraforming-mars-corp-vp-forecast
   Environment="PATH=/usr/bin:/usr/local/bin"
   ExecStart=/usr/local/bin/uvicorn web.backend:app --host 0.0.0.0 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

2. **Enable and start:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable terraforming-mars-vp
   sudo systemctl start terraforming-mars-vp
   ```

3. **Set up Nginx reverse proxy** `/etc/nginx/sites-available/terraforming-mars-vp`:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

4. **Enable site:**
   ```bash
   sudo ln -s /etc/nginx/sites-available/terraforming-mars-vp /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

5. **Set up SSL with Let's Encrypt:**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

---

## Important Considerations

### Model Persistence

The trained model (`web/model.pkl`) needs to be:
- **Option 1:** Committed to git (if small enough)
- **Option 2:** Generated on first deploy via `/train` endpoint
- **Option 3:** Stored in cloud storage (S3, GCS, etc.) and downloaded on startup
- **Option 4:** Use persistent volumes (Docker/Railway)

### Environment Variables

You may want to set:
- `PYTHON_VERSION=3.11`
- `MODEL_PATH=/path/to/model.pkl` (if custom)
- `CSV_PATH=/path/to/games.csv` (if custom)

### Port Configuration

Most cloud platforms set `PORT` environment variable. Update `run_server.py` or use:
```bash
uvicorn web.backend:app --host 0.0.0.0 --port ${PORT:-8000}
```

### File Size Limits

- Some platforms have file size limits (e.g., Heroku 500MB slug limit)
- Model files can be large - consider compression or cloud storage

---

## Recommended Deployment Flow

1. **For Quick Testing:** Use Render or Railway (free tiers)
2. **For Production:** Use AWS/GCP with proper CI/CD
3. **For Learning:** Start with Docker locally, then try Render

---

## Troubleshooting

- **Port binding errors:** Make sure to use `0.0.0.0` not `127.0.0.1`
- **Model not found:** Train model on first deploy or include in deployment
- **Import errors:** Ensure all dependencies are in `requirements.txt`
- **CORS issues:** Already handled in backend, but check if deploying to custom domain

