# Deploying Iris ML Model to Google Vertex AI

This guide provides step-by-step instructions for deploying your MLflow-tracked Iris classification model to Google Vertex AI for production use.

## Prerequisites

Before starting, ensure you have:
- A Google Cloud Platform (GCP) account
- A GCP project with billing enabled
- Google Cloud SDK installed locally
- Docker installed on your machine
- Your trained MLflow model (from running `example1_train.py`)

## Step 1: Set Up Google Cloud Environment

### 1.1 Install Google Cloud SDK
```bash
# macOS
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### 1.2 Initialize and Authenticate
```bash
# Initialize gcloud
gcloud init

# Authenticate with your Google account
gcloud auth login

# Set your project (replace PROJECT_ID with your actual project ID)
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 1.3 Set Up Artifact Registry
```bash
# Create a repository for Docker images
gcloud artifacts repositories create iris-model-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Repository for Iris ML model"

# Configure Docker authentication
gcloud auth configure-docker us-central1-docker.pkg.dev
```

## Step 2: Prepare Your Model for Deployment

### 2.1 Create Model Serving Code

Create a new file `app.py` for serving your model:

```python
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
from google.cloud import storage
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global variable to store the loaded model
model = None

def load_model():
    """Load the MLflow model"""
    global model
    try:
        # Option 1: Load from MLflow artifacts (if you have MLflow server)
        # model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")
        
        # Option 2: Load from local mlruns directory
        # Find the latest run and load the model
        import os
        mlruns_path = "./mlruns"
        experiments = [d for d in os.listdir(mlruns_path) if d.isdigit()]
        if experiments:
            exp_path = os.path.join(mlruns_path, experiments[0])
            runs = [d for d in os.listdir(exp_path) if len(d) == 32]  # MLflow run IDs are 32 chars
            if runs:
                latest_run = runs[-1]  # Get the latest run
                model_path = os.path.join(exp_path, latest_run, "artifacts", "model")
                if os.path.exists(model_path):
                    model = mlflow.sklearn.load_model(model_path)
                    logging.info(f"Model loaded successfully from {model_path}")
                    return True
        
        logging.error("No model found in mlruns directory")
        return False
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Get input data from request
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        
        # Map prediction to class names
        class_names = ['setosa', 'versicolor', 'virginica']
        predicted_class = class_names[prediction]
        
        return jsonify({
            "prediction": int(prediction),
            "predicted_class": predicted_class,
            "probabilities": {
                class_names[i]: prob for i, prob in enumerate(probability)
            }
        })
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load the model on startup
    if load_model():
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port)
    else:
        logging.error("Failed to load model. Exiting.")
        exit(1)
```

### 2.2 Create Requirements File for Deployment

Create `requirements-deploy.txt`:

```
Flask==3.1.2
mlflow==3.4.0
scikit-learn==1.7.2
numpy==2.3.3
google-cloud-storage==2.10.0
gunicorn==23.0.0
```

### 2.3 Create Dockerfile

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code and model artifacts
COPY app.py .
COPY mlruns/ ./mlruns/

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "60", "app:app"]
```

## Step 3: Build and Push Docker Image

### 3.1 Build Docker Image
```bash
# Build the Docker image
docker build -t iris-model .

# Tag for Artifact Registry
docker tag iris-model us-central1-docker.pkg.dev/$PROJECT_ID/iris-model-repo/iris-model:latest
```

### 3.2 Push to Artifact Registry
```bash
# Push the image
docker push us-central1-docker.pkg.dev/$PROJECT_ID/iris-model-repo/iris-model:latest
```

## Step 4: Deploy to Vertex AI

### 4.1 Create Model Resource

Create a file `model-config.yaml`:

```yaml
displayName: "iris-classification-model"
description: "Iris species classification using Logistic Regression"
containerSpec:
  imageUri: "us-central1-docker.pkg.dev/PROJECT_ID/iris-model-repo/iris-model:latest"
  ports:
  - containerPort: 8080
  env:
  - name: "PORT"
    value: "8080"
```

### 4.2 Upload Model to Vertex AI
```bash
# Replace PROJECT_ID with your actual project ID
sed -i '' "s/PROJECT_ID/$PROJECT_ID/g" model-config.yaml

# Upload the model
gcloud ai models upload \
    --region=us-central1 \
    --display-name=iris-classification-model \
    --container-image-uri=us-central1-docker.pkg.dev/$PROJECT_ID/iris-model-repo/iris-model:latest \
    --container-ports=8080 \
    --container-predict-route=/predict \
    --container-health-route=/health
```

### 4.3 Create Endpoint
```bash
# Create an endpoint
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name=iris-endpoint
```

### 4.4 Deploy Model to Endpoint
```bash
# Get the model ID (from previous step output or list models)
MODEL_ID=$(gcloud ai models list --region=us-central1 --filter="displayName:iris-classification-model" --format="value(name)" | head -1)

# Get the endpoint ID
ENDPOINT_ID=$(gcloud ai endpoints list --region=us-central1 --filter="displayName:iris-endpoint" --format="value(name)" | head -1)

# Deploy model to endpoint
gcloud ai endpoints deploy-model $ENDPOINT_ID \
    --region=us-central1 \
    --model=$MODEL_ID \
    --display-name=iris-deployment \
    --machine-type=n1-standard-2 \
    --min-replica-count=1 \
    --max-replica-count=3 \
    --traffic-split=0=100
```

## Step 5: Test Your Deployment

### 5.1 Get Endpoint Details
```bash
# Get endpoint details
gcloud ai endpoints describe $ENDPOINT_ID --region=us-central1
```

### 5.2 Test with Sample Request

Create a test file `test_prediction.py`:

```python
from google.cloud import aiplatform
import json

# Initialize the client
aiplatform.init(project="your-project-id", location="us-central1")

# Get the endpoint
endpoint = aiplatform.Endpoint("projects/your-project-id/locations/us-central1/endpoints/ENDPOINT_ID")

# Sample iris features: [sepal_length, sepal_width, petal_length, petal_width]
test_instances = [
    {"features": [5.1, 3.5, 1.4, 0.2]},  # Should predict setosa
    {"features": [6.2, 2.2, 4.5, 1.5]},  # Should predict versicolor
    {"features": [6.3, 3.3, 6.0, 2.5]}   # Should predict virginica
]

# Make predictions
predictions = endpoint.predict(instances=test_instances)
print(json.dumps(predictions.predictions, indent=2))
```

### 5.3 Test with curl
```bash
# Get the endpoint URL
ENDPOINT_URL=$(gcloud ai endpoints describe $ENDPOINT_ID --region=us-central1 --format="value(deployedModels[0].serviceAccount)")

# Test prediction
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/us-central1/endpoints/$ENDPOINT_ID:predict" \
  -d '{
    "instances": [
      {"features": [5.1, 3.5, 1.4, 0.2]}
    ]
  }'
```

## Step 6: Monitor and Manage

### 6.1 Set Up Monitoring
```bash
# Enable monitoring
gcloud ai endpoints update $ENDPOINT_ID \
    --region=us-central1 \
    --enable-monitoring
```

### 6.2 View Logs
```bash
# View logs
gcloud logging read "resource.type=aiplatform.googleapis.com/Endpoint" --limit=50
```

### 6.3 Update Model (CI/CD Integration)

To integrate with your existing CI/CD pipeline, add these steps to your `.github/workflows/ci-cd.yml`:

```yaml
- name: Authenticate to Google Cloud
  uses: google-github-actions/auth@v1
  with:
    credentials_json: ${{ secrets.GCP_SA_KEY }}

- name: Set up Cloud SDK
  uses: google-github-actions/setup-gcloud@v1

- name: Build and push Docker image
  run: |
    docker build -t iris-model .
    docker tag iris-model us-central1-docker.pkg.dev/${{ secrets.PROJECT_ID }}/iris-model-repo/iris-model:${{ github.sha }}
    docker push us-central1-docker.pkg.dev/${{ secrets.PROJECT_ID }}/iris-model-repo/iris-model:${{ github.sha }}

- name: Deploy to Vertex AI
  run: |
    # Update model with new image
    gcloud ai models upload \
      --region=us-central1 \
      --display-name=iris-classification-model-${{ github.sha }} \
      --container-image-uri=us-central1-docker.pkg.dev/${{ secrets.PROJECT_ID }}/iris-model-repo/iris-model:${{ github.sha }}
```

## Step 7: Clean Up Resources (Optional)

```bash
# Undeploy model from endpoint
gcloud ai endpoints undeploy-model $ENDPOINT_ID \
    --region=us-central1 \
    --deployed-model-id=DEPLOYED_MODEL_ID

# Delete endpoint
gcloud ai endpoints delete $ENDPOINT_ID --region=us-central1

# Delete model
gcloud ai models delete $MODEL_ID --region=us-central1
```

## Cost Optimization Tips

1. **Use appropriate machine types**: Start with `n1-standard-2` and scale based on traffic
2. **Set auto-scaling**: Configure min/max replicas based on expected load
3. **Monitor usage**: Use Cloud Monitoring to track prediction requests and costs
4. **Consider batch predictions**: For large datasets, use Vertex AI Batch Prediction instead of online endpoints

## Troubleshooting

### Common Issues:

1. **Docker build failures**: Ensure all dependencies are in requirements-deploy.txt
2. **Model loading errors**: Check MLflow artifacts are copied correctly
3. **Endpoint deployment timeout**: Increase machine type or check container startup time
4. **Permission errors**: Ensure service account has necessary IAM roles

### Useful Commands:

```bash
# Check deployment status
gcloud ai endpoints describe $ENDPOINT_ID --region=us-central1

# View model versions
gcloud ai models list --region=us-central1

# Check logs
gcloud logging read "resource.type=aiplatform.googleapis.com/Endpoint" --limit=10
```

## Next Steps

1. **Implement A/B testing** with traffic splitting
2. **Set up model monitoring** for data drift detection
3. **Create automated retraining** pipelines
4. **Implement model versioning** strategy
5. **Add comprehensive testing** before deployment

This deployment strategy provides a production-ready solution for your Iris classification model on Google Vertex AI with proper monitoring, scaling, and CI/CD integration.