# Deploying Iris ML Model to Microsoft Azure ML Studio

This guide provides step-by-step instructions for deploying your MLflow-tracked Iris classification model to Microsoft Azure Machine Learning Studio for production use.

## Prerequisites

Before starting, ensure you have:
- An Azure account with an active subscription
- Azure CLI installed locally
- Docker installed on your machine
- Your trained MLflow model (from running `example1_train.py`)
- Python environment with Azure ML SDK

## Step 1: Set Up Azure Environment

### 1.1 Install Azure CLI and ML Extension
```bash
# macOS
brew install azure-cli

# Or download from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Install Azure ML extension
az extension add -n ml
```

### 1.2 Login and Set Up Subscription
```bash
# Login to Azure
az login

# List subscriptions
az account list --output table

# Set your subscription (replace with your subscription ID)
az account set --subscription "your-subscription-id"

# Set environment variables
export SUBSCRIPTION_ID="your-subscription-id"
export RESOURCE_GROUP="iris-ml-rg"
export WORKSPACE_NAME="iris-ml-workspace"
export LOCATION="eastus"
```

### 1.3 Create Resource Group and ML Workspace
```bash
# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure ML workspace
az ml workspace create \
    --name $WORKSPACE_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION
```

### 1.4 Install Azure ML SDK
```bash
# Install Azure ML SDK in your virtual environment
"/Users/arun/Documents/RamSELabs/Corporate Training/Course Materials/cognizant/mlops-lab1/.venv/bin/python" -m pip install azure-ai-ml azure-identity azure-ml-mlflow
```

## Step 2: Prepare Your Model for Azure ML

### 2.1 Create Azure ML Configuration

Create `azure-ml-config.yaml`:

```yaml
subscription_id: "your-subscription-id"
resource_group: "iris-ml-rg"
workspace_name: "iris-ml-workspace"
```

### 2.2 Create Model Registration Script

Create `register_model.py`:

```python
import os
import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
import yaml

def load_config():
    """Load Azure ML configuration"""
    with open('azure-ml-config.yaml', 'r') as file:
        return yaml.safe_load(file)

def find_latest_mlflow_model():
    """Find the latest MLflow model in mlruns directory"""
    mlruns_path = "./mlruns"
    experiments = [d for d in os.listdir(mlruns_path) if d.isdigit()]
    if experiments:
        exp_path = os.path.join(mlruns_path, experiments[0])
        runs = [d for d in os.listdir(exp_path) if len(d) == 32]  # MLflow run IDs are 32 chars
        if runs:
            latest_run = runs[-1]  # Get the latest run
            model_path = os.path.join(exp_path, latest_run, "artifacts", "model")
            if os.path.exists(model_path):
                return model_path, latest_run
    return None, None

def register_model():
    """Register MLflow model to Azure ML"""
    # Load configuration
    config = load_config()
    
    # Initialize Azure ML client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name'],
    )
    
    # Find latest MLflow model
    model_path, run_id = find_latest_mlflow_model()
    if not model_path:
        print("No MLflow model found!")
        return None
    
    print(f"Found model at: {model_path}")
    print(f"Run ID: {run_id}")
    
    # Register the model
    model = Model(
        path=model_path,
        name="iris-classification-model",
        description="Iris species classification using Logistic Regression trained with MLflow",
        version=run_id[:8],  # Use first 8 chars of run ID as version
        tags={"framework": "scikit-learn", "algorithm": "logistic-regression"}
    )
    
    registered_model = ml_client.models.create_or_update(model)
    print(f"Model registered: {registered_model.name}:{registered_model.version}")
    return registered_model

if __name__ == "__main__":
    register_model()
```

### 2.3 Create Scoring Script

Create `score.py`:

```python
import json
import pickle
import numpy as np
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import logging

def init():
    """Initialize the model for scoring"""
    global model
    
    # Get the model path from environment variable
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'iris-classification-model')
    
    try:
        # Load the MLflow model
        model = mlflow.sklearn.load_model(model_path)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise

def run(raw_data):
    """Run prediction on input data"""
    try:
        # Parse the input data
        data = json.loads(raw_data)
        
        # Handle different input formats
        if 'data' in data:
            input_data = np.array(data['data'])
        elif 'features' in data:
            input_data = np.array(data['features'])
        else:
            # Assume direct feature array
            input_data = np.array(data)
        
        # Ensure correct shape
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        # Make predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        # Map predictions to class names
        class_names = ['setosa', 'versicolor', 'virginica']
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "prediction": int(pred),
                "predicted_class": class_names[pred],
                "probabilities": {
                    class_names[j]: float(prob[j]) for j in range(len(class_names))
                }
            })
        
        return json.dumps(results)
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return json.dumps({"error": str(e)})
```

### 2.4 Create Environment Configuration

Create `conda-env.yaml`:

```yaml
name: iris-model-env
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
    - mlflow==3.4.0
    - scikit-learn==1.7.2
    - numpy==2.3.3
    - pandas==2.3.3
    - azure-ai-ml
    - azure-identity
```

### 2.5 Create Deployment Configuration

Create `deploy_model.py`:

```python
import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

def load_config():
    """Load Azure ML configuration"""
    with open('azure-ml-config.yaml', 'r') as file:
        return yaml.safe_load(file)

def deploy_model():
    """Deploy model to Azure ML online endpoint"""
    # Load configuration
    config = load_config()
    
    # Initialize Azure ML client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name'],
    )
    
    # Create or get endpoint
    endpoint_name = "iris-classification-endpoint"
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Endpoint for iris classification model",
        auth_mode="key",
    )
    
    try:
        endpoint_result = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"Endpoint created: {endpoint_result.name}")
    except Exception as e:
        print(f"Endpoint might already exist: {e}")
        endpoint_result = ml_client.online_endpoints.get(endpoint_name)
    
    # Create environment
    environment = Environment(
        name="iris-model-environment",
        description="Environment for iris classification model",
        conda_file="conda-env.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    
    env_result = ml_client.environments.create_or_update(environment)
    print(f"Environment created: {env_result.name}:{env_result.version}")
    
    # Get the latest model
    model = ml_client.models.get("iris-classification-model", label="latest")
    print(f"Using model: {model.name}:{model.version}")
    
    # Create deployment
    deployment = ManagedOnlineDeployment(
        name="iris-deployment",
        endpoint_name=endpoint_name,
        model=model,
        environment=env_result,
        code_configuration=CodeConfiguration(
            code="./",
            scoring_script="score.py"
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )
    
    deployment_result = ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Deployment created: {deployment_result.name}")
    
    # Set traffic to 100% for this deployment
    endpoint_result.traffic = {"iris-deployment": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint_result).result()
    print("Traffic allocation updated")
    
    return endpoint_result, deployment_result

if __name__ == "__main__":
    deploy_model()
```

## Step 3: Register and Deploy Your Model

### 3.1 Update Configuration File
```bash
# Update azure-ml-config.yaml with your actual values
sed -i '' "s/your-subscription-id/$SUBSCRIPTION_ID/g" azure-ml-config.yaml
```

### 3.2 Register the Model
```bash
# Run the model registration script
"/Users/arun/Documents/RamSELabs/Corporate Training/Course Materials/cognizant/mlops-lab1/.venv/bin/python" register_model.py
```

### 3.3 Deploy the Model
```bash
# Run the deployment script
"/Users/arun/Documents/RamSELabs/Corporate Training/Course Materials/cognizant/mlops-lab1/.venv/bin/python" deploy_model.py
```

## Step 4: Test Your Deployment

### 4.1 Create Test Script

Create `test_endpoint.py`:

```python
import json
import yaml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def load_config():
    """Load Azure ML configuration"""
    with open('azure-ml-config.yaml', 'r') as file:
        return yaml.safe_load(file)

def test_endpoint():
    """Test the deployed endpoint"""
    # Load configuration
    config = load_config()
    
    # Initialize Azure ML client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name'],
    )
    
    # Test data
    test_data = {
        "data": [
            [5.1, 3.5, 1.4, 0.2],  # Should predict setosa
            [6.2, 2.2, 4.5, 1.5],  # Should predict versicolor
            [6.3, 3.3, 6.0, 2.5]   # Should predict virginica
        ]
    }
    
    # Make prediction
    endpoint_name = "iris-classification-endpoint"
    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        request_file=None,
        deployment_name="iris-deployment"
    )
    
    print("Test successful!")
    print(f"Response: {response}")

def test_with_sample_data():
    """Test with sample data using REST API"""
    config = load_config()
    
    # Get endpoint URI and key
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name'],
    )
    
    endpoint = ml_client.online_endpoints.get("iris-classification-endpoint")
    print(f"Endpoint URI: {endpoint.scoring_uri}")
    
    # Get endpoint keys
    keys = ml_client.online_endpoints.get_keys("iris-classification-endpoint")
    print(f"Primary Key: {keys.primary_key}")
    
    # Example curl command
    print("\nExample curl command:")
    print(f"""curl -X POST "{endpoint.scoring_uri}" \\
  -H "Authorization: Bearer {keys.primary_key}" \\
  -H "Content-Type: application/json" \\
  -d '{{"data": [[5.1, 3.5, 1.4, 0.2]]}}'""")

if __name__ == "__main__":
    test_with_sample_data()
```

### 4.2 Run Tests
```bash
# Test the endpoint
"/Users/arun/Documents/RamSELabs/Corporate Training/Course Materials/cognizant/mlops-lab1/.venv/bin/python" test_endpoint.py
```

## Step 5: Monitor and Manage

### 5.1 View Endpoint Status
```bash
# List endpoints
az ml online-endpoint list --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME

# Get endpoint details
az ml online-endpoint show --name iris-classification-endpoint --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME
```

### 5.2 Monitor Deployments
```bash
# List deployments
az ml online-deployment list --endpoint-name iris-classification-endpoint --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME

# Get deployment logs
az ml online-deployment get-logs --name iris-deployment --endpoint-name iris-classification-endpoint --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME
```

### 5.3 Create Monitoring Script

Create `monitor_endpoint.py`:

```python
import yaml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import time

def load_config():
    """Load Azure ML configuration"""
    with open('azure-ml-config.yaml', 'r') as file:
        return yaml.safe_load(file)

def monitor_endpoint():
    """Monitor endpoint health and performance"""
    config = load_config()
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name'],
    )
    
    endpoint_name = "iris-classification-endpoint"
    
    while True:
        try:
            # Get endpoint status
            endpoint = ml_client.online_endpoints.get(endpoint_name)
            print(f"Endpoint Status: {endpoint.provisioning_state}")
            
            # Get deployment status
            deployments = ml_client.online_deployments.list(endpoint_name)
            for deployment in deployments:
                print(f"Deployment {deployment.name}: {deployment.provisioning_state}")
            
            time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            print("Monitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_endpoint()
```

## Step 6: CI/CD Integration

### 6.1 Update GitHub Actions Workflow

Add these steps to your `.github/workflows/ci-cd.yml`:

```yaml
  deploy-azure:
    runs-on: ubuntu-latest
    needs: build-and-deploy
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install Azure CLI
        run: |
          curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
          az extension add -n ml

      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install azure-ai-ml azure-identity azure-ml-mlflow

      - name: Register model to Azure ML
        run: |
          python register_model.py
        env:
          AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
          AZURE_RESOURCE_GROUP: ${{ secrets.AZURE_RESOURCE_GROUP }}
          AZURE_WORKSPACE_NAME: ${{ secrets.AZURE_WORKSPACE_NAME }}

      - name: Deploy to Azure ML
        run: |
          python deploy_model.py
        env:
          AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
          AZURE_RESOURCE_GROUP: ${{ secrets.AZURE_RESOURCE_GROUP }}
          AZURE_WORKSPACE_NAME: ${{ secrets.AZURE_WORKSPACE_NAME }}
```

### 6.2 Set Up GitHub Secrets

Add these secrets to your GitHub repository:

```bash
# Get Azure credentials for GitHub Actions
az ad sp create-for-rbac --name "github-actions-iris-ml" --role contributor \
    --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP \
    --sdk-auth
```

Add the following secrets to GitHub:
- `AZURE_CREDENTIALS`: Output from the above command
- `AZURE_SUBSCRIPTION_ID`: Your subscription ID
- `AZURE_RESOURCE_GROUP`: Your resource group name
- `AZURE_WORKSPACE_NAME`: Your workspace name

## Step 7: Advanced Features

### 7.1 A/B Testing with Multiple Deployments

Create `ab_testing.py`:

```python
import yaml
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential

def setup_ab_testing():
    """Set up A/B testing with traffic splitting"""
    config = load_config()
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name'],
    )
    
    endpoint_name = "iris-classification-endpoint"
    
    # Create second deployment (e.g., with different model version)
    model_v2 = ml_client.models.get("iris-classification-model", version="2")
    
    deployment_v2 = ManagedOnlineDeployment(
        name="iris-deployment-v2",
        endpoint_name=endpoint_name,
        model=model_v2,
        environment="iris-model-environment:1",
        code_configuration=CodeConfiguration(
            code="./",
            scoring_script="score.py"
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )
    
    ml_client.online_deployments.begin_create_or_update(deployment_v2).result()
    
    # Split traffic: 80% to v1, 20% to v2
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {
        "iris-deployment": 80,
        "iris-deployment-v2": 20
    }
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    print("A/B testing setup complete: 80% v1, 20% v2")
```

### 7.2 Model Performance Monitoring

Create `performance_monitor.py`:

```python
import json
import time
from datetime import datetime
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pandas as pd

def collect_performance_metrics():
    """Collect and analyze model performance metrics"""
    # This would typically integrate with Azure Monitor
    # and collect real prediction data for analysis
    
    print("Performance monitoring would track:")
    print("- Prediction latency")
    print("- Accuracy over time")
    print("- Data drift detection")
    print("- Model degradation")
```

## Step 8: Batch Inference

### 8.1 Create Batch Scoring Script

Create `batch_score.py`:

```python
import os
import json
import mlflow
import pandas as pd
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import BatchEndpoint, BatchDeployment
from azure.identity import DefaultAzureCredential

def create_batch_endpoint():
    """Create batch inference endpoint"""
    config = load_config()
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['subscription_id'],
        resource_group_name=config['resource_group'],
        workspace_name=config['workspace_name'],
    )
    
    # Create batch endpoint
    batch_endpoint = BatchEndpoint(
        name="iris-batch-endpoint",
        description="Batch endpoint for iris classification",
    )
    
    ml_client.batch_endpoints.begin_create_or_update(batch_endpoint).result()
    
    # Create batch deployment
    model = ml_client.models.get("iris-classification-model", label="latest")
    
    batch_deployment = BatchDeployment(
        name="iris-batch-deployment",
        endpoint_name="iris-batch-endpoint",
        model=model,
        environment="iris-model-environment:1",
        code_configuration=CodeConfiguration(
            code="./",
            scoring_script="batch_score_script.py"
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
        max_concurrency_per_instance=2,
        mini_batch_size=10,
        retry_settings={"max_retries": 3, "timeout": 300},
    )
    
    ml_client.batch_deployments.begin_create_or_update(batch_deployment).result()
    print("Batch endpoint and deployment created")
```

## Step 9: Clean Up Resources

### 9.1 Delete Deployments and Endpoints
```bash
# Delete online deployment
az ml online-deployment delete --name iris-deployment --endpoint-name iris-classification-endpoint --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --yes

# Delete online endpoint
az ml online-endpoint delete --name iris-classification-endpoint --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --yes

# Delete batch deployment and endpoint (if created)
az ml batch-deployment delete --name iris-batch-deployment --endpoint-name iris-batch-endpoint --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --yes
az ml batch-endpoint delete --name iris-batch-endpoint --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --yes
```

### 9.2 Delete Workspace and Resource Group
```bash
# Delete ML workspace
az ml workspace delete --name $WORKSPACE_NAME --resource-group $RESOURCE_GROUP --yes

# Delete resource group (this will delete all resources in it)
az group delete --name $RESOURCE_GROUP --yes
```

## Cost Optimization Tips

1. **Choose appropriate instance types**: Start with Standard_DS2_v2 and scale based on needs
2. **Use auto-scaling**: Configure min/max instances based on traffic patterns
3. **Monitor resource usage**: Use Azure Monitor to track costs and usage
4. **Consider spot instances**: For batch processing, use lower-cost spot instances
5. **Implement auto-shutdown**: Set up automatic scaling down during low usage periods

## Troubleshooting

### Common Issues:

1. **Authentication errors**: Ensure Azure CLI is logged in and has proper permissions
2. **Model registration failures**: Check MLflow artifacts path and format
3. **Deployment timeouts**: Increase timeout values or check resource availability
4. **Scoring script errors**: Validate script locally before deployment

### Useful Commands:

```bash
# Check workspace details
az ml workspace show --name $WORKSPACE_NAME --resource-group $RESOURCE_GROUP

# List all models
az ml model list --workspace-name $WORKSPACE_NAME --resource-group $RESOURCE_GROUP

# Get deployment logs
az ml online-deployment get-logs --name iris-deployment --endpoint-name iris-classification-endpoint --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME

# Test endpoint
az ml online-endpoint invoke --name iris-classification-endpoint --request-file test_data.json --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME
```

## Security Best Practices

1. **Use managed identity**: Enable system-assigned managed identity for secure access
2. **Network isolation**: Configure private endpoints for enhanced security
3. **Key rotation**: Regularly rotate endpoint keys
4. **Access control**: Use Azure RBAC for fine-grained permissions
5. **Audit logging**: Enable audit logs for compliance and monitoring

## Next Steps

1. **Implement MLOps pipelines** with Azure DevOps or GitHub Actions
2. **Set up data drift monitoring** with Azure ML dataset monitors
3. **Create automated retraining** workflows
4. **Implement model explainability** with Azure ML interpretability features
5. **Set up comprehensive monitoring** with Azure Monitor and Application Insights

This deployment strategy provides a production-ready solution for your Iris classification model on Microsoft Azure ML Studio with proper monitoring, scaling, and CI/CD integration.