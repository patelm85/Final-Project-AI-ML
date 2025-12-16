# Walmart Sales Forecasting MLOps Project

A comprehensive MLOps project for forecasting Walmart weekly sales using automated machine learning, model tracking, deployment, and drift monitoring.

## Project Overview

This project implements a complete MLOps pipeline for sales forecasting, including:

- Data cleaning and time-based train/validation/test splitting
- Automated model selection using H2O AutoML
- Manual training of top three models with MLflow tracking
- Remote MLflow tracking server setup (AWS EC2 + PostgreSQL + S3)
- FastAPI model serving with three prediction endpoints
- Model drift analysis using Evidently


## Dataset

The dataset contains Walmart store sales data from 2010-2012 with the following features:

- **Store**: Store identifier
- **Date**: Weekly date (DD-MM-YYYY format)
- **Weekly_Sales**: Target variable (weekly sales in USD)
- **Holiday_Flag**: Binary indicator for holiday weeks
- **Temperature**: Average temperature
- **Fuel_Price**: Fuel price
- **CPI**: Consumer Price Index
- **Unemployment**: Unemployment rate

## Project Structure

```
walmart-sales-forecasting-mlops/
├── data/
│   ├── raw/
│   │   └── Walmart.csv
│   └── processed/
│       ├── train.csv
│       ├── validate.csv
│       └── test.csv
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── automl/
│   │   ├── __init__.py
│   │   └── run_h2o_automl.py
│   ├── models/
│   │   ├── train_xgboost.py
│   │   ├── train_lightgbm.py
│   │   └── train_randomforest.py
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   └── drift/
│       └── drift_analysis.py
├── experiments/
├── mlflow/
├── reports/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Prerequisites

- Python 3.9 or higher
- UV package manager (for Python environment management)
- AWS Account (for EC2 and S3)
- Neon.tech account (for PostgreSQL database)
- Git

## Initial Setup

### Step 1: Install UV

If you don't have UV installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Clone Repository

```bash
git clone <repository-url>
cd walmart-sales-forecasting-mlops
```

### Step 3: Set Up Python Environment

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -e .
```
![Alt text for the image](/images/install.png "Optional title text")

### Step 4: Configure Environment Variables

Copy the example environment file and update with your credentials:

```bash
cp env.example .env
```

Edit `.env` with your actual values:

```bash
export MLFLOW_BACKEND_STORE_URI="postgresql://user:password@host:5432/mlflowdb?sslmode=require"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="s3://your-bucket-name/mlflow"
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="eu-north-1"
```

**Important**: Never commit `.env` file to Git. It's already in `.gitignore`.

Load environment variables:

```bash
source .env
```

### Step 5: Set Up Remote MLflow Infrastructure

Before running model training scripts, you need to set up the remote MLflow tracking server. This involves:

1. **Setting up Neon PostgreSQL database** - For MLflow backend storage
2. **Creating AWS S3 bucket** - For artifact storage
3. **Launching AWS EC2 instance** - For hosting MLflow server
4. **Configuring MLflow server** - Connect all components

**Detailed instructions** for infrastructure setup are provided in [`mlflow/SETUP_GUIDE.md`](mlflow/SETUP_GUIDE.md). Follow that guide to complete the cloud infrastructure setup before proceeding with model training.

**Quick reference** after setup:
- Your MLflow tracking URI will be: `http://YOUR-EC2-PUBLIC-IP:5000`
- Set it in your environment: `export MLFLOW_TRACKING_URI="http://YOUR-EC2-PUBLIC-IP:5000"`
- **Important:** If you encounter DNS rebinding errors (403 Invalid Host header), see the [Troubleshooting section](#mlflow-connection-issues) for detailed step-by-step fix instructions

### Step 6: Verify Data File

Ensure the raw data file is in place:

```bash
# Check if Walmart.csv exists
ls -lh data/raw/Walmart.csv
```

If the file is missing, place `Walmart.csv` in the `data/raw/` directory.

## Project Workflow

Follow these steps in order to complete the entire pipeline:

### Step 1: Data Cleaning and Time-Based Splitting

Clean the dataset and create time-based train/validation/test splits:

```bash
python src/data/make_dataset.py
```
![Alt text for the image](/images/make_dataset.png "Optional title text")

**What this does:**
- Loads and cleans the Walmart.csv dataset
- Handles missing values using median imputation
- Removes outliers using 3*IQR method
- Converts dates from DD-MM-YYYY to datetime format
- Normalizes text fields (strips whitespace)
- Encodes categorical variables
- Sorts data chronologically by date
- Splits into train (35%), validate (35%), and test (30%) sets
- Saves processed datasets to `data/processed/`:
  - `train.csv`
  - `validate.csv`
  - `test.csv`

**Expected output:**
- Cleaned datasets with date ranges printed
- Split statistics displayed
- CSV files saved in `data/processed/`

### Step 2: H2O AutoML Analysis

Run H2O AutoML on the training dataset to identify top-performing model types:

```bash
python src/automl/run_h2o_automl.py
```

![Alt text for the image](/images/run_h2o.png "Optional title text")

**What this does:**
- Trains multiple models using H2O AutoML (max_models=20)
- Generates a leaderboard ranked by RMSE
- Saves the leaderboard to `leaderboard.csv`
- Displays the top three model types/algorithms

**Note:** Based on the leaderboard results, you'll identify the top 3 model types to train manually in the next step. Common model types from H2O AutoML include:
- XGBoost (XGBoost)
- GBM (Gradient Boosting Machine)
- DRF (Distributed Random Forest)
- StackedEnsemble

### Step 3: Manual Model Training with MLflow Tracking

Train the three selected models and log everything to MLflow:

**Important:** Ensure your remote MLflow server is running and configured. See [`mlflow/SETUP_GUIDE.md`](mlflow/SETUP_GUIDE.md) if you haven't set it up yet.

**Before running training scripts, verify MLflow connection and load AWS credentials:**

```bash
# 1. Load environment variables (including AWS credentials for S3)
source .env
# OR if .env doesn't exist, create it from env.example and fill in your credentials

# 2. Set MLflow tracking URI (if not in .env)
export MLFLOW_TRACKING_URI="http://YOUR-EC2-PUBLIC-IP:5000"

# 3. Verify AWS credentials are loaded
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY
# Should show your credentials (not empty)

# 4. Test MLflow connection
curl http://YOUR-EC2-PUBLIC-IP:5000/health
# Should return: OK

# 5. Test with Python
python -c "import mlflow; mlflow.set_tracking_uri('http://YOUR-EC2-PUBLIC-IP:5000'); print('Connected!')"
# Should print: Connected!
```

**If you get DNS rebinding errors, see the [Troubleshooting section](#mlflow-connection-issues) below.**

**Once connection is verified, train each model:**

```bash
# Train each model
python src/models/train_xgboost.py
python src/models/train_lightgbm.py
python src/models/train_randomforest.py
```
Experiment: walmart-sales-forecast

View run in MLflow UI: http://56.228.2.60:5000/#/experiments/2/runs/9484be354a8b4c13972c3eefc3877737 

![Alt text for the image](/images/xboost_mflows.png "Optional title text")

View run in MLflow UI: http://56.228.2.60:5000/#/experiments/2/runs/8284d7d0da4e4cb0ac42e4094f854c93

![Alt text for the image](/images/lightbm_mflow.png "Optional title text")

View run in MLflow UI: http://56.228.2.60:5000/#/experiments/2/runs/f3ab586cfd534a1b9074b01f95e2ab0b

![Alt text for the image](/images/random_forest.png "Optional title text")

**What each training script does:**
- Loads cleaned training and validation datasets from `data/processed/`
- Trains the model with optimized hyperparameters
- Evaluates model performance on validation set
- Evaluates model performance on test set
- Logs to MLflow:
  - Parameters (hyperparameters, model type, etc.)
  - Metrics (RMSE, MAE, R² for both validation and test sets)
  - Artifacts (model files, plots, feature importance charts)
- Registers the model in MLflow Model Registry

**All models log to the same MLflow experiment:** `walmart-sales-forecast`

**Verify in MLflow UI:**
- Navigate to `http://YOUR-EC2-PUBLIC-IP:5000`
- Find the "walmart-sales-forecast" experiment
- Compare runs to identify the champion model

![Alt text for the image](/images/mflows.png "Optional title text")

### Step 4: Model Registry and Comparison

After all three models are trained, compare them in the MLflow UI:

1. **Access MLflow UI:**
   ```bash
   # Open in browser
   http://YOUR-EC2-PUBLIC-IP:5000
   ```

2. **Navigate to Experiment:**
   - Click on "walmart-sales-forecast" experiment
   - View all three model runs

![Alt text for the image](/images/mlflow_models.png "Optional title text")

3. **Compare Models:**
   - Select all three runs
   - Click "Compare" to view metrics side-by-side
   - Metrics to compare: RMSE, MAE, R²

4. **Identify Champion Model:**
   - Champion = model with lowest RMSE and MAE, highest R²
   - Document your findings

![Alt text for the image](/images/chart_view.png "Optional title text")

5. **Register Models:**
   - Register each model in the Model Registry
   - You can use one model name with three versions, or three separate model names

### Step 5: FastAPI Deployment

Deploy the trained models as a REST API for serving predictions:

```bash
# Start the FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
![Alt text for the image](/images/run_fastapi.png "Optional title text")

![Alt text for the image](/images/api_docs.png "Optional title text")

**API Endpoints:**
- `POST /predict_model1` - Predictions from model 1
- `POST /predict_model2` - Predictions from model 2
- `POST /predict_model3` - Predictions from model 3
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)

**Example Prediction Request:**

```bash
# Test all three models with the same data
curl -X POST "http://localhost:8000/predict_model1" -H "Content-Type: application/json" -d '{"store": 1, "date": "2012-11-01", "holiday_flag": 0, "temperature": 65.5, "fuel_price": 3.45, "cpi": 211.0, "unemployment": 7.5}'
```
![Alt text for the image](/images/api_xgboost.png "Optional title text")

```bash
curl -X POST "http://localhost:8000/predict_model2" -H "Content-Type: application/json" -d '{"store": 1, "date": "2012-11-01", "holiday_flag": 0, "temperature": 65.5, "fuel_price": 3.45, "cpi": 211.0, "unemployment": 7.5}'
```
![Alt text for the image](/images/api_lightgbm.png "Optional title text")

```bash
curl -X POST "http://localhost:8000/predict_model3" -H "Content-Type: application/json" -d '{"store": 1, "date": "2012-11-01", "holiday_flag": 0, "temperature": 65.5, "fuel_price": 3.45, "cpi": 211.0, "unemployment": 7.5}'
```
![Alt text for the image](/images/api_randomforest.png "Optional title text")

Single prediction:
```bash
curl -X POST "http://localhost:8000/predict_model1" \
  -H "Content-Type: application/json" \
  -d '{
    "store": 1,
    "date": "2012-11-01",
    "holiday_flag": 0,
    "temperature": 65.5,
    "fuel_price": 3.5,
    "cpi": 220.0,
    "unemployment": 7.5
  }'
```

Batch prediction:
```bash
curl -X POST "http://localhost:8000/predict_model1" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "store": 1,
      "date": "2012-11-01",
      "holiday_flag": 0,
      "temperature": 65.5,
      "fuel_price": 3.5,
      "cpi": 220.0,
      "unemployment": 7.5
    },
    {
      "store": 2,
      "date": "2012-11-08",
      "holiday_flag": 0,
      "temperature": 68.2,
      "fuel_price": 3.6,
      "cpi": 220.5,
      "unemployment": 7.4
    }
  ]'
```

**Expected Response:**
```json
{
  "predictions": [1645321.45],
  "model_name": "xgboost",
  "model_version": "1",
  "timestamp": "2025-12-08T12:00:00Z"
}
```

### Step 6: Model Drift Analysis

Monitor model performance over time using drift detection:

```bash
# Run drift analysis (uses scipy.stats - no additional installation needed)
python src/drift/drift_analysis.py
```

**What this does:**
- Uses the test set as reference data (baseline)
- Simulates production data from a newer timeframe (last 30% of test dataset)
- Analyzes using statistical tests (Kolmogorov-Smirnov and Mann-Whitney U tests):
  - **Data drift**: Detects changes in feature distributions
  - **Target drift**: Detects changes in Weekly_Sales distribution
- Generates:
  - **HTML report** in `reports/drift_report.html` (comprehensive visual report)
  - Statistical results in `reports/drift_results.json` (detailed metrics)
  - Visualization plot in `reports/drift_analysis.png` (distribution comparisons)

**View Results:**
```bash
# Open the HTML report (recommended - includes all results and visualizations)
open reports/drift_report.html
# or
xdg-open reports/drift_report.html  # Linux

# Or view individual files:
cat reports/drift_results.json  # JSON results
xdg-open reports/drift_analysis.png  # Visualization plot
```

**Report Contents:**
The HTML report (`drift_report.html`) includes:
- Executive summary with key metrics
- Target variable (Weekly_Sales) drift analysis
- Feature-by-feature drift analysis table
- Statistical test results (KS and Mann-Whitney U tests)
- Embedded visualization showing distribution comparisons
- Interpretation guide for understanding the results
- Recommendations for model retraining

## Remote MLflow Infrastructure Setup

This project uses a remote MLflow tracking server hosted on AWS EC2, with PostgreSQL backend (Neon.tech) and S3 artifact storage.

**Complete setup instructions** are provided in [`mlflow/SETUP_GUIDE.md`](mlflow/SETUP_GUIDE.md). That guide covers:

1. **Neon PostgreSQL Setup** - Creating database and getting connection string
2. **AWS S3 Bucket Creation** - Setting up artifact storage
3. **AWS EC2 Instance Launch** - Launching and configuring the server
4. **MLflow Server Configuration** - Installing and running MLflow on EC2
5. **Local Machine Configuration** - Connecting your training scripts

**Quick Checklist:**
- [ ] Neon PostgreSQL database created
- [ ] AWS S3 bucket created
- [ ] AWS EC2 instance launched and running
- [ ] MLflow server installed and running on EC2
- [ ] Security groups configured (port 5000 open)
- [ ] Environment variables configured (see `env.example`)
- [ ] MLflow UI accessible at `http://YOUR-EC2-IP:5000`

**After Setup:**
- Set `MLFLOW_TRACKING_URI` in your `.env` file
- Verify connection: `curl http://YOUR-EC2-IP:5000`
- Access MLflow UI in browser to confirm it's working

## Project Structure Details

```
walmart-sales-forecasting-mlops/
├── data/
│   ├── raw/
│   │   └── Walmart.csv              # Raw dataset (not committed to Git)
│   └── processed/                   # Generated datasets (not committed)
│       ├── train.csv
│       ├── validate.csv
│       └── test.csv
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py          # Data cleaning and splitting
│   ├── automl/
│   │   ├── __init__.py
│   │   └── run_h2o_automl.py        # H2O AutoML analysis
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_xgboost.py         # XGBoost model training
│   │   ├── train_lightgbm.py        # LightGBM model training
│   │   └── train_randomforest.py    # Random Forest model training
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application
│   │   └── schemas.py               # Pydantic schemas for API
│   └── drift/
│       ├── __init__.py
│       └── drift_analysis.py        # Model drift detection
├── mlflow/
│   ├── SETUP_GUIDE.md               # Infrastructure setup instructions
│   └── tracking_uri.txt             # MLflow tracking URI (generated)
├── experiments/                     # Local MLflow experiments (if used)
├── reports/
│   └── drift_report.html            # Generated drift analysis report
├── notebooks/                       # Notebooks (excluded from submission)
├── .env                             # Environment variables (not committed)
├── env.example                      # Example environment file
├── mlflow-key.pem                   # EC2 SSH key (not committed to Git)
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # UV project configuration
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## Dependencies

All dependencies are specified in `pyproject.toml` and `requirements.txt`.

**Key Dependencies:**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, h2o, xgboost, lightgbm
- **MLOps**: mlflow, boto3, psycopg2-binary
- **API**: fastapi, uvicorn, pydantic
- **Monitoring**: evidently
- **Visualization**: matplotlib, seaborn

**Install all dependencies:**
```bash
uv pip install -e .
```

## Environment Configuration

The project uses environment variables for configuration. Copy `env.example` to `.env` and fill in your values:

```bash
cp env.example .env
# Edit .env with your actual credentials
```

**Required Environment Variables:**
- `MLFLOW_BACKEND_STORE_URI` - PostgreSQL connection string from Neon
- `MLFLOW_DEFAULT_ARTIFACT_ROOT` - S3 bucket URI for artifacts
- `AWS_ACCESS_KEY_ID` - AWS IAM access key
- `AWS_SECRET_ACCESS_KEY` - AWS IAM secret key
- `AWS_DEFAULT_REGION` - AWS region (e.g., eu-north-1)

**Optional:**
- `MLFLOW_TRACKING_URI` - Can be set in code or environment

## Security Notes

1. **Never commit sensitive files to Git:**
   - `.env` file (contains credentials)
   - `*.pem` files (SSH keys)
   - Generated data files

2. **These are already in `.gitignore`:**
   - `.env`
   - `*.pem`
   - `data/processed/`
   - `mlflow/`
   - `experiments/`

3. **Best Practices:**
   - Use IAM roles instead of access keys when possible
   - Rotate credentials regularly
   - Restrict EC2 security groups to your IP only
   - Use least-privilege IAM policies

## Troubleshooting

### Data Processing Issues

**Problem:** Date parsing errors
```bash
# Solution: Check date format in Walmart.csv
head -5 data/raw/Walmart.csv
# Should be DD-MM-YYYY format
```

**Problem:** Missing data file
```bash
# Solution: Ensure Walmart.csv is in data/raw/
ls -lh data/raw/Walmart.csv
```

### MLflow Connection Issues

**Problem:** Cannot connect to MLflow server
```bash
# Check if server is running
curl http://YOUR-EC2-IP:5000

# Verify EC2 security group allows port 5000
# Check MLflow service status on EC2
ssh -i mlflow-key.pem ubuntu@YOUR-EC2-IP
sudo systemctl status mlflow
```

**Problem:** DNS Rebinding Error (403 Invalid Host header)

This error occurs when MLflow server rejects requests from IP addresses due to DNS rebinding protection. Follow these steps to fix it:

**Step 1: SSH into EC2 instance**
```bash
ssh -i mlflow-key.pem ubuntu@YOUR-EC2-IP
```

**Step 2: Create virtual environment and install MLflow (if not already done)**
```bash
# Create a virtual environment for MLflow (required for Ubuntu 22.04+)
python3 -m venv ~/mlflow-env

# Activate the virtual environment
source ~/mlflow-env/bin/activate

# Install MLflow and dependencies
pip install mlflow psycopg2-binary boto3

# Verify installation
python -m mlflow --version

# Deactivate (we'll use the venv in the service)
deactivate
```

**Step 3: Update MLflow service file**
```bash
sudo nano /etc/systemd/system/mlflow.service
```

**Replace the entire file with this configuration:**
```ini
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=ubuntu
Environment="MLFLOW_ALLOW_HOSTS=*"
EnvironmentFile=/home/ubuntu/.mlflow_env
ExecStart=/home/ubuntu/mlflow-env/bin/python -m mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts "*"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Key points:**
- `--allowed-hosts "*"` - This flag fixes the DNS rebinding error (required for MLflow 3.0+)
- `Environment="MLFLOW_ALLOW_HOSTS=*"` can help but the flag is more reliable
- Use full path to venv Python: `/home/ubuntu/mlflow-env/bin/python`
- Use `python -m mlflow server` instead of `mlflow server`

**Step 4: Restart the service**
```bash
sudo systemctl daemon-reload
sudo systemctl restart mlflow
sudo systemctl status mlflow
```

You should see `Active: active (running)` in green.

![Alt text for the image](/images/mflow_status.png "Optional title text")

**Step 5: Verify connection from local machine**
```bash
# In your local terminal (not EC2)
export MLFLOW_TRACKING_URI="http://YOUR-EC2-IP:5000"

# Test connection
curl http://YOUR-EC2-IP:5000/health
# Should return: OK

# Or test with Python
python -c "import mlflow; mlflow.set_tracking_uri('http://YOUR-EC2-IP:5000'); print('Connected!')"
# Should print: Connected!
```

**Problem:** Service won't start (exit code 203/EXEC)
```
# This means MLflow executable path is wrong
# Solution: Use virtual environment path in service file:
# ExecStart=/home/ubuntu/mlflow-env/bin/python -m mlflow server ...
# See mlflow/SETUP_GUIDE.md for correct service configuration
```

**Problem:** Address already in use (port 5000)
```bash
# This means MLflow is already running
# Check if service is running
sudo systemctl status mlflow

# If you manually started MLflow, stop it first
# Press Ctrl+C if running in terminal
# Or kill the process:
sudo lsof -ti:5000 | xargs sudo kill -9

# Then restart the service
sudo systemctl restart mlflow
```

**Problem:** Database connection errors
```bash
# Verify Neon connection string
# Check Neon IP allowlist includes EC2 IP
# Test connection from EC2
psql "YOUR-CONNECTION-STRING"
```

### S3 Access Issues

**Problem:** Unable to locate credentials (NoCredentialsError)
```bash
# This happens when AWS credentials are not in the environment
# Solution: Load credentials from .env file
source .env

# Verify credentials are loaded
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# If .env doesn't exist, create it:
cp env.example .env
# Then edit .env with your actual AWS credentials
```

**Problem:** InvalidAccessKeyId - "The AWS Access Key Id you provided does not exist"
```bash
# This means the access key is invalid or was deleted/rotated
# Solution 1: Load credentials and verify they work
source .env
aws s3 ls s3://mlflow-artifacts-patel/

# Solution 2: If the key is invalid, create new access keys in AWS IAM:
# 1. Go to AWS IAM Console → Users → mlflow-s3-user
# 2. Security credentials tab → Access keys → Create access key
# 3. Copy the new access key ID and secret access key
# 4. Update .env file with new credentials
# 5. Reload: source .env
# 6. Verify: aws s3 ls s3://mlflow-artifacts-patel/
```

**Note:** If S3 upload fails, the training script will continue and:
- Metrics will still be logged to MLflow (stored in PostgreSQL)
- Model will still be logged to MLflow
- Artifacts will be saved locally in `artifacts/` directory
- You can manually upload artifacts later or fix credentials and re-run

**Problem:** S3 access denied errors
```bash
# Verify IAM credentials are correct
aws s3 ls s3://your-bucket-name/

# Check IAM user has S3 permissions
# Verify bucket name matches exactly
```

### Training Script Issues

**Problem:** Models not logging to MLflow
```bash
# Verify MLFLOW_TRACKING_URI is set
echo $MLFLOW_TRACKING_URI

# Check MLflow server is accessible
curl $MLFLOW_TRACKING_URI

# Verify experiment name matches: "walmart-sales-forecast"
```

### FastAPI Issues

**Problem:** API not loading models
```bash
# Check models are registered in MLflow Model Registry
# Verify model names/versions in src/api/main.py
# Check MLflow tracking URI is configured in API
```

## Assignment Requirements Compliance

This project satisfies all assignment requirements:

- **Dataset Selection** - Public Walmart sales dataset
- **Data Cleaning** - Comprehensive cleaning pipeline
- **Time-Based Splitting** - 35% train, 35% validate, 30% test
- **H2O AutoML** - Automated model selection
- **Manual Model Training** - Three models with MLflow logging
- **Remote MLflow Setup** - EC2 + PostgreSQL + S3
- **Model Registry** - All models registered
- **FastAPI Deployment** - Three prediction endpoints
- **Drift Analysis** - Statistical drift detection using scipy
- **UV-based Environment** - No notebooks in submission
- **Reproducible** - All scripts are standalone Python files

## Contributing

This is an educational project. For questions or issues, please refer to the assignment documentation.

## License

This project is for educational purposes only.

## References

- Walmart Dataset: [Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- H2O AutoML: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
- FastAPI: https://fastapi.tiangolo.com/
- Evidently AI: https://docs.evidentlyai.com/

