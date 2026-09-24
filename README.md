## **Youtube Comment Sentiment Analysis (End-to-End MLOps Project)**

Developed an end-to-end **Machine Learning–based sentiment analysis system** for Youtube comments using Natural Language Processing (NLP). The project follows modern **MLOps best practices**, leveraging **DVC for data versioning and pipeline management**, **MLflow for experiment tracking and model lifecycle management**, and **DAGsHub for centralized experiment visualization**, ensuring reproducibility, scalability, and efficient experimentation.

---

### **Key Responsibilities & Features**

* Collected, cleaned, and preprocessed large-scale Reddit comment data (tokenization, stopword removal, text normalization).
* Engineered textual features using TF-IDF and/or word embeddings for sentiment classification.
* Built and trained sentiment classification models achieving **98% accuracy** on the validation dataset.
* Designed and implemented **DVC pipelines** for data ingestion, preprocessing, feature engineering, training, and evaluation.
* Integrated **MLflow** to track experiments, hyperparameters, metrics (accuracy, precision, recall, F1-score), and model artifacts.
* Used **DAGsHub** as a remote backend for MLflow to enable centralized experiment tracking and collaboration.
* Evaluated model performance using confusion matrix and standard NLP classification metrics.
* Containerized the application using **Docker** for consistent deployment across environments.
* Maintained a clean, modular project structure with Git-based version control and reproducible workflows.

---

### **Tools & Technologies**

Python, NLP, Scikit-learn / TensorFlow / PyTorch, **DVC**, **MLflow**, **DAGsHub**, Docker, Pandas, NumPy, Git

---

### **Outcome**

* Delivered a high-performance sentiment analysis pipeline with **98% accuracy** and full experiment traceability.
* Enabled reproducible ML workflows and faster iteration through structured MLOps practices.

---

## **Project Structure**

```
youtube-sentiment-analysis/
│
├── .dvc/                       
├── .github/                      # CI/CD workflows
│
├── artifacts/                    # Pipeline outputs (auto-generated)
├── logs/                         # Runtime logs (auto-generated)
├── models/                       # Trained models (DVC tracked, auto-generated)
│
├── backend/
│   ├── app.py                    # API for model inference
│   └── requirements.txt          # Backend dependencies
│
├── scripts/                      # Utility & automation scripts
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py     # Data collection/loading
│   │   └── data_preprocessing.py # Text preprocessing
│   │
│   ├── model/
│   │   ├── model_building.py     # Model training
│   │   ├── model_evaluation.py   # Evaluation (98% accuracy)
│   │   └── register_model.py     # MLflow model registration
│   │
│   └── dagshub_config.py         # MLflow + DAGsHub configuration
│
├── params.yaml                   # Hyperparameters & paths (manual)
├── README.md                     # Project documentation (manual)
├── requirements.txt              # Project dependencies (manual)
├── setup.py                      # Package configuration (manual)
├── template.py                   # Project scaffolding script (manual)
│
├── dvc.yaml                      # DVC pipeline definition (manual)
├── dvc.lock                      # Pipeline lock file (auto-generated)
│
├── Dockerfile                    # Container configuration
├── .env                          # Environment variables (manual)
├── .gitignore
├── .dvcignore
└── LICENSE
```

---

## **Data Import**

The data ingestion process downloads the dataset from a remote source and saves it locally.

* **Source URL**:
  `https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv`

* **Destination**:
  Data is split and saved to `artifacts/data/` as `train.csv` and `test.csv`.

To run data ingestion manually:

```bash
python src/data/data_ingestion.py
```

---

## **DVC Pipeline**

This project uses **DVC (Data Version Control)** to manage the complete machine learning pipeline.

To reproduce the entire pipeline:

```bash
dvc repro
```

### **Pipeline Stages**

1. **Data Ingestion** – Downloads and splits raw data
2. **Data Preprocessing** – Cleans and preprocesses text
3. **Model Building** – Trains the sentiment classification model
4. **Model Evaluation** – Evaluates performance and generates metrics
5. **Model Registration** – Registers the trained model using MLflow

---

## **MLflow Tracking & DAGsHub Integration**

* **MLflow** is used to track:

  * Experiments
  * Hyperparameters
  * Metrics (Accuracy, Precision, Recall, F1-Score)
  * Model artifacts
* **DAGsHub** acts as a remote backend for MLflow, enabling:

  * Centralized experiment visualization
  * Team collaboration
  * Model comparison across runs

> All experiment runs are logged automatically during pipeline execution.

---

## **Dockerization**

The project is fully containerized using **Docker** to ensure consistency across development and production environments.

### Build Docker Image

```bash
docker build -t youtube-sentiment-analysis .
```

### Run Docker Container

```bash
docker run -p 5000:5000 youtube-sentiment-analysis
```

This launches the backend API for real-time sentiment prediction.

---

## **Getting Started**

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Reproduce the full pipeline:

   ```bash
   dvc repro
   ```
3. (Optional) Run the API:

   ```bash
   python backend/app.py
   ```
