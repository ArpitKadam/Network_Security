# 🌐 Network Security Project for Phishing Data

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.85%2B-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.0-orange.svg)
![Dagshub](https://img.shields.io/badge/Dagshub-Enabled-brightgreen.svg)

---

## 🚀 Project Overview
The **Network Security Project** is a comprehensive solution designed to detect phishing activities using advanced machine learning techniques. The project seamlessly integrates data ingestion, preprocessing, model training, and prediction into an interactive and scalable pipeline.

---

## ✨ Features
- 🔍 **Phishing Detection**: Leverage trained models to detect phishing activities.
- 🛠️ **Interactive API**: FastAPI-powered endpoints for training and predictions.
- 💾 **MongoDB Integration**: Efficiently manage data ingestion.
- 📊 **Model Tracking**: Track experiments and results with MLflow.
- 🐳 **Dockerized Deployment**: Simplified application setup with AWS ECR(Elastic Container Registry).
- ✅ **Data Validation**: Schema-driven data validation for accuracy.

---

## 🛠️ Installation Guide
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ArpitKadam/Network_Security.git
   cd Network_Security
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv myenv
   myenv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file and add your MongoDB connection string:
   ```env
   MONGO_DB_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net/<database>
   ```

5. **Run the application**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

---

## 💡 Technologies Used
- **Programming**: Python
- **Frameworks**: FastAPI, scikit-learn
- **Database**: MongoDB
- **Deployment**: Docker
- **Experiment Tracking**: MLflow, Dagshub

---

## 🌟 DAGsHub Integration
Enhance your workflow with DAGsHub:
1. Log in to [DAGsHub](https://dagshub.com/).
2. Connect your GitHub repository.
3. Configure MLflow tracking:
   ```python
   mlflow.set_tracking_uri("https://dagshub.com/<username>/<repo>.mlflow")
   ```

---

## 🔍 MLflow Tracking
Track experiments and monitor models:
1. Start the MLflow server:
   ```bash
   mlflow ui
   ```
2. Access the dashboard at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 🌐 MongoDB Linking Guide
1. Create a MongoDB Atlas cluster.
2. Obtain your connection string:
   ```
   mongodb+srv://<username>:<password>@cluster.mongodb.net/<database>
   ```
3. Add the connection string to `.env`:
   ```env
   MONGO_DB_URL=<your_connection_string>
   ```
4. Verify the integration with this snippet:
   ```python
   from pymongo import MongoClient
   import certifi
   
   ca = certifi.where()
   client = MongoClient(mongo_db_url, tlsCAFile=ca)
   database = client["<database_name>"]
   collection = database["<collection_name>"]
   ```

---

## 📂 Project Structure
```
├── Artifacts              # Generated files and models
├── data_schema            # Data validation schemas
├── final_models           # Pretrained models
├── logs                   # Logs for execution
├── mlruns                 # MLflow experiment data
├── networksecurity        # Source code modules
├── prediction_output      # Prediction results
├── templates              # HTML templates
├── Dockerfile             # Docker configuration
├── requirements.txt       # Dependencies
├── app.py                 # Main application
└── setup.py               # Project setup
```

---

## 🤝 Contributing
We welcome contributions! Fork the repository, make your changes, and submit a pull request.

---

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🌟 Get Started Today!
Ready to explore the world of phishing detection? Follow the steps above and bring this project to life!
