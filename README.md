
# 🛡️ The Multi-Modal Graph AI Neural Real-Time Fraud Engine

An enterprise-grade, multi-modal fraud detection microservice that combines **Graph Neural Network (GNN) embeddings**, **real-time vector similarity screening**, and **FastAPI REST endpoints** with an interactive **Streamlit Intelligence Hub**.

---

## 🚀 Key Features

* 📊 **Interactive Streamlit Hub (`app.py`):** Real-time evaluation portal for live transaction parameter testing, credit scoring inputs, and immediate risk classification feedback.
* 🔍 **Multi-Modal Risk Screening:** Evaluates user profile metrics, transaction velocity, and graph structure embeddings to detect critical fraud rings and suspicious patterns.
* ⚡ **REST API Integration:** Interoperates directly with high-throughput FastAPI backends (`/predict/fraud`) with clear status classification (`CRITICAL`, `SUSPICIOUS`, `LEGITIMATE`).
* 🧠 **Extensible Vector Architecture:** Built to ingest multi-modal graph embeddings and real-time transaction features for sub-second inference latency.

---

## 🛠️ Tech Stack & Dependencies

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Real-time GNN Fraud Intelligence Hub |
| **Backend Service** | FastAPI / Uvicorn | Async REST API Inference Server |
| **HTTP Client** | Requests | Asynchronous & Synchronous backend communication |
| **AI / Machine Learning** | PyTorch / GNN Core | Vector similarity & graph neural network risk scoring |

---

## 📦 Project Structure

```text
The-Multi-Modal-Graph-AI-Neural-Real-Time-Fraud-Engine/
├── app.py                # Streamlit Executive Intelligence Dashboard
├── api/
│   └── main.py           # FastAPI Backend Service (/predict/fraud)
├── engine/
│   ├── gnn_model.py      # Graph Neural Network Core Model
│   └── similarity.py     # Vector Similarity & Embeddings Engine
├── requirements.txt      # Python dependencies
└── README.md              # Project Documentation
```

---

## ⚡ Quick Start & Setup

### 1. Prerequisites
* Python 3.10+ installed on your system.

### 2. Install Dependencies
```bash
pip install streamlit requests fastapi uvicorn torch
```

### 3. Start the FastAPI Backend Server
In your first terminal, launch the backend API:
```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Launch the Streamlit Intelligence Hub
In a second terminal, start the Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test transactions in real time.

---

## 📜 Git Version History & Commits

```bash
git add .
git commit -m "feat: launch multi-modal graph fraud engine and streamlit intelligence portal"
git commit -m "feat(api): integrate real-time /predict/fraud endpoint connection"
git commit -m "style: configure high-visibility risk level badges and JSON telemetry inspector"
```

---

## 👤 Author & Credits

* **Developer:** Abhay Panchal  
* **Role:** Full-Stack & AI Systems Developer  
* **Email:** [abhaypanchal095@gmail.com](mailto:abhaypanchal095@gmail.com)  
* **Project:** The Multi-Modal Graph AI Neural Real-Time Fraud Engine
