# 🧠 RAG Hallucination Detector

An interactive machine learning application that detects whether an AI-generated response is likely to contain a hallucination based on key RAG and response-generation features.

The project uses a trained **Random Forest** classifier and provides an interactive **Streamlit** interface for real-time prediction.

## 🚀 Live Demo

**Try the application:**
[RAG Hallucination Detector — Streamlit App](https://rag-hallucination-detector-by-devendra35.streamlit.app/?utm_source=chatgpt.com)

## 📌 Overview

Large Language Models (LLMs) can generate responses that sound convincing but are not supported by the retrieved information in a Retrieval-Augmented Generation (RAG) system.

This project addresses that problem by using machine learning to estimate the probability that a response contains a hallucination.

The application analyzes six input features related to LLM generation and retrieval quality and returns:

* Hallucination / No Hallucination
* Hallucination probability
* Prediction confidence
* Prediction probability visualization
* Detailed input and model output

The Streamlit application loads the trained model and feature configuration from serialized `.pkl` files.

## ✨ Features

* 🧠 Machine learning-based hallucination detection
* 🌡️ Temperature setting analysis
* 🔎 Vector database similarity analysis
* 📝 Prompt token analysis
* 💬 Response token analysis
* 📚 Sentence complexity analysis
* 🎭 Subjectivity analysis
* 📊 Hallucination probability
* 🎯 Prediction confidence
* 📈 Probability visualization
* 🖥️ Interactive Streamlit interface

## 🤖 Machine Learning Model

The deployed application uses:

| Component        | Details               |
| ---------------- | --------------------- |
| Algorithm        | Random Forest         |
| Training Samples | 24,000                |
| Test Samples     | 6,000                 |
| Features         | 6                     |
| Task             | Binary Classification |

The application reports the following test metrics:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 96.22% |
| Precision | 96.81% |
| Recall    | 97.24% |
| F1 Score  | 97.03% |
| ROC-AUC   | 99.22% |

These metrics are displayed directly by the deployed application's model-information section.

## 📊 Input Features

The model uses six features:

1. **Temperature Setting** — controls the randomness of LLM generation.
2. **Vector DB Similarity Score** — represents similarity between the query and retrieved documents.
3. **Prompt Tokens** — number of tokens in the input prompt.
4. **Response Tokens** — number of tokens in the generated response.
5. **Sentence Complexity Index** — represents response sentence complexity.
6. **Subjectivity Score** — represents the subjectivity of the response.

These are the six features passed to the trained model by the Streamlit application.

## 🔄 How It Works

```text
                 User Input
                     │
                     ▼
          ┌─────────────────────┐
          │   RAG Features      │
          │                     │
          │ • Temperature       │
          │ • Similarity Score  │
          │ • Prompt Tokens     │
          │ • Response Tokens   │
          │ • Complexity        │
          │ • Subjectivity      │
          └──────────┬──────────┘
                     │
                     ▼
              Random Forest
                  Model
                     │
                     ▼
          ┌─────────────────────┐
          │   Prediction        │
          ├─────────────────────┤
          │ Hallucination       │
          │ Probability         │
          │ Confidence          │
          └─────────────────────┘
```

## 📁 Project Structure

```text
Rag-Hallucination-detector/
│
├── 01_data_exploration.ipynb
├── app.py
├── rag_hallucination_detector.pkl
├── rag_hallucination_features.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

The repository currently contains the exploration notebook, Streamlit application, trained model, feature configuration, dependency file, and Git configuration.

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/devendra35/Rag-Hallucination-detector.git
cd Rag-Hallucination-detector
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔍 Using the Application

1. Open the application.
2. Adjust the six input features using the interactive controls.
3. Click **Detect Hallucination**.
4. The model generates a prediction.
5. Review the hallucination probability and confidence.
6. Examine the prediction probability chart and detailed analysis.

The current interface implements these controls and prediction outputs directly in `app.py`.

## 📚 Dataset

The project was developed using the **LLM RAG Hallucination and Faithfulness Benchmark** dataset.

The dataset contains synthetic RAG interactions with text, metadata, engineered features, and hallucination-related targets.

## 🎯 Project Objective

The primary objective is to demonstrate how machine learning can be applied to identify potential hallucinations in RAG-based LLM responses using structured features.

This project combines:

```text
RAG
+
LLM Reliability
+
Feature Engineering
+
Machine Learning
+
Streamlit
```

## 📈 Future Improvements

* Add text-based analysis of the prompt, context, and response.
* Experiment with additional machine learning algorithms.
* Add explainable AI techniques such as SHAP.
* Improve hallucination-type classification.
* Add context-faithfulness prediction.
* Add automated model monitoring.
* Improve the Streamlit dashboard.
* Evaluate the model on real-world RAG responses.

## 👨‍💻 Author

**Devendra Khanal**

GitHub: [@devendra35](https://github.com/devendra35?utm_source=chatgpt.com)

## ⭐ Acknowledgment

Dataset: **LLM RAG Hallucination and Faithfulness Benchmark**

This project was created for exploring machine learning approaches to **LLM hallucination detection and RAG reliability**.
