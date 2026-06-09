# 🤖 AI & ML Exam Preparation — Python Code Lab

> A structured collection of Machine Learning algorithms and Python library demos, built for DUET CS-310 (Artificial Intelligence) exam preparation.

---

## 📁 Repository Structure

```
ai-prep/
├── ML-Algorithms-Codes/
│   ├── LinearRegression.py        # House price prediction with visualization
│   ├── LogisticRegression.py      # Breast cancer classification + ROC curve
│   ├── KNNClassifier.py           # Iris dataset classification (K=5)
│   ├── K-Means-Clusteing.py       # Iris clustering with silhouette score
│   └── KMeansClustering.py        # Customer segmentation + Elbow method
│
├── Python-Libraries-Code/
│   ├── Numpycode.py               # Arrays, matrix operations, stats
│   ├── pandasCode.py              # DataFrames, filtering, aggregation
│   ├── MatplotlibCode.py          # Line, bar, and scatter plots
│   └── SkLearnCode.py             # KNN + Confusion Matrix display
│
├── pyproject.toml                 # Project dependencies
└── .gitignore
```

---

## 🧠 ML Algorithms Covered

| Algorithm | File | Dataset | Key Concepts |
|---|---|---|---|
| Linear Regression | `LinearRegression.py` | Synthetic (House Prices) | MSE, R² Score, Regression Line |
| Logistic Regression | `LogisticRegression.py` | Breast Cancer (sklearn) | Confusion Matrix, ROC Curve, AUC |
| K-Nearest Neighbors | `KNNClassifier.py` | Iris | Feature Scaling, Accuracy, Classification Report |
| K-Means Clustering | `K-Means-Clusteing.py` | Iris | WCSS / Inertia, Silhouette Score |
| K-Means (Advanced) | `KMeansClustering.py` | Synthetic (Customers) | Elbow Method, Centroid Visualization |

---

## 📦 Python Libraries Demonstrated

| Library | File | Topics |
|---|---|---|
| NumPy | `Numpycode.py` | 1D/2D arrays, math ops, transpose, stats |
| Pandas | `pandasCode.py` | DataFrame creation, filtering, aggregation |
| Matplotlib | `MatplotlibCode.py` | Line plot, bar chart, scatter plot |
| Scikit-learn | `SkLearnCode.py` | Full ML pipeline + ConfusionMatrixDisplay |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) *(recommended)* or pip

### Install dependencies

**Using uv:**
```bash
uv sync
```

**Using pip:**
```bash
pip install matplotlib numpy pandas scikit-learn seaborn reportlab
```

---

## ▶️ Running the Code

```bash
# Example — run any script directly
python ML-Algorithms-Codes/LinearRegression.py
python ML-Algorithms-Codes/LogisticRegression.py
python ML-Algorithms-Codes/KNNClassifier.py
python ML-Algorithms-Codes/KMeansClustering.py

python Python-Libraries-Code/Numpycode.py
python Python-Libraries-Code/pandasCode.py
python Python-Libraries-Code/MatplotlibCode.py
python Python-Libraries-Code/SkLearnCode.py
```

---

## 📋 Dependencies

From `pyproject.toml`:

```toml
matplotlib >= 3.10.9
numpy >= 2.4.6
pandas-stubs ~= 3.0.3
scikit-learn >= 1.9.0
seaborn >= 0.13.2
reportlab >= 4.5.1
```

---

## 👨‍💻 Author

**Sohaib Hafeez**
Roll No: 24F-CS-085 | Section A2
B.S. Computer Science — DUET, Karachi

[![GitHub](https://img.shields.io/badge/GitHub-M--sohaib--Hafeez-181717?style=flat&logo=github)](https://github.com/M-sohaib-Hafeez)

---

## 📄 License

This project is for academic and educational purposes.