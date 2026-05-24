# 🔬 HIGGS Boson Detection — Random Forest Classifier

> **Machine Learning Classification Assignment**  
> Dataset Size: ~7.5 GB | Model: Random Forest | Language: Python

---

## 📌 Problem Statement

Classify particle collision events as **Signal** (Higgs Boson produced) or **Background** (noise) using 28 physics-derived features from the HIGGS dataset published by UCI Machine Learning Repository.

---

## 📦 Dataset

| Property       | Details                                               |
|----------------|-------------------------------------------------------|
| Name           | HIGGS Dataset                                         |
| Source         | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS) |
| Size           | ~7.5 GB (uncompressed) / ~2.6 GB (compressed)         |
| Rows           | 11,000,000                                            |
| Features       | 28 physics features                                   |
| Target         | Binary (0 = Background, 1 = Signal)                  |

The dataset is **automatically downloaded** when you run `main.py`.

---

## 🏗️ Project Structure

```
ml_classification_project/
│
├── main.py             # Full pipeline (download → preprocess → train → evaluate)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── results/            # Auto-generated output graphs & metrics
    ├── 1_confusion_matrix.png
    ├── 2_roc_curve.png
    ├── 3_feature_importances.png
    ├── 4_class_distribution.png
    ├── 5_probability_distribution.png
    └── metrics_summary.txt
```

---

## 🔄 Pipeline Steps

```
1. Download Dataset    →  Auto-downloads HIGGS.csv.gz (~2.6 GB) from UCI
2. Read Dataset        →  Loads 500,000 rows with proper column names
3. Preprocess          →  Null check, Train/Test split (80/20), StandardScaler
4. Train Model         →  RandomForestClassifier (100 trees, max_depth=15)
5. Evaluate & Plot     →  5 result graphs + metrics summary
```

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/ml_classification_project.git
cd ml_classification_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python main.py
```

> **Note:** The dataset (~2.6 GB compressed) will be downloaded automatically on first run. Make sure you have ~10 GB free disk space and a stable internet connection.

---

## 📊 Results

### Confusion Matrix
![Confusion Matrix](results/1_confusion_matrix.png)

### ROC Curve
![ROC Curve](results/2_roc_curve.png)

### Feature Importances
![Feature Importances](results/3_feature_importances.png)

### Class Distribution
![Class Distribution](results/4_class_distribution.png)

### Probability Distribution
![Probability Distribution](results/5_probability_distribution.png)

---

## 🧰 Tech Stack

| Tool            | Version  |
|-----------------|----------|
| Python          | 3.9+     |
| scikit-learn    | 1.3+     |
| pandas          | 2.0+     |
| numpy           | 1.24+    |
| matplotlib      | 3.7+     |
| seaborn         | 0.12+    |

---

## 📈 Model Configuration

```python
RandomForestClassifier(
    n_estimators = 100,
    max_depth    = 15,
    n_jobs       = -1,       # Use all CPU cores
    random_state = 42
)
```

---

## 👤 Author

**[Your Full Name]**  
Roll Number: [Your Roll Number]  
Course: [Your Course Name]

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
