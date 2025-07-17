# 🌸 Iris Flower Classification with Gradio Interface and 100% Accuracy

This project uses the popular **Iris dataset** from `sklearn` to classify iris flowers into their respective species. We've built an **interactive web app using Gradio** for real-time predictions and performed **exploratory data analysis** through visualizations like histograms, scatter plots, pie charts, and heatmaps. 

Both **Random Forest** and **K-Nearest Neighbors (KNN)** classifiers achieved **100% accuracy**, showing excellent model performance on this dataset.

---

## 📁 Project Overview

- **Dataset**: Iris Dataset from `sklearn.datasets`
- **Problem Type**: Multiclass Classification
- **Features Used**:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target**: Iris Species (*Setosa*, *Versicolor*, *Virginica*)
- **Interface**: Built with **Gradio** for interactive predictions

---

## 🔍 Exploratory Data Analysis (EDA)

We explored the dataset using the following plots:

### 📊 Visualizations
- **Histogram**: Distribution of each feature
- **Scatter Plot**: Feature-wise scatter with color-coded species
- **Pie Chart**: Proportion of each class
- **Heatmap**: Correlation matrix of all features

These visualizations helped identify important patterns and separability between classes—especially for petal features.

---

## 🧠 Machine Learning Models

### ✅ Random Forest Classifier
- Ensemble method using decision trees.
- Achieved **100% accuracy** on test data.

### ✅ K-Nearest Neighbors (KNN)
- Distance-based algorithm.
- Also achieved **100% accuracy**.

Both models were evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

| Model                | Accuracy |
|---------------------|----------|
| Random Forest        | 100%     |
| K-Nearest Neighbors  | 100%     |

---

## 🌐 Gradio Interface

We developed a user-friendly **Gradio web interface** for real-time predictions.

### Features:
- Accepts 4 numerical inputs: sepal and petal length/width
- Displays predicted iris species using trained model
- Lightweight and easy to use in a browser

> ✅ Just run the script and open the Gradio link to test the model live!

---

## 📌 Project Structure
iris_classifier_gradio/
│
├── iris_classifier.ipynb # Jupyter notebook with EDA, modeling, and Gradio app
├── app.py # Python script to run Gradio interface
├── requirements.txt # List of dependencies
├── README.md # Project documentation
└── visuals/ # Folder with EDA plots (histogram, scatter, etc.)

yaml
Copy
Edit

---

## ⚙️ Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/iris-classifier-gradio.git
cd iris-classifier-gradio
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Gradio app:

bash
Copy
Edit
python app.py
🧪 Technologies Used
Python

Scikit-learn

Pandas

Matplotlib, Seaborn

Gradio

✨ Future Enhancements
Add model switching (RF/KNN toggle in UI)

Deploy on Hugging Face Spaces or Streamlit Cloud

Introduce advanced classifiers like SVM or XGBoost

📷 Sample Visualizations
Histogram	Scatter Plot	Pie Chart	Heatmap
✅	✅	✅	✅

💡 Conclusion
With effective data exploration and two reliable machine learning models, this project achieved perfect accuracy. The Gradio interface makes it interactive and deployable—ideal for educational demos and quick prototyping.


