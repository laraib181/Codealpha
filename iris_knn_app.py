
# iris_knn_app.py

import gradio as gr
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
best_knn = KNeighborsClassifier(n_neighbors=3)
best_knn.fit(X_train_scaled, y_train)

# Define prediction function
def classify_iris(sepal_length, sepal_width, petal_length, petal_width):
    input_features = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = best_knn.predict(input_features)
    return iris.target_names[prediction[0]]

# Create Gradio interface
iface = gr.Interface(
    fn=classify_iris,
    inputs=[
        gr.Slider(minimum=0, maximum=10, value=5.1, label="Sepal Length (cm)"),
        gr.Slider(minimum=0, maximum=5, value=3.5, label="Sepal Width (cm)"),
        gr.Slider(minimum=0, maximum=10, value=1.4, label="Petal Length (cm)"),
        gr.Slider(minimum=0, maximum=5, value=0.2, label="Petal Width (cm)")
    ],
    outputs=gr.Label(),
    title="🌸 Iris Flower Classifier (KNN)",
    description="Classify the species of an Iris flower based on its measurements using a KNN model."
)

# Launch interface
if __name__ == "__main__":
    iface.launch(debug=True)
