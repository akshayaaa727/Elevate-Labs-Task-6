K-Nearest Neighbors (KNN) Classification Tutorial 
This project provides a hands-on guide to implementing the K-Nearest Neighbors (KNN) algorithm, a fundamental supervised machine learning technique for classification. The Python script uses the well-known Iris dataset to walk through feature normalization, hyperparameter tuning, model evaluation, and the visualization of decision boundaries.

Project Workflow
The script knn_classification.py is structured to follow a clear, educational workflow:

Data Preparation: It loads the Iris dataset and normalizes the features using StandardScaler, a critical step for distance-based algorithms like KNN.

Hyperparameter Tuning: It experiments with different values of K (the number of neighbors) and plots the model's error rate to find the optimal value.

Model Training: It trains a KNeighborsClassifier using the best value of K found in the previous step.

Model Evaluation: It assesses the final model's performance using key metrics like accuracy and generates a confusion matrix for a detailed look at prediction success and failure across classes.

Decision Boundary Visualization: To provide an intuitive understanding of how the model works, it plots the decision boundaries on a 2D-subset of the data.
