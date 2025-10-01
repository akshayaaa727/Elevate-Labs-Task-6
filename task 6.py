# =============================================================================
# Task 6: K-Nearest Neighbors (KNN) Classification
# =============================================================================

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("--- K-Nearest Neighbors (KNN) Classification ---")

# =============================================================================
# 1. Load Dataset and Normalize Features
# =============================================================================
print("\n--- 1. Loading and Preparing Data ---")
# Load the Iris dataset
try:
    # The Iris dataset often comes without headers, so we name them.
    # If your file has headers, you can remove the `names` parameter.
    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    data = pd.read_csv('iris.csv', header=None, names=col_names)
    print("Iris dataset loaded successfully.")
    print("First 5 rows:\n", data.head())
except FileNotFoundError:
    print("Error: 'iris.csv' not found. Please download the dataset and place it in the correct directory.")
    exit()

# Separate features (X) and target (y)
X = data.drop('species', axis=1)
y = data['species']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")

# Normalize the features
# KNN is a distance-based algorithm, so feature scaling is crucial.
print("\nNormalizing features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =============================================================================
# 2. & 3. Experiment with Different Values of K
# =============================================================================
print("\n--- 2 & 3. Finding the Optimal Value for K ---")

error_rate = []
k_range = range(1, 30)

# Loop through k values to find the best one
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, y_train)
    pred_i = knn.predict(X_test_scaled)
    # error_rate is the proportion of incorrect predictions
    error_rate.append(np.mean(pred_i != y_test))

# Plot the error rate vs. K value
plt.figure(figsize=(10, 6))
plt.plot(k_range, error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.savefig('k_value_error_rate.png')
print("Plot of 'Error Rate vs. K Value' saved as 'k_value_error_rate.png'.")
print("Examine the plot to find the 'elbow' point, which suggests an optimal K.")

# Let's choose a K value from the plot (e.g., where the error rate is low and stable)
# For this dataset, a value around 5-11 is usually good. We'll pick K=7.
OPTIMAL_K = 7
print(f"Based on the plot, we will use K={OPTIMAL_K} for the final model.")


# =============================================================================
# 4. Train Final Model and Evaluate
# =============================================================================
print(f"\n--- 4. Training Final Model with K={OPTIMAL_K} and Evaluating ---")

# Train the final KNN model
knn_final = KNeighborsClassifier(n_neighbors=OPTIMAL_K)
knn_final.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn_final.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Final Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", class_report)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
print("Confusion matrix plot saved as 'confusion_matrix.png'.")


# =============================================================================
# 5. Visualize Decision Boundaries
# =============================================================================
print("\n--- 5. Visualizing Decision Boundaries ---")
print("Note: Decision boundary visualization requires reducing features to 2D.")
print("We will use 'petal_length' and 'petal_width' for this visualization.")

# Prepare data with only two features
X_vis = X[['petal_length', 'petal_width']]
y_vis = y

# We need to re-scale and re-train the model on the 2D data
scaler_vis = StandardScaler()
X_vis_scaled = scaler_vis.fit_transform(X_vis)

knn_vis = KNeighborsClassifier(n_neighbors=OPTIMAL_K)
knn_vis.fit(X_vis_scaled, y_vis)

# Create a mesh grid for the plot
h = .02  # step size in the mesh
x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict the class for each point in the mesh grid
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Convert string labels to numbers for plotting
y_numeric = y.astype('category').cat.codes
Z_numeric = pd.Series(Z).astype('category').cat.codes.values.reshape(xx.shape)

# Define a color map
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ['red', 'green', 'blue']

# Plot the decision boundaries
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z_numeric, cmap=cmap_light)

# Plot the training points
sns.scatterplot(x=X_vis_scaled[:, 0], y=X_vis_scaled[:, 1], hue=y,
                palette=cmap_bold, alpha=1.0, edgecolor="black")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f'KNN (K={OPTIMAL_K}) Decision Boundaries for Iris Petals')
plt.xlabel('Petal Length (Standardized)')
plt.ylabel('Petal Width (Standardized)')
plt.legend(title='Species')
plt.grid(True)
plt.savefig('decision_boundaries.png')
print("Decision boundaries plot saved as 'decision_boundaries.png'.")

print("\n\n===== Task Complete =====")
