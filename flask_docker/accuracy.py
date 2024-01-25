from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Introduce data leakage by duplicating some features
X_leakage = np.hstack([X, X[:, :5]])  # Duplicating the first 5 features
X_leakage, y_leakage = shuffle(X_leakage, y, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for data leakage
if np.array_equal(X_train, X_test) or np.array_equal(y_train, y_test):
    print("Data Leakage Detected!")

# Create a simple logistic regression model
model = LogisticRegression(penalty='l2', C=1.0, random_state=42)


# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

# Check cross-validation folds setup
if len(np.unique(cv_scores)) == 1 and np.unique(cv_scores)[0] == 1.0:
    print("Cross-Validation Folds Issue Detected!")

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Check model evaluation
if accuracy == 1.0:
    print("Model Evaluation Issue Detected!")

# Check model complexity
if model.coef_.size > 10:
    print("Model Complexity Issue Detected!")

# Print results
print(f"Initial Accuracy: {accuracy}")
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean accuracy: {np.mean(cv_scores)}")
