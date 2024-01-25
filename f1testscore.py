from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# Generate synthetic data (replace with your actual data)
X, y = make_classification(n_samples=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# print(f"Number of features in X_train: {X_train.shape[1]}")
# print(f"Number of features in X_test: {X_test.shape[1]}")


# Create a pipeline
pipe = make_pipeline(
    MinMaxScaler(),
    VarianceThreshold(threshold=0.01),
    LogisticRegression(solver='lbfgs', max_iter=1000)
)


# Fit the model and evaluate performance
pipe.fit(X_train, y_train)
f1_test_score = f1_score(y_test, pipe.predict(X_test))
print(f"F1 Score on Test Set: {f1_test_score:.4f}")

