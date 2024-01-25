from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression

model = ... 



X, y = make_classification(n_samples=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Assuming you have a function for training your model, modify it as needed
def train_model(X_train, y_train):
    # Your training code here
    # Make sure to use the correct feature names
    model.fit(X_train, y_train)
    return model


# Create a pipeline
pipe = make_pipeline(
    MinMaxScaler(),
    VarianceThreshold(threshold=0.01),
    LogisticRegression(solver='lbfgs', max_iter=1000)
)

pipe.fit(X_train, y_train)
f1_test_score = f1_score(y_test, pipe.predict(X_test))
print(f"F1 Score on Test Set: {f1_test_score:.4f}")

# Example usage:
new_model = train_model(X_train, y_train)
joblib.dump(new_model, "new_model3.pkl")

