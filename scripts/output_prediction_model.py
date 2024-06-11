import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

# Load the dataset
data = pd.read_json('/Users/blake/LeaderAI/updated_training_dataset.json')
data.dropna(subset=['Response', 'Cluster'], inplace=True)
data['Response'] = data['Response'].astype(str)

# Split data into features and target
X = data['Response']
y = data['Cluster']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Define a Pipeline that includes vectorization and a placeholder for the classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression())  # Placeholder, will be replaced in grid search
])

# Define the parameter grid, note that `clf` is a step in the pipeline
param_grid = [
    {
        'clf': [LogisticRegression()],
        'clf__penalty': ['l2'],
        'clf__C': [1, 10, 100],
        'vect__max_features': [100, 500],
        'vect__ngram_range': [(1, 1), (1, 2)]
    },
    {
        'clf': [SVC()],
        'clf__kernel': ['linear', 'rbf'],
        'clf__C': [0.1, 1, 10],
        'vect__max_features': [100, 500],
        'vect__ngram_range': [(1, 1), (1, 2)]
    },
    {
        'clf': [MultinomialNB()],
        'clf__alpha': [0.1, 1.0, 10.0],
        'vect__max_features': [100, 500],
        'vect__ngram_range': [(1, 1), (1, 2)]
    },
    {
        'clf': [DecisionTreeClassifier()],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5],
        'vect__max_features': [100, 500],
        'vect__ngram_range': [(1, 1), (1, 2)]
    }
]

# Configure GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best model and parameters
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score: {:.3f}".format(grid_search.best_score_))

# Evaluate on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Set Accuracy: {:.3f}".format(test_accuracy))

# Serialize the best model
dump(best_model, 'best_text_classifier.joblib')
print("Model saved as 'best_text_classifier.joblib'")