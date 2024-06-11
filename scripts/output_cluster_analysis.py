import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import math

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
dataset = pd.read_excel('/Users/blake/Downloads/Custom Leader AI response test output.xlsx', sheet_name='Sheet1')

# Print column names to identify the correct ones
print("Column names in dataset:", dataset.columns.tolist())

# Replace 'Score' and 'Reasons' with actual column names after printing
dataset_mean = dataset['Response Score (1 to 10) 1 is poor and 10 is excellent'].mean()
cutoff = math.ceil(dataset_mean)
unsatisfactory_answers = dataset[dataset['Response Score (1 to 10) 1 is poor and 10 is excellent'] < cutoff]
reasons = unsatisfactory_answers['Additional Feedback'].fillna("No feedback provided")  # Handling NaN

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and not token in stop_words]
    return ' '.join(lemmatized)

processed_reasons = reasons.apply(preprocess)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(processed_reasons)

# Clustering
n_clusters = 2  # Adjust the number of clusters based on your data
model = KMeans(n_clusters=n_clusters, random_state=42)
model.fit(X)

# Assign the cluster labels to the original data
unsatisfactory_answers['Cluster'] = model.labels_

# Identify and label satisfactory answers
satisfactory_answers = dataset[dataset['Response Score (1 to 10) 1 is poor and 10 is excellent'] >= cutoff]
satisfactory_answers['Cluster'] = 2

# Combine the datasets back together
final_dataset = pd.concat([unsatisfactory_answers, satisfactory_answers], ignore_index=True)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(range(len(final_dataset)), [0]*len(final_dataset), c=final_dataset['Cluster'], cmap='viridis', alpha=0.6)
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
plt.xlabel('Data Points')
plt.title('Cluster Distribution')
plt.yticks([])
plt.show()

# Print the top terms per cluster
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(n_clusters):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"Cluster {i}: {', '.join(top_terms)}")

# Review the clusters
for i in range(n_clusters):
    print(f"\nCluster {i} reasons:")
    print(unsatisfactory_answers[unsatisfactory_answers['Cluster'] == i]['Additional Feedback'].tolist())


final_dataset.to_csv('training_dataset.csv')