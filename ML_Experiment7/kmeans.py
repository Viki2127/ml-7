
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset
data = {
    'X': [1, 1.5, 3, 5, 3.5, 4.5, 3.5],
    'Y': [1, 2, 4, 7, 5, 5, 4.5]
}

df = pd.DataFrame(data)

# Apply K-Means clustering
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

kmeans.fit(df)
labels = kmeans.labels_

# Plot clusters
plt.scatter(df['X'], df['Y'], c=labels)

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("K-Means Clustering")

plt.show()
