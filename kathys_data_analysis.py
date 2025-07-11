import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

summary_statistics = pd.read_csv("summary_statistics.csv")
summary_statistics.head()

summary_statistics.describe()

summary_statistics['density'].hist(bins=50)
plt.title('Distribution of Matrix Densities')

summary_statistics['constraints'].hist(bins=50)
plt.title('Distribution of Constraints')

summary_statistics['variables'].hist(bins=50)
plt.title('Distribution of Variables')

summary_statistics['matrix_rank'].hist(bins=50)
plt.title('Distribution of Matrix Rank')

plt.scatter(summary_statistics['variables'], summary_statistics['nonzeros'])
plt.xlabel('Variables')
plt.ylabel('Nonzeros')
plt.title('Nonzeros vs Variables')
plt.grid(True)

plt.scatter(summary_statistics['variables'], summary_statistics['constraints'])
plt.xlabel('Variables')
plt.ylabel('Constraints')
plt.title('Constraints vs Variables')
plt.grid(True)

plt.scatter(summary_statistics['constraints'], summary_statistics['nonzeros'])
plt.xlabel('Constraints')
plt.ylabel('Nonzeros')
plt.title('Nonzeros vs Constraints')
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(summary_statistics['nonzeros'], summary_statistics['density'])
plt.xlabel('Nonzeros')
plt.ylabel('Density')
plt.title('Nonzeros vs Density')
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(summary_statistics['sparsity_%'], summary_statistics['density'], 'o-')
plt.xlabel('Sparsity (%)')
plt.ylabel('Density')
plt.title('Density vs Sparsity (%)')
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(summary_statistics['row_nnz_variance'], summary_statistics['col_nnz_variance'])
plt.xlabel('Row NNZ Variance')
plt.ylabel('Column NNZ Variance')
plt.title('Row vs Column NNZ Variance')
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(summary_statistics['nonzeros'], summary_statistics['row_nnz_variance'])
plt.xlabel('Nonzeros')
plt.ylabel('Row NNZ Variance')
plt.title('Nonzeros vs Row NNZ Variance')
plt.grid(True)
plt.show()

numeric_df = summary_statistics.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 10)) 
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()



features = [
    'variables', 'constraints', 'nonzeros', 'density', 'sparsity_%',
    'row_nnz_variance', 'col_nnz_variance',
    'avg_row_L2_norm', 'max_row_L2_norm',
    'integer_like_%', 'matrix_rank',
    'zero_rows', 'zero_columns'
]

X = summary_statistics[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow method
inertia = []
silhouette_scores = []

K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow and Silhouette
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, 'o-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'o-g')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')

plt.tight_layout()
plt.show()

best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=0, n_init='auto')
summary_statistics['cluster'] = kmeans.fit_predict(X_scaled)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# plot clusters using PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=summary_statistics['cluster'].astype(str),
                palette='Set2', s=80, edgecolor='black')
plt.title('PCA Projection (2D) of K-Means Clusters (k=5)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.legend(title='Cluster')
plt.show()


# plot in 3D
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                     c=summary_statistics['cluster'], cmap='Set2', s=80)
ax.set_title('PCA Projection (3D) of K-Means Clusters (k=5)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()

# rank clusters based on explained variance
pca = PCA(n_components=5)
pca.fit(X_scaled)
explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(6, 4))
plt.bar(range(0, 5), explained_var, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()

# Cluster centers (scaled space)
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
print("Cluster Centers:")
print(cluster_centers.round(2))

pca = PCA(n_components=5)
pca.fit(X_scaled)
explained = pca.explained_variance_ratio_
components = pd.DataFrame(pca.components_, columns=features)
print("Explained Variance:", explained)
print("Top PCA Components:\n", components)


