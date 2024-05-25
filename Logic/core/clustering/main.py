import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/Users/hajmohammadrezaee/Desktop/MIR-Project/Logic/core')
from word_embedding.fasttext_data_loader import FastTextDataLoader
from word_embedding.fasttext_model import FastText, preprocess_text
from dimension_reduction import DimensionReduction
from clustering_metrics import ClusteringMetrics
from clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks

project_name = 'clustering'
run_name = 0
k_values = list(range(2, 9))
# 0. Embedding Extraction
# TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.

'''
dataloader = FastTextDataLoader('../IMDB_crawled.json', preprocess_text)
X, y = dataloader.create_train_data()
document_labels = list(dataloader.le.inverse_transform(y))
ft_model = FastText()
ft_model.prepare(None, 'load')
embed = []
with tqdm(X) as pbar:
    for x in pbar:
        embed.append(ft_model.get_query_embedding(x))

embed = np.array(embed)

np.save('cluster_embeddings.npy', embed)
np.save('cluster_labels.npy', y)
with open('document_labels.json', 'w') as f:
    f.write(json.dumps(document_labels))
    f.close()

'''
embed = np.load('cluster_embeddings.npy')
y = np.load('cluster_labels.npy')

with open('document_labels.json', 'r') as f:
    document_labels = json.loads(f.read())
    f.close()


# 1. Dimension Reduction
# TODO: Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.


dr = DimensionReduction()
pca_embed = dr.pca_reduce_dimension(embed, 2)
dr.wandb_plot_explained_variance_by_components(embed, project_name, run_name)

# Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.

tsne_embed = dr.wandb_plot_2d_tsne(embed, project_name, run_name)
# 2. Clustering
## K-Means Clustering
# Implement the K-means clustering algorithm from scratch.
# Create document clusters using K-Means.
# Run the algorithm with several different values of k.
# For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
clustering = ClusteringUtils()
metrics = ClusteringMetrics()
clustering.visualize_elbow_method_wcss(tsne_embed, k_values, project_name, run_name)
clustering.plot_kmeans_cluster_scores(tsne_embed, y, k_values, project_name, run_name)
for k in k_values:
    clustering.visualize_kmeans_clustering_wandb(tsne_embed, k, document_labels, project_name, run_name)
## Hierarchical Clustering
# Perform hierarchical clustering with all different linkage methods.
# Visualize the results.
for linkage_method in ['single', 'average', 'complete', 'ward']:
    clustering.wandb_plot_hierarchical_clustering_dendrogram(tsne_embed, linkage_method, project_name, run_name)
# 3. Evaluation
# Using clustering metrics, evaluate how well your clustering method is performing.
complete_pred = clustering.cluster_hierarchical_complete(tsne_embed)
average_pred = clustering.cluster_hierarchical_average(tsne_embed)
single_pred = clustering.cluster_hierarchical_single(tsne_embed)
ward_pred = clustering.cluster_hierarchical_ward(tsne_embed)

for method, pred in [('complete', complete_pred),\
                    ('average', average_pred),\
                    ('single', single_pred),\
                    ('ward', ward_pred)]:
    print(f'test result on {method} linkage method')
    print(f'purity score: {metrics.purity_score(y, pred)}')
    print(f'silhouette score: {metrics.silhouette_score(tsne_embed, pred)}')
    print(f'adjusted rand score: {metrics.adjusted_rand_score(y, pred)}')
