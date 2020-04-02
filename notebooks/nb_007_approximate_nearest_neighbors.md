---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# KNN 
Zaimplementować wykorzystanie kernelu gaussowskiego w wyszukiwaniu najbliższych sąsiadów przez approximate NN


```
For Euclidean distance, KD Tree and Ball Tree are the very best[1]. But, they can be only used for Euclidean distance nearest neighbor search due to their data structure. Further, b0th of them tend to perform poorly on high dimensional datasets (eg: dimension > 60). Their performance can be even worse than brute force approach in such scenarios.

Locality sensitive hashing[2] the most popular approximate nearest neighbor search method, which can be used effectively with high dimensional datasets. This can be used for most of the distance metrics (Euclidean, Cosine, etc.) by selecting the correct approximation method (such as random projection). There are also different variants of LSH technique for nearest neighbor search such as LSH forest[3].

https://www.quora.com/What-are-the-most-popular-Nearest-Neighbor-Search-algorithms
```

```python
%load_ext autoreload
%autoreload 2
```

```python
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
```

```python
import faiss
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import pairwise_kernels
```

```python
# Define similar interface for each method

# Gamma has been set to 1 / num_features * 0.1
# Otherwise all rbf kernel-based methods won't work properly (sparse similarity matrix). 
# The default value of this function is: 1 / num_features

# Faiss
class FaissInterface:
    """ Baseline """
    def __init__(self, x):
        self.name = 'faiss'
        self.index = faiss.IndexFlatL2(x.shape[1])  
        self.index.add(x)
        
    def get(self, q, k):
        _, idx = self.index.search(q, k)
        return idx
    
    
# sklearn.neighbors.NearestNeighbors
class NearestNeighborsInterfalce:
    def __init__(self, x):
        self.name = 'NearestNeighbors'
        self.n_jobs = -1  # PARAM
        self.algorithm='auto'   # PARAM
        self.nn = NearestNeighbors(K, n_jobs=self.n_jobs, algorithm=self.aglgorithm)
        _ = self.nn.fit(x)
        
    def get(self, q, k):
        _, idx = nn.kneighbors(q, k)
        return idx
    
    
# sklearn.metrics.pairwise.rbf_kernel
class rbfKernelInterface:
    def __init__(self, x):
        self.name = 'rbf_kernel'
        self.x = x
        self.gamma = 1 / x.shape[1] * 0.1
        
    def get(self, q, k):

        # Calculate similarity between the query and all the vectors in the dataset
        similarity_matrix = rbf_kernel(q, self.x, gamma=self.gamma).reshape(-1)

        # Get K most similar vectors
        return np.argpartition(similarity_matrix, -k)[-k:]
    
    
# sklearn.gaussian_process.kernels.RBF
class RBFInterface:
    
    def __init__(self, x):
        self.name = 'RBF'
        self.x = x
        self.rbf = RBF()
        
    def get(self, q, k):
        similarity_matrix = self.rbf(query, self.x).reshape(-1)
        return np.argpartition(similarity_matrix, -k)[-k:]
    
# sklearn.metrics.pairwise_kernels
class PairwiseMetricsInterface:
    def __init__(self, x):
        self.name = 'pairwise_kernels'
        self.x = x
        self.gamma = 1 / x.shape[1] * 0.1
    
    def get(self, q, k):
        matrix = pairwise_kernels(q, self.x, metric='rbf', gamma=self.gamma).reshape(-1)
        return np.argpartition(matrix, -k)[-k:]
```

```python
# Params
K = [3, 5, 10]
N_SAMPLES = [100, 1000, 10000, 100000]
N_FEATURES = [10, 50, 150]
KNN_IMPLEMENTATIONS = [FaissInterface, rbfKernelInterface, RBFInterface, PairwiseMetricsInterface]
OUTPUT_FILE = 'results/007_knn_algorithms_comparision.csv'

# To avoid unnecessary calculations
SKIP_EXPERIMENT = os.path.isfile(OUTPUT_FILE)
```

# Experiments

```python
# Main loop
if not SKIP_EXPERIMENT:
    
    results = []
    # Grid search
    for k in K:
        for n_samp in N_SAMPLES:
            for n_feat in N_FEATURES:

                # Create dataset
                x, _ = make_classification(n_samp, n_feat)
                x = x.astype('float32')
                baseline = None

                # Create query
                query = x[50].reshape(1, -1)

                # Compare methods
                for knn_method in KNN_IMPLEMENTATIONS:

                    # Initialize and calculate the time
                    algorithm_object = knn_method(x)
                    print(f'\n {algorithm_object.name} (k={k}) {n_samp}x{n_feat} :', end='\n\t')
                    calculated_time = %timeit -n 1000 -o algorithm_object.get(query, k)

                    # Define baseline
                    if algorithm_object.name == 'faiss':
                        baseline = set(algorithm_object.get(query, k).flatten().tolist())
                        print(f'\tNew baseline: {baseline}')

                    # Check output
                    example_output = algorithm_object.get(query, k).flatten().tolist()
                    similar_to_baseline = all([value in baseline for value in example_output])
                    print(f'\tSimilar to baseline: {similar_to_baseline}')
                    results.append({
                        'algorithm': algorithm_object.name,
                        'k': k,
                        'n_samples': n_samp,
                        'n_features': n_feat,
                        'similar_to_baseline': similar_to_baseline,
                        'average_time': calculated_time.average,
                        'average_time_std': calculated_time.stdev,
                    })
                    
    # Save the results
    data = pd.DataFrame(results)
    data.to_csv(OUTPUT_FILE)
else:
    print(f'f{OUTPUT_FILE} found. Skipping the experiment.')
```

# Results

```python
data = data if not SKIP_EXPERIMENT else pd.read_csv(OUTPUT_FILE)
```

```python
for k in K:
    for ns in N_SAMPLES:
        for nf in N_FEATURES:
            scope = data[(data['k'] == k) & (data['n_features'] == nf) & (data['n_samples'] == ns)]
            best_methods = scope.nsmallest(3, 'average_time')
            results = ' '.join([f'{name} ({round(time * 1000, 4)})' for name, time in zip(best_methods.algorithm, best_methods.average_time)])
            print(f'k = {k} | {ns}x{nf} -> {results} (ms)')
```
