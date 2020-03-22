# Collective Loss Functions

## 1. Introduction
The problem we are addressing is the design of a new family of loss functions in deep learning. Currently used functions measure the classification/regression error on a per-individual basis, without considering the characteristic or importance of an instance being classified. We want to include the weight of an instance based on the typicality of the instance as measured by the entropy of that instance’s nearest neighborhood in the similarity graph. 

The idea is to first create an instance similarity graph, which computes pairwise similarities between instances, and using a cut-off threshold (based on quantiles or desired density of the resulting graph) creates an instance similarity graph. For each instance we can compute the homogeneity of the neighborhood of that instance. If there are many instances of the same class in the neighborhood, the focal instance is a typical example of its class and should therefore be classified correctly, otherwise the resulting model will not generalize well. On the other hand, if the entropy of the neighborhood of an instance is large (the instance is similar to both instances of its class and other classes), then the error committed on such instance is less consequential.

This way we will be able to put more emphasis on the classification error of important instances, hopefully leading to better generalization of computed models.

## 2. Milestones
The project will take 6 month, we want to present the results at one of the major machine learning conferences. Key milestones are: 
- developing implementation of a collective loss function in PyTorch
- running excessive experiments on benchmark datasets

If time and resources permit, we would like to extend our experiments to recurrent neural networks as well, but first we must validate our idea on simple feed forward architectures. 

## 3. Installation
### 3.1 CPU
``` 
conda env create -f environment.yml 
```

### 3.2 GPU
``` 
(TODO) conda env create -f environment-gpu.yml 
```

## 4. Usage
``` 
(TODO) python run_experiment <experiment name> 
```

## 5. References
*Under construction*
