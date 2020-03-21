import faiss
import torch
from torch import Tensor, FloatTensor
from typing import Callable

# Example
def MSE() -> Callable[[Tensor, Tensor], Tensor]:
    def mse(prediction: Tensor, target: Tensor) -> float:
        return ((target - prediction) ** 2).mean()
    return mse

def KNNHingeLoss(train_dataset: Tensor, train_predictions: Tensor, k: int = 3, alpha = 0.3) -> Callable[[Tensor, Tensor], Tensor]:
    """ 
        Implementation of KNN Hinge Loss

        Note: 
            The variable `train_dataset` should contain train examples as well as validation examples.
            Otherwise, there is no way to calculate loss on validation dataset.
            We can also create an different object of this loss only for validation purposes.

        TODO: Resolve the method of validation.

     """
    # Create an index (L2)
    num_columns, num_attributes = train_dataset.shape
    index = faiss.IndexFlatL2(num_attributes)
    index.add(train_dataset.numpy())
    
    def knn_hingle_loss(predictions: Tensor, target: Tensor, input_data: Tensor) -> Tensor:
        
        assert predictions.shape == target.shape, 'Invalid target shape!'

        # Find k most similar vectors in the training dataset
        scores, indexes = index.search(input_data.numpy(), k + 1)
        
        # Get classes of the most similiar vectors
        knn = train_predictions[[indexes]][:, 1:]
        
        # Get and return most frequent class
        knn_classes, _ = torch.mode(knn, dim=1)
        knn_classes = knn_classes.reshape(-1, 1)

        # Calculate loss value

        loss = torch.max(torch.zeros_like(knn_classes), 1 - target * predictions 
                         + alpha * (torch.abs(target - knn_classes) * torch.abs(target - predictions))
                         + torch.abs(target - predictions) * torch.abs(knn_classes - predictions))
                        
        assert loss.shape == predictions.shape, 'Invalid loss shape!'
        return loss.mean()
    
    return knn_hingle_loss


def KNNMSELoss(train_dataset: Tensor, train_predictions: Tensor, k: int = 3) -> Callable[[Tensor, Tensor], Tensor]:
    """ 
        Implementation of KNN MSE Loss.

        Note: 
            The variable `train_dataset` should contain train examples as well as validation examples.
            Otherwise, there is no way to calculate loss on validation dataset.
            We can also create an different object of this loss only for validation purposes.

        TODO: Resolve the method of validation.

     """

    def entropy(vector: Tensor) -> Tensor:
        """ Calculates Entropy using `torch` components. Works only with 1-D vectors. """
        return -1. * torch.sum(torch.Tensor([probability * torch.log2(probability) for probability in vector]))


    def calculate_entropy_of_tensor(values: Tensor, base_probability: float) -> Tensor:
        """ Calculates entropy independently for each vector in a tensor
            Returns results as 1-D tensor """
        output_vector = []
        for single_example in values:
            _, counts = torch.unique(single_example, return_counts=True)
            probablity_vector = counts * base_probability
            output_vector.append(entropy(probablity_vector))

        return torch.Tensor(output_vector).reshape(-1, 1)

    num_columns, num_attributes = train_dataset.shape
    index = faiss.IndexFlatL2(num_attributes)
    index.add(train_dataset.numpy())

    def knn_mse_loss(prediction: Tensor, target: Tensor, input_data: Tensor) -> Tensor:
        
        assert prediction.shape == target.shape, 'Invalid target shape!'

        # Find k most similar vectors in the training dataset
        scores, indexes = index.search(input_data.numpy(), k + 1)

        # Get classes of the most similiar vectors
        knn = train_predictions[[indexes]][:, 1:]
        
        # Calculate entropy
        base_probability = 1. / k
        entropies = calculate_entropy_of_tensor(knn, base_probability)
        assert entropies.shape == target.shape, 'Invalid entropies shape!'
        
        entropies = entropies.reshape(-1, 1)
        loss = torch.exp(entropies) * ((target - prediction) ** 2)

        assert loss.shape == prediction.shape, 'Invalid loss shape!'
        return loss.mean()

    return knn_mse_loss

