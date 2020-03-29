import torch
from torch import Tensor, FloatTensor
from typing import Callable


def MSE() -> Callable[[Tensor, Tensor], Tensor]:
    def mse(input: Tensor, target: Tensor) -> float:
        return ((target - input) ** 2).mean()
    return mse


def HingeLoss() -> Callable[[Tensor, Tensor], Tensor]:
    def hingeloss(input: Tensor, target: Tensor) -> float:
        return torch.sum(torch.max(torch.Tensor([0]), 0.5 - input * target) ** 2)
    return hingeloss


def entropy(values: Tensor) -> Tensor:
    """ Calculates entropy independently for each vector in a tensor
        Returns results as 1-D tensor """
    _entropy = lambda vector: -1. * torch.sum(torch.Tensor(
        [probability * torch.log2(probability) for probability in vector]))

    output_vector = []

    for vector in values:
        _, counts = torch.unique(vector, return_counts=True)
        probablity_vector = counts * 1. / len(vector)
        output_vector.append(_entropy(probablity_vector))

    return torch.Tensor(output_vector).reshape(-1, 1)


def KNNHingeLoss(train_x: Tensor, train_y: Tensor, k: int = 3, alpha: float = 0.3) -> Callable[[Tensor, Tensor], Tensor]:
    """ 
        Implementation of KNN Hinge Loss

        Note: 
            The variable `train_x` should contain train examples as well as validation examples.
            Otherwise, there is no way to calculate loss on validation dataset.
            We can also create an different object of this loss only for validation purposes.

        TODO: Resolve the method of validation.

     """
    import faiss

    # Create an index (L2)
    num_columns, num_attributes = train_x.shape
    index = faiss.IndexFlatL2(num_attributes)
    index.add(train_x.numpy())
    
    def knn_hingle_loss(predictions: Tensor, target: Tensor, input_data: Tensor) -> Tensor:
        
        assert predictions.shape == target.shape, 'Invalid target shape!'

        # Find k most similar vectors in the training dataset
        scores, indexes = index.search(input_data.numpy(), k + 1)
        
        # Get classes of the most similiar vectors
        knn = train_y[[indexes]][:, 1:]
        
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


def KNNMSELoss(train_x: Tensor, train_y: Tensor, k: int = 3) -> Callable[[Tensor, Tensor], Tensor]:
    """ 
        Implementation of KNN MSE Loss.

        Note: 
            The variable `train_x` should contain train examples as well as validation examples.
            Otherwise, there is no way to calculate loss on validation dataset.
            We can also create an different object of this loss only for validation purposes.

        TODO: Resolve the method of validation.

     """
    import faiss

    num_columns, num_attributes = train_x.shape
    index = faiss.IndexFlatL2(num_attributes)
    index.add(train_x.numpy())

    def knn_mse_loss(prediction: Tensor, target: Tensor, input_data: Tensor) -> Tensor:
        
        assert prediction.shape == target.shape, 'Invalid target shape!'

        # Find k most similar vectors in the training dataset
        scores, indexes = index.search(input_data.numpy(), k + 1)

        # Get classes of the most similiar vectors
        knn = train_y[[indexes]][:, 1:]
        
        # Calculate entropy
        entropies = entropy(knn)
        assert entropies.shape == target.shape, 'Invalid entropies shape!'
        
        entropies = entropies.reshape(-1, 1)
        loss = torch.exp(-1 * entropies) * ((target - prediction) ** 2)

        assert loss.shape == prediction.shape, 'Invalid loss shape!'
        return loss.mean()

    return knn_mse_loss
