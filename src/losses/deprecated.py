import torch
from torch import Tensor
from typing import Callable

from src.functional import entropy


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

    # Deprecated
    print('Warning: This loss function (KNNHingeLoss) is deprecated \
        and will be removed in a future version.')

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

    # Deprecated
    print('Warning: This loss function (KNNMSELoss) is deprecated \
        and will be removed in a future version.')

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
