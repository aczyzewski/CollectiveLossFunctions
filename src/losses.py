import faiss
import torch
from torch import Tensor, FloatTensor
from typing import Callable

def MSE() -> Callable[[Tensor, Tensor], Tensor]:
    def mse(prediction: Tensor, target: Tensor) -> float:
        return ((target - prediction) ** 2).mean()
    return mse

def KNNHingeLoss(train_dataset: Tensor, train_predictions: Tensor, k: int = 3, alpha = 0.3) -> Callable[[Tensor, Tensor], Tensor]:
    
    # Create an index (L2)
    num_columns, num_attributes = train_dataset.shape
    index = faiss.IndexFlatL2(num_attributes)
    index.add(train_dataset.numpy())
    
    def knn_hingle_loss(predictions: Tensor, target: Tensor, input_data: Tensor) -> Tensor:
        
        # Find k most similar vectors in the training dataset
        scores, indexes = index.search(input_data.numpy(), k + 1)
        
        # Get classes of the most similiar vectors
        knn = train_predictions[[indexes]][:, 1:]
        
        # Get and return most frequent class
        knn_classes, _ = torch.mode(knn, dim=1)
        
        # Calculate loss value
        loss = torch.max(torch.zeros_like(knn_classes), 1 - target * predictions 
                         + alpha * (torch.abs(target - knn_classes) * torch.abs(target - predictions))
                         + torch.abs(target - predictions) * torch.abs(knn_classes - predictions))
                         
        return loss.mean()
    
    return knn_hingle_loss