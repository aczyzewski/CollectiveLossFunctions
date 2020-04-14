from typing import Tuple

import faiss
import numpy as np


class AbstractKNN:
    """ Should be used to create an unified interface of any NN library. """

    def __init__(self, data_x: np.array, data_y: np.array,
                 metric: str = 'L2', use_gpu: bool = False,
                 verbose: bool = False) -> None:

        self.data_x = data_x.astype('float32')
        self.data_y = data_y.astype('int32')
        self.use_gpu = use_gpu
        self.metric = metric
        self.verbose = verbose

    def get(self, query: np.array, k: int, exclude_query: bool = True
            ) -> Tuple[np.array, np.array, np.array]:
        """ Returns kNN of a query.

            Arguments:
                exclude_query (bool): if a query is in the dataset
                    the function will remove the query from results.
                return_distances (bool): if True the the function will
                    return calculated distances also

            Returns:
                (neighboors, distances, classes)
        """
        raise NotImplementedError


class FaissKNN(AbstractKNN):
    """ Interface of Facebook's faiss library

        Available metrics:
            Supported: METRIC_L2
            Not supported: METRIC_INNER_PRODUCT, METRIC_L1, METRIC_Linf,
                METRIC_Lp, METRIC_Canberra, METRIC_BrayCurtis,
                METRIC_JensenShannon
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.metric != 'L2':
            raise Exception('Warning: faiss wraper doesn\'t support other \
                            metrics than L2 in current version.')

        if self.use_gpu:
            print('Warning: (faiss) GPU is not supported in current version.')

        # Create an index
        num_columns, num_attributes = self.data_x.shape
        self.index = faiss.IndexFlatL2(num_attributes)
        self.index.add(self.data_x)

    def get(self, query: np.array, k: int, exclude_query: bool = False,
            ) -> Tuple[np.array, np.array, np.array]:

        if not isinstance(query, np.float32):
            print('Warning: Query should be represented as float32 array')
            query = query.astype('float32')

        # Find K most similar instances in the input
        _k = k if not exclude_query else k + 1
        distances, indices = self.index.search(query, _k)

        if exclude_query:
            distances = distances[:, 1:]
            indices = indices[:, 1:]

        # Retrieve classes of the indicies
        classes = self.data_y[tuple([indices])]

        return distances, indices, classes
