from typing import Tuple

import faiss
import numpy as np


class AbstractKNN:
    """ Should be used to create an unified interface of any NN library. """

    def __init__(self, data_x: np.array, data_y: np.array, metric: str = 'L2',
                 k: int = 3, use_gpu: bool = False, verbose: bool = False,
                 precompute: bool = False) -> None:

        self.data_x = data_x
        self.data_y = data_y
        self.k = k
        self.use_gpu = use_gpu
        self.metric = metric
        self.verbose = verbose
        self.cache = None
        self.precompute = precompute

    def get(self, query: np.array, k: int, exclude_query: bool = True
            ) -> Tuple[np.array, np.array, np.array]:
        """ Returns kNN of a query.

            Args:
                exclude_query (bool): if a query is in the dataset
                    the function will remove it from results.

            Returns:
                (neighboors, distances, classes)
        """
        raise NotImplementedError

    def precompute(self, cache_distances: bool = True,
                   cache_classes: bool = True) -> None:
        """ Caches k-neighboors (distances and classes) for every
            example in the dataset. """
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

        self.data_x = self.data_x.astype('float32')
        self.data_y = self.data_y.astype('int32')
        assert len(self.data_y.shape) == 2 and self.data_y.shape[1] == 1, \
            'Invalid Y data shape!'

        # Create an index
        num_columns, num_attributes = self.data_x.shape
        self.index = faiss.IndexFlatL2(num_attributes)
        self.index.add(self.data_x)

        if self.precompute:
            self.precompute_results()

    def get(self, query: np.array, exclude_query: bool = False,
            use_cache: bool = True) -> Tuple[np.array, np.array, np.array]:
        """ Returns calculated distances, indices, and classes of the nearest
            neighboorhood of a given query. If the query is an integer and
            self.compute parameter is True the function will use cached results

            Note: the query type should be float32
        """

        if self.precompute and self.cache is not None and use_cache:
            # If precompute is True, we assume that user will provide
            # list of indicies instead of a vector of values
            query = query.astype('int')
            return [arr[query] for arr in self.cache]

        # Find K most similar instances in the input
        _k = self.k if not exclude_query else self.k + 1
        distances, indices = self.index.search(query, _k)

        if exclude_query:
            distances = distances[:, 1:]
            indices = indices[:, 1:]

        # Retrieve classes of the indicies
        classes = self.data_y[tuple([indices])].squeeze(2)
        return distances, indices, classes

    def precompute_results(self, remove_data: bool = True) -> None:
        """ Caches neighboorhood of any point in the dataset """
        self.cache = self.get(self.data_x, exclude_query=True,
                              use_cache=False)

