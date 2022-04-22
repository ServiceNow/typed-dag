import numpy as np


class Categorial:
    """
    Categorial distribution, used to sample types
    if p not specified, assign equal probabilities to each type
    :param n_types: number of types, k
    :param p: parameters (p_1,...,p_k) of the distribution
    """

    def __init__(self, n_types: int, p: list = None):
        self.n_types = n_types
        if p is None:
            p = [1.0 / n_types] * n_types
        self.p = p

    def __call__(self, n_nodes: int):
        """
        Sample n_nodes samples from the distribution
        :param n_nodes: number of samples (= number of nodes in the graph)
        :returns: nd.array of samples
        """
        return np.random.choice(self.n_types, n_nodes, p=self.p)


class Covering_Categorial:
    """
    Compared to a categorial distribution, a node is associated to
    each type and then we proceed as a categorical distribution
    to sample the other types
    if p not specified, assign equal probabilities to each type
    :param n_types: number of types, k
    :param p: parameters (p_1,...,p_k) of the distribution
    """

    def __init__(self, n_types: int, p: list = None):
        self.n_types = n_types
        if p is None:
            p = [1.0 / n_types] * n_types
        self.p = p

    def __call__(self, n_nodes: int):
        """
        Sample n_nodes samples from the distribution
        :param n_nodes: number of samples (= number of nodes in the graph)
        :returns: nd.array of samples
        """
        if n_nodes < self.n_types:
            # raise ValueError(f"n_nodes ({n_nodes}) has to be lower than \
            #                  n_types ({self.n_types})")
            types = np.random.choice(self.n_types, n_nodes, p=self.p)
        elif n_nodes == self.n_types:
            types = np.arange(self.n_types)
        else:
            first_types = np.arange(self.n_types)
            last_types = np.random.choice(self.n_types, size=n_nodes - self.n_types, p=self.p, replace=True)
            types = np.concatenate([first_types, last_types])
        np.random.shuffle(types)

        return types


class GaussianList:
    """
    List of Gaussian distributions (one per type), used to sample entities
    :param n_types: number of types, k
    :param dim: dimension of each Gaussian
    :param mu: nd.array of size (n_types, dim)
    :param cov: covariance sigma, nd.array of size (n_types, dim, dim)
    """

    def __init__(self, n_types: int, dim: int, mu: np.ndarray = None, cov: np.ndarray = None):
        self.n_types = n_types
        self.dim = dim

        if mu is None:
            mu = np.random.uniform(low=0, high=100, size=(n_types, dim))
        if cov is None:
            cov = np.identity(dim).reshape(1, dim, dim)
            cov = np.broadcast_to(cov, (n_types, dim, dim))
        self.mu = mu
        self.cov = cov

    def __call__(self, types: np.ndarray) -> np.ndarray:
        """
        Sample n_nodes samples given the types
        :param types: the type of each variable, nd.array (n)
        :returns: data
        """
        n_nodes = types.shape[0]
        data = np.zeros((n_nodes, self.dim))
        for i, c in enumerate(types):
            data[i] = np.random.multivariate_normal(self.mu[c], self.cov[c])

        return data


class EtaBeta:
    """
    Beta distribution to sample eta, the prob of linking types together
    :param b1: 1st parameter of the beta distribution
    :param b2: 2nd parameter of the beta distribution
    """

    def __init__(self, b1: int = 0.5, b2: int = 0.5):
        self.b1 = b1
        self.b2 = b2

    def __call__(self, order: np.ndarray) -> np.ndarray:
        """
        Sample a matrix eta given an order
        :param order: a permutation of the node indices
        :returns: eta (n_types, n_types)
        """
        n_types = order.shape[0]
        eta = np.zeros((n_types, n_types))

        for i, o in enumerate(order):
            for j in range(i):
                eta[o, order[j]] = np.random.beta(self.b1, self.b2)

        return eta
