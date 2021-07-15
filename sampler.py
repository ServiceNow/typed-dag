import numpy as np
from typing import Tuple
from graph_utils import is_acyclic


def sample_dag(n_nodes: int, prob: float = 0.5) -> np.ndarray:
    """ Sample a DAG of n_nodes variables
    :param n_nodes: Number of nodes in the DAG
    :param prob: Probability of having an edge
    :returns: adjacency matrix of the sampled DAG
    """
    dag = np.zeros((n_nodes, n_nodes))
    permutation = np.zeros((n_nodes, n_nodes))
    order = np.random.choice(n_nodes, size=n_nodes, replace=False)

    for i in range(n_nodes):
        permutation[i, order[i]] = 1
        for j in range(i+1, n_nodes):
            if np.random.rand() < prob:
                dag[i, j] = 1

    # permute the DAG
    dag = np.matmul(permutation, dag)
    dag = np.matmul(dag, permutation.T)

    return dag


def sample_t_dag(n_nodes: int, n_types: int, prob: float) -> Tuple[np.ndarray, np.ndarray, list]:
    """ Sample a t-DAG of n_nodes variables that
    is type ordered
    :param n_nodes: Number of nodes in the t-DAG
    :param n_types: Number of types
    :param prob: Probability of having an edge
    :returns: 1) adjacency matrix of the sampled t-DAG
              2) adjacency matrix over the types
              3) the list of types associated to each node

    Note: the sampling make sure at least one representative
    of each type is present in the t-DAG
    """
    # sample types
    if n_nodes < n_types:
        raise ValueError(f"n_nodes ({n_nodes}) has to be lower than n_types ({n_types})")
    elif n_nodes == n_types:
        types = np.arange(n_types)
    else:
        types_first = np.arange(n_types)
        types_last = np.random.choice(n_types, size=n_nodes - n_types, replace=True)
        types = np.concatenate([types_first, types_last])
    np.random.shuffle(types)
    order = np.arange(n_nodes)
    np.random.shuffle(order)

    # sample matrix of type interaction
    type_matrix = np.zeros((n_types, n_types))
    for i in range(n_types - 1):
        for j in range(i + 1, n_types):
            if np.random.rand() < 0.5:
                type_matrix[i, j] = prob
            else:
                type_matrix[j, i] = prob

    count = 0
    while count < 10:
        # sample t-DAG
        dag = np.zeros((n_nodes, n_nodes))
        for i, node in enumerate(order):
            for j in range(i + 1):
                if np.random.rand() < type_matrix[types[order[j]], types[order[i]]]:
                    dag[order[j], order[i]] = 1

        if is_acyclic(dag):
            break
        count += 1

    if count >= 10:
        raise ValueError("Could not sample an acyclic graph")

    return dag, type_matrix, types
