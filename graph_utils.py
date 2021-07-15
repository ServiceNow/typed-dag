import numpy as np


def is_acyclic(adjacency: np.ndarray) -> bool:
    """
    Check if adjacency matrix is acyclic
    :param adjacency: adjacency matrix
    :returns: True if acyclic
    """
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0:
            return False
    return True


def has_same_immoralities(g1: np.ndarray, g2: np.ndarray) -> bool:
    """
    Check if g1 and g2 have the same immoralities
    :param g1: adjacency matrix of a graph (can be only partially directed)
    :param g2: adjacency matrix of a graph (can be only partially directed)
    :returns: True if they have the same immoralities
    """
    for node in range(g1.shape[0]):
        for par1 in range(g1.shape[0]):
            for par2 in range(g1.shape[0]):
                # check if have parents that are not married
                if par1 != par2:
                    if g1[par1, node] == 1 and g1[par2, node] == 1 and \
                       g1[node, par1] == 0 and g1[node, par2] == 0 and \
                       g1[par1, par2] == 0 and g1[par2, par1] == 0:
                        if not (g2[par1, node] == 1 and g2[par2, node] and
                                g2[node, par1] == 0 and g2[node, par2] == 0 and
                                g2[par1, par2] == 0 and g2[par2, par1] == 0):
                            return False

    for node in range(g2.shape[0]):
        for par1 in range(g2.shape[0]):
            for par2 in range(g2.shape[0]):
                # check if parents are not married
                if par1 != par2:
                    if g2[par1, node] == 1 and g2[par2, node] == 1 and \
                       g2[node, par1] == 0 and g2[node, par2] == 0 and \
                       g2[par1, par2] == 0 and g2[par2, par1] == 0:
                        if not (g1[par1, node] == 1 and g1[par2, node] and
                                g1[node, par1] == 0 and g1[node, par2] == 0 and
                                g1[par1, par2] == 0 and g1[par2, par1] == 0):
                            return False

    return True


def get_children(dag: np.ndarray, nodes: list) -> list:
    """
    Return the index of the children of the nodes 'nodes'
    :param dag: adjacency matrix of the DAG
    :param nodes: list of nodes to get the children
    :returns: the list of children index
    """
    rows = dag[nodes, :]
    row = np.sum(rows, axis=0)
    row = (row > 0) * 1
    children = row * (np.arange(dag.shape[0]) + 1)

    return (children[children > 0] - 1).astype(int)


def has_directed_path(dag: np.ndarray, a: int, b: int) -> bool:
    """
    Check if there are some directed path from a to b
    :param dag: the adjacency matrix
    :param a: the index of the first node
    :param b: the index of the second node
    :returns: True if is there is a path, else False
    """
    children = []
    nodes = [a]

    # remove undirected edges
    for i in range(dag.shape[0]):
        for j in range(dag.shape[0]):
            if dag[i, j] == 1 and dag[j, i] == 1:
                dag[i, j] = 0
                dag[j, i] = 0

    for i in range(dag.shape[0]):
        children = get_children(dag, nodes)
        if b in children:
            return True
        nodes = children

    return False


def dag_union(dag_list: list) -> np.ndarray:
    """
    Takes a list of graph and outputs
    a graph that is the union of it.
    (If -> and <- then output --)
    :param dag_list: a list of DAG (np.ndarray)
    :returns: the graph
    """
    graph = np.zeros_like(dag_list[0])

    for dag in dag_list:
        graph += dag

    return (graph > 0) * 1.
