"""
Copyright 2021 ServiceNow

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import numpy as np
from typing import Tuple
from graph_utils import is_acyclic


def sample_t_dag(
    n_nodes: int, n_types: int, prob_inter: float, prob_intra: float, type_no_sources: bool = False
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Sample a t-DAG of n_nodes variables that
    is type ordered
    :param n_nodes: Number of nodes in the t-DAG
    :param n_types: Number of types
    :param prob_inter: Probability of having an edge between types
    :param prob_intra: Probability of having an edge within types
    :param type_no_sources: If True, all types will have at least one parent type. Otherwise, no constraint.
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
        # Make sure that each type appears at least once
        types_first = np.arange(n_types)
        types_last = np.random.choice(n_types, size=n_nodes - n_types, replace=True)
        types = np.concatenate([types_first, types_last])

    # Randomly assign types to variables
    np.random.shuffle(types)

    # Randomly assign an order in the permutation to variables
    order = np.arange(n_nodes)
    np.random.shuffle(order)

    # Sample matrix of type interaction
    types_ok = False
    while not types_ok:
        type_matrix = np.zeros((n_types, n_types))
        for i in range(n_types - 1):
            for j in range(i + 1, n_types):
                # Orient t-edges randomly
                if np.random.rand() < 0.5:
                    type_matrix[i, j] = prob_inter
                else:
                    type_matrix[j, i] = prob_inter

        types_ok = not type_no_sources or np.all(type_matrix.sum(axis=0) > 0)

    # Add intra-type t-edges
    type_matrix[np.arange(type_matrix.shape[0]), np.arange(type_matrix.shape[0])] = prob_intra

    # sample t-DAG
    dag = np.zeros((n_nodes, n_nodes))
    for i, node in enumerate(order):
        for j in range(i):
            if np.random.rand() < type_matrix[types[order[j]], types[order[i]]]:
                dag[order[j], order[i]] = 1

    assert is_acyclic(dag), "Produced a cyclic graph. That's not normal."

    return dag, type_matrix, types
