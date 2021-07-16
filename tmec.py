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
import itertools
from typing import Tuple
from graph_utils import is_acyclic, has_same_immoralities


def typed_meek(cpdag: np.ndarray, types: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the Meek algorithm with the type consistency
    as described in Section 5 (from the CPDAG)
    :param cpdag: adjacency matrix of the CPDAG
    :param types: list of type of each node
    """
    n_nodes = cpdag.shape[0]
    types = np.asarray(types)
    k = len(np.unique(types))
    i = 0
    iter_max = 10
    first = True

    p = np.copy(cpdag)
    previous_p = None
    type_g = np.zeros((k, k))

    # repeat until the graph is not changed by the algorithm
    # or too high number of iteration
    while True and i < iter_max:
        i += 1
        # find edges that are already oriented set in type_g
        for a in range(n_nodes):
            for b in range(n_nodes):
                if a != b:
                    if p[a, b] == 1 and p[b, a] == 1 and \
                       not (type_g[types[a], types[b]] +
                       type_g[types[b], types[a]] == 1):

                        type_g[types[a], types[b]] = 1
                        type_g[types[b], types[a]] = 1
                    if p[a, b] == 1 and p[b, a] == 0:
                        type_g[types[a], types[b]] = 1
                        type_g[types[b], types[a]] = 0

        # Step 2: Orient all edges of the same type in the same direction
        for a in range(n_nodes):
            for b in range(n_nodes):
                if a != b:
                    if type_g[types[a], types[b]] == 1 and \
                       type_g[types[b], types[a]] == 0 and \
                       p[a, b] == 1 and p[b, a] == 1:
                        p[a, b] == 1
                        p[b, a] == 0

        # Step 3: Apply Meek's rules (R1, R2, R3, R4) and the two-type fork rule (R5)
        for a in range(n_nodes):
            for b in range(n_nodes):
                # if edge a - b is undirected, orient as a -> b
                # if it falls into one of the five rules
                if p[a, b] == 1 and p[b, a] == 1:
                    for c in range(n_nodes):
                        if a != c and b != c:
                            # R1: c -> a - b ==> a -> b
                            if p[a, c] == 0 and p[c, a] == 1 and p[b, c] == 0 and p[c, b] == 0:
                                p[b, a] = 0
                            # R2: a -> c -> b and a - b ==> a -> b
                            elif p[a, c] == 1 and p[c, a] == 0 and p[b, c] == 0 and p[c, b] == 1:
                                p[b, a] = 0
                            # R5: b - a - c and t(c) = t(b) ==> a -> b and a -> c
                            # (two-type fork)
                            elif p[a, c] == 1 and p[c, a] == 1 and types[b] == types[c]:
                                p[b, a] = 0
                                p[c, a] = 0
                            else:

                                for d in range(n_nodes):
                                    if d != a and d != b and d != c:
                                        # R3: a - c -> b and a - d -> b and c -/- d ==> a -> b
                                        if (p[a, c] == 1 and p[c, a] == 1 and
                                           p[b, c] == 0 and p[c, b] == 1 and
                                           p[a, d] == 1 and p[d, a] == 1 and
                                           p[b, d] == 0 and p[d, b] == 1 and
                                           p[c, d] == 0 and p[d, c] == 0):
                                            p[b, a] = 0
                                        # R4: a - d -> c -> b and a - - c ==> a -> b
                                        elif (p[a, d] == 1 and p[d, a] == 1 and
                                              p[c, d] == 0 and p[d, c] == 1 and
                                              p[b, c] == 0 and p[c, b] == 1 and
                                              (p[a, c] == 1 or p[c, a] == 1)):
                                            p[b, a] = 0

        if not first and (previous_p == p).all():
            break
        elif first:
            first = False
        previous_p = np.copy(p)
        if i >= iter_max:
            raise Exception(f"Typed Meek is stucked. More than {iter_max} iterations.")

    return p, type_g


def tmec_enumeration(dag_: np.ndarray, types: list, g_compatibility_: np.ndarray) -> list:
    """
    Finds all the possible DAGs represented by a t-CPDAG
    :param dag_: adjacency matrix of the t-CPDAG
    :param types: list of types of each node
    :param g_compatibility_: adjacency matrix of the graph over types
    :returns: the list of possible DAGs
    """
    n_nodes = dag_.shape[0]
    n_types = len(np.unique(types))
    dag_list = []
    edges_type = []
    g = np.copy(dag_)
    g_compatibility = np.copy(g_compatibility_)

    # orient every unoriented edges in g_class
    for i in range(n_types):
        for j in range(i+1, n_types):
            if g_compatibility[i, j] == 1 and g_compatibility[j, i] == 1:
                edges_type.append([i, j])

    # cartesian product to generate all the possibilities
    universe = list(itertools.product([0, 1], repeat=len(edges_type)))
    print(f"Size of class: {len(universe)}")

    # enumerate all the DAG in universe
    for instance in universe:
        dag = np.copy(g)
        g_comp = np.copy(g_compatibility)

        for i, edge in enumerate(instance):
            g_comp[edges_type[i][0], edges_type[i][1]] = edge
            g_comp[edges_type[i][1], edges_type[i][0]] = 1 - edge

        for i in range(n_nodes):
            for j in range(n_nodes):
                if dag[i, j] == 1 and dag[j, i] == 1:
                    if g_comp[types[i], types[j]] == 1:
                        dag[j, i] = 0
                    elif g_comp[types[j], types[i]] == 1:
                        dag[i, j] = 0

        # and add DAG only if it is acyclic
        # and if it does have extra v-structures
        if is_acyclic(dag) and has_same_immoralities(dag, dag_):
            dag_list.append(dag)

    return dag_list
