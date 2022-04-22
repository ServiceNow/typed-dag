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

from causaldag import DAG, PDAG
from itertools import combinations, permutations, product
from typing import Tuple


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
                    if (
                        g1[par1, node] == 1
                        and g1[par2, node] == 1
                        and g1[node, par1] == 0
                        and g1[node, par2] == 0
                        and g1[par1, par2] == 0
                        and g1[par2, par1] == 0
                    ):
                        if not (
                            g2[par1, node] == 1
                            and g2[par2, node]
                            and g2[node, par1] == 0
                            and g2[node, par2] == 0
                            and g2[par1, par2] == 0
                            and g2[par2, par1] == 0
                        ):
                            return False

    for node in range(g2.shape[0]):
        for par1 in range(g2.shape[0]):
            for par2 in range(g2.shape[0]):
                # check if parents are not married
                if par1 != par2:
                    if (
                        g2[par1, node] == 1
                        and g2[par2, node] == 1
                        and g2[node, par1] == 0
                        and g2[node, par2] == 0
                        and g2[par1, par2] == 0
                        and g2[par2, par1] == 0
                    ):
                        if not (
                            g1[par1, node] == 1
                            and g1[par2, node]
                            and g1[node, par1] == 0
                            and g1[node, par2] == 0
                            and g1[par1, par2] == 0
                            and g1[par2, par1] == 0
                        ):
                            return False

    return True


class EmptySetException(Exception):
    pass


def _update_tedge_orientation(G, type_g, types):
    """
    Detects which t-edges are oriented and unoriented and updates the type compatibility graph

    """
    type_g = np.copy(type_g)

    for a, b in permutations(range(G.shape[0]), 2):
        # XXX: No need to consider the same-type case, since the type matrix starts at identity.
        if types[a] == types[b]:
            continue
        # Detect unoriented t-edges
        if G[a, b] == 1 and G[b, a] == 1 and not (type_g[types[a], types[b]] + type_g[types[b], types[a]] == 1):
            type_g[types[a], types[b]] = 1
            type_g[types[b], types[a]] = 1
        # Detect oriented t-edges
        if G[a, b] == 1 and G[b, a] == 0:
            type_g[types[a], types[b]] = 1
            type_g[types[b], types[a]] = 0

    return type_g


def _orient_tedges(G, type_g, types):
    """
    Ensures that edges that belong to oriented t-edges are consistently oriented.

    Note: will not change the orientation of edges that are already oriented, even if they clash with the direction
          of the t-edge. This can happen if the CPDAG was not type consistant at the start of t-Meek.

    """
    G = np.copy(G)
    for a, b in permutations(range(G.shape[0]), 2):
        if type_g[types[a], types[b]] == 1 and type_g[types[b], types[a]] == 0 and G[a, b] == 1 and G[b, a] == 1:
            G[a, b] = 1
            G[b, a] = 0
    return G


def typed_meek(cpdag: np.ndarray, types: list, iter_max: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the Meek algorithm with the type consistency as described in Section 5 (from the CPDAG).

    :param cpdag: adjacency matrix of the CPDAG
    :param types: list of type of each node
    :param iter_max: The maximum number of iterations. If reached, an exception will be raised.
    """
    n_nodes = cpdag.shape[0]
    types = np.asarray(types)
    n_types = len(np.unique(types))

    G = np.copy(cpdag)
    type_g = np.eye(n_types)  # Identity matrix to allow intra-type edges

    # repeat until the graph is not changed by the algorithm
    # or too high number of iteration
    previous_G = np.copy(G)
    i = 0
    while True and i < iter_max:
        """
        Each iteration is broken down into three stages:
        1) Update t-edge orientations based on the CPDAG
        2) Orient all edges that are part of the same t-edge consistently (if the t-edge is oriented)
        3) Apply the Meek rules (only one per iteration) to orient the remaining edges.

        Note: Edges are oriented one-by-one in step 3, but these orientations will be propagated to the whole
              t-edge once we return to step (1).

        """
        i += 1
        # Step 1: Determine the orientation of all t-edges based on t-edges (some may be unoriented)
        type_g = _update_tedge_orientation(G, type_g, types)

        # Step 2: Orient all edges of the same type in the same direction if their t-edge is oriented.
        # XXX: This will not change the orientation of oriented edges (i.e., if the CPDAG was not consistant)
        G = _orient_tedges(G, type_g, types)

        # Step 3: Apply Meek's rules (R1, R2, R3, R4) and the two-type fork rule (R5)
        for a, b, c in permutations(range(n_nodes), 3):
            # Orient any undirected edge a - b as a -> b if any of the following rules is satisfied:
            if G[a, b] != 1 or G[b, a] != 1:
                # Edge a - b is already oriented
                continue

            # R1: c -> a - b ==> a -> b
            if G[a, c] == 0 and G[c, a] == 1 and G[b, c] == 0 and G[c, b] == 0:
                G[b, a] = 0
            # R2: a -> c -> b and a - b ==> a -> b
            elif G[a, c] == 1 and G[c, a] == 0 and G[b, c] == 0 and G[c, b] == 1:
                G[b, a] = 0
            # R5: b - a - c and t(c) = t(b) ==> a -> b and a -> c (two-type fork)
            elif (
                G[a, c] == 1
                and G[c, a] == 1
                and G[b, c] == 0
                and G[c, b] == 0
                and types[b] == types[c]
                and types[b] != types[a]  # Make sure there are at least two types
            ):
                G[b, a] = 0
                G[c, a] = 0
            else:

                for d in range(n_nodes):
                    if d != a and d != b and d != c:
                        # R3: a - c -> b and a - d -> b, c -/- d ==> a -> b, and a - b
                        if (
                            G[a, c] == 1
                            and G[c, a] == 1
                            and G[b, c] == 0
                            and G[c, b] == 1
                            and G[a, d] == 1
                            and G[d, a] == 1
                            and G[b, d] == 0
                            and G[d, b] == 1
                            and G[c, d] == 0
                            and G[d, c] == 0
                        ):
                            G[b, a] = 0
                        # R4: a - d -> c -> b and a - - c ==> a -> b
                        elif (
                            G[a, d] == 1
                            and G[d, a] == 1
                            and G[c, d] == 0
                            and G[d, c] == 1
                            and G[b, c] == 0
                            and G[c, b] == 1
                            and (G[a, c] == 1 or G[c, a] == 1)
                        ):
                            G[b, a] = 0

        if (previous_G == G).all():
            break
        if i >= iter_max:
            raise Exception(f"Typed Meek is stucked. More than {iter_max} iterations.")

        previous_G = np.copy(G)

    return G, type_g


def tmec_enumeration(cpdag: np.ndarray, types: list, type_g: np.ndarray) -> list:
    """
    Finds all the possible DAGs represented by a t-CPDAG
    :param cpdag: A PDAG that does not violate type consistency
    :param types: list of types of each node
    :param type_g: adjacency matrix of the graph over types
    :returns: the list of possible DAGs
    """
    n_nodes = cpdag.shape[0]
    n_types = len(np.unique(types))

    type_g = np.copy(type_g)

    # Find every unoriented t-edge
    unoriented_tedges = []
    for i, j in combinations(range(n_types), 2):
        if type_g[i, j] == 1 and type_g[j, i] == 1:
            unoriented_tedges.append([i, j])

    # Enumerate every possible orientation for each unoriented t-edge
    dag_list = []
    for orientation in product([0, 1], repeat=len(unoriented_tedges)):
        t_cpdag = np.copy(cpdag)
        oriented_type_g = np.copy(type_g)

        # Orient each undirected t-edge with a given orientation
        for i, edge_orientation in enumerate(orientation):
            oriented_type_g[unoriented_tedges[i][0], unoriented_tedges[i][1]] = edge_orientation
            oriented_type_g[unoriented_tedges[i][1], unoriented_tedges[i][0]] = 1 - edge_orientation

        # Orient all unoriented inter-type edges
        for i, j in permutations(range(n_nodes), 2):
            if types[i] == types[j] or t_cpdag[i, j] + t_cpdag[j, i] != 2:
                # Either an intra-type edge or an inter-type that is already oriented. Don't touch.
                continue

            # Make sure the t-edge between these variables exists (i.e., types are connected)
            assert (
                oriented_type_g[types[i], types[j]] + oriented_type_g[types[j], types[i]] == 1
            ), "Found connected nodes with no t-edge."

            # Orient according to t-edge orientation
            if oriented_type_g[types[i], types[j]] == 1:
                t_cpdag[j, i] = 0
            else:
                t_cpdag[i, j] = 0

        # Enumerate all DAGs that can be generated from the current t-CPDAG
        # Edges that remain to be oriented are intra-type.
        for d in PDAG.from_amat(t_cpdag).all_dags():
            d = DAG(nodes=np.arange(t_cpdag.shape[0]), arcs=d).to_amat()[0]
            # Add DAG only if it is acyclic,
            # it does not have extra v-structures,
            # and its skeleton is identical
            # XXX: this last condition is required since sometimes t-edge orientations lead to impossible
            #      t-essential graphs and causaldag returns incorrect graphs with missing edges.
            if (
                is_acyclic(d)
                and has_same_immoralities(d, cpdag)
                and PDAG.from_amat(t_cpdag).skeleton == DAG.from_amat(d).skeleton
            ):
                dag_list.append(d)

    if len(dag_list) == 0:
        raise EmptySetException(
            "Error: t-MEC enumeration returned no valid t-DAG. CPDAG probably violates type consistency or constains a cycle."
        )

    return dag_list


def to_tessential_graph(cpdag: np.ndarray, types: np.ndarray) -> np.ndarray:
    """
    Convert essential graph (CPDAG) into t-essential graph

    :param cpdag: A PDAG that does not violate type consistency
    :param types: list of types of each node
    """
    t_ess_G, type_comp = typed_meek(cpdag, types)

    try:
        tmec_enum = tmec_enumeration(t_ess_G, types, type_comp)
        t_ess_G = (sum(tmec_enum) > 0).astype(int)
        failed = 0
    except EmptySetException as e:
        print("***********", e)
        t_ess_G = cpdag
        failed = 1

    return t_ess_G, failed
