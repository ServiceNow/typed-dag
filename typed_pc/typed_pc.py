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

>> Different immorality/two-type fork orientation methods for a typed version of PC

"""
import logging
import networkx as nx
import numpy as np

from collections import Counter, defaultdict
from itertools import combinations, permutations
from typing import Tuple
from random import shuffle

from graph_utils import _has_both_edges, _has_any_edge, _orient, type_of


#######
# This is the part where we orient all immoralities and two-type forks.
# The behavior used to orient t-edges depends on the chosen strategy:
#   * Naive: orient as first encountered orientation
#   * Majority: orient using the most frequent orientation
#######


def orient_forks_naive(skeleton, sep_sets):
    """
    Orient immoralities and two-type forks

    Strategy: naive -- orient as first encountered

    """
    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()

    # Orient all immoralities and two-type forks
    # TODO: DEBUG using shuffling to test hypothesis
    combos = list(combinations(node_ids, 2))
    shuffle(combos)
    for (i, j) in combos:
        adj_i = set(dag.successors(i))
        adj_j = set(dag.successors(j))

        # If j is a direct child of i
        if j in adj_i:
            continue

        # If i is a direct child of j
        if i in adj_j:
            continue

        # If i and j are directly connected, continue.
        if sep_sets[i][j] is None:
            continue

        common_k = adj_i & adj_j  # Common direct children of i and j
        for k in common_k:
            # Case: we have an immorality i -> k <- j
            if k not in sep_sets[i][j] and k in dag.successors(i) and k in dag.successors(j):
                # XXX: had to add the last two conditions in case k is no longer a child due to t-edge orientation
                logging.debug(
                    f"S: orient immorality {i} (t{type_of(dag, i)}) -> {k} (t{type_of(dag, k)}) <- {j} (t{type_of(dag, j)})"
                )
                _orient(dag, i, k)
                _orient(dag, j, k)

            # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
            elif (
                type_of(dag, i) == type_of(dag, j)
                and type_of(dag, i) != type_of(dag, k)
                and _has_both_edges(dag, i, k)
                and _has_both_edges(dag, j, k)
            ):
                logging.debug(
                    f"S: orient two-type fork {i} (t{type_of(dag, i)}) <- {k} (t{type_of(dag, k)}) -> {j} (t{type_of(dag, j)})"
                )
                _orient(dag, k, i)  # No need to orient k -> j. Will be done in this call since i,j have the same type.

    return dag


def orient_forks_majority_top1(skeleton, sep_sets):
    """
    Orient immoralities and two-type forks

    Strategy: majority -- orient using the most frequent orientation
    Particularity: Find the t-edge with most evidence, orient, repeat evidence collection.

    """
    dag = skeleton.to_directed()
    node_ids = skeleton.nodes()
    n_types = len(np.unique([type_of(dag, n) for n in dag.nodes()]))

    oriented_tedge = True
    while oriented_tedge:

        # Accumulator for evidence of t-edge orientation
        # We will count how often we see the t-edge in each direction and choose the most frequent one.
        tedge_evidence = np.zeros((n_types, n_types))
        oriented_tedge = False

        # Some immoralities will contain edges between variables of the same type. These will not be
        # automatically oriented when we decide on the t-edge orientations. To make sure that we orient
        # them correctly, we maintain a list of conditional orientations, i.e., how should an intra-type
        # edge be oriented if we make a specific orientation decision for the t-edges.
        conditional_orientations = defaultdict(list)

        # Step 1: Gather evidence of orientation for all immoralities and two-type forks that involve more than one type
        for (i, j) in combinations(node_ids, 2):
            adj_i = set(dag.successors(i))
            adj_j = set(dag.successors(j))

            # If j is a direct child of i, i is a direct child of j, or ij are directly connected
            if j in adj_i or i in adj_j or sep_sets[i][j] is None:
                continue

            for k in adj_i & adj_j:  # Common direct children of i and j
                # Case: we have an immorality i -> k <- j
                if k not in sep_sets[i][j]:
                    # Check if already oriented
                    # if not _has_both_edges(dag, i, k) or not _has_both_edges(dag, j, k):
                    #     continue
                    if not _has_both_edges(dag, i, k) and not _has_both_edges(dag, j, k):
                        # Fully oriented
                        continue

                    # Ensure that we don't have only one type. We will orient these later.
                    if type_of(dag, i) == type_of(dag, j) == type_of(dag, k):
                        continue

                    logging.debug(
                        f"Step 1: evidence of orientation {i} (t{type_of(dag, i)}) -> {k} (t{type_of(dag, k)}) <- {j} (t{type_of(dag, j)})"
                    )
                    # Increment t-edge orientation evidence
                    tedge_evidence[type_of(dag, i), type_of(dag, k)] += 1
                    tedge_evidence[type_of(dag, j), type_of(dag, k)] += 1

                    # Determine conditional orientations
                    conditional_orientations[(type_of(dag, j), type_of(dag, k))].append((i, k))
                    conditional_orientations[(type_of(dag, i), type_of(dag, k))].append((j, k))

                # Case: we have an orientable two-type fork, i.e., it is not an immorality, so i <- k -> j
                elif type_of(dag, i) == type_of(dag, j) and type_of(dag, i) != type_of(dag, k):
                    # Check if already oriented
                    if not _has_both_edges(dag, i, k) or not _has_both_edges(dag, j, k):
                        continue

                    logging.debug(
                        f"Step 1: evidence of orientation {i} (t{type_of(dag, i)}) <- {k} (t{type_of(dag, k)}) -> {j} (t{type_of(dag, j)})"
                    )
                    # Count evidence only once per t-edge
                    tedge_evidence[type_of(dag, k), type_of(dag, i)] += 2

        # Step 2: Orient t-edges based on evidence
        np.fill_diagonal(tedge_evidence, 0)
        ti, tj = np.unravel_index(tedge_evidence.argmax(), tedge_evidence.shape)
        if np.isclose(tedge_evidence[ti, tj], 0):
            continue

        # Orient!
        print("Evidence", tedge_evidence[ti, tj])
        print(conditional_orientations)
        oriented_tedge = True
        first_ti = [n for n in dag.nodes() if type_of(dag, n) == ti][0]
        first_tj = [n for n in dag.nodes() if type_of(dag, n) == tj][0]
        logging.debug(
            f"Step 2: orienting t-edge according to max evidence. t{ti} -> t{tj} ({tedge_evidence[ti, tj]}) vs t{ti} <- t{tj} ({tedge_evidence[tj, ti]})"
        )
        _orient(dag, first_ti, first_tj)
        cond = Counter(conditional_orientations[ti, tj])
        for (n1, n2), count in cond.items():
            logging.debug(f"... conditional orientation {n1}->{n2} (count: {count}).")
            if (n2, n1) in cond and cond[n2, n1] > count:
                logging.debug(
                    f"Skipping this one. Will orient its counterpart ({n2}, {n1}) since it's more frequent: {cond[n2, n1]}."
                )
            else:
                _orient(dag, n1, n2)
    logging.debug("Steps 1-2 completed. Moving to single-type forks.")

    # Step 3: Orient remaining immoralities (all variables of the same type)
    for (i, j) in combinations(node_ids, 2):
        adj_i = set(dag.successors(i))
        adj_j = set(dag.successors(j))

        # If j is a direct child of i, i is a direct child of j, ij are directly connected
        if j in adj_i or i in adj_j or sep_sets[i][j] is None:
            continue

        for k in adj_i & adj_j:  # Common direct children of i and j
            # Case: we have an immorality i -> k <- j
            if k not in sep_sets[i][j]:
                # Only single-type immoralities
                if not (type_of(dag, i) == type_of(dag, j) == type_of(dag, k)):
                    continue
                logging.debug(
                    f"Step 3: orient immorality {i} (t{type_of(dag, i)}) -> {k} (t{type_of(dag, k)}) <- {j} (t{type_of(dag, j)})"
                )
                _orient(dag, i, k)
                _orient(dag, j, k)

    return dag
