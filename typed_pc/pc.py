"""
Note: This code is a snippet taken from the pcalg package.

"""
import logging
import networkx as nx
import numpy as np

from itertools import combinations, permutations
from typing import Tuple

from graph_utils import _has_both_edges, _has_any_edge, _orient


def pc_meek_rules(dag):
    """
    Step 3: Meek rules portion of the PC algorithm

    """
    node_ids = dag.nodes()

    # For all the combination of nodes i and j, apply the following
    # rules.
    old_dag = dag.copy()
    while True:
        for (i, j) in permutations(node_ids, 2):
            # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
            # such that k and j are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Look all the predecessors of i.
                for k in dag.predecessors(i):
                    # Skip if there is an arrow i->k.
                    if dag.has_edge(i, k):
                        continue
                    # Skip if k and j are adjacent.
                    if _has_any_edge(dag, k, j):
                        continue
                    # Make i-j into i->j
                    logging.debug("R1: remove edge (%s, %s)" % (j, i))
                    _orient(dag, i, j)
                    break

            # Rule 2: Orient i-j into i->j whenever there is a chain
            # i->k->j.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where k is i->k.
                succs_i = set()
                for k in dag.successors(i):
                    if not dag.has_edge(k, i):
                        succs_i.add(k)
                # Find nodes j where j is k->j.
                preds_j = set()
                for k in dag.predecessors(j):
                    if not dag.has_edge(j, k):
                        preds_j.add(k)
                # Check if there is any node k where i->k->j.
                if len(succs_i & preds_j) > 0:
                    # Make i-j into i->j
                    logging.debug("R2: remove edge (%s, %s)" % (j, i))
                    _orient(dag, i, j)

            # Rule 3: Orient i-j into i->j whenever there are two chains
            # i-k->j and i-l->j such that k and l are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)
                # For all the pairs of nodes in adj_i,
                for (k, l) in combinations(adj_i, 2):
                    # Skip if k and l are adjacent.
                    if _has_any_edge(dag, k, l):
                        continue
                    # Skip if not k->j.
                    if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                        continue
                    # Skip if not l->j.
                    if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                        continue
                    # Make i-j into i->j.
                    logging.debug("R3: remove edge (%s, %s)" % (j, i))
                    _orient(dag, i, j)
                    break

            # Rule 4: Orient i-j into i->j whenever there are two chains
            # i-k->l and k->l->j such that k and j are nonadjacent.
            # TODO: validate me
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)

                # Find nodes l where l -> j
                preds_j = set()
                for l in dag.predecessors(j):
                    if not dag.has_edge(j, l):
                        preds_j.add(l)

                # Find nodes where k -> l
                for k in adj_i:
                    for l in preds_j:
                        if dag.has_edge(k, l) and not dag.has_edge(l, k):
                            _orient(dag, i, j)
                            break

        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()

    return dag
