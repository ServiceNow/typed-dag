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

>> Utility functions for graphs

"""
import logging

from itertools import permutations


def _has_both_edges(dag, i: int, j: int):
    """
    Check if edge i-j is unoriented
    """
    return dag.has_edge(i, j) and dag.has_edge(j, i)


def _has_any_edge(dag, i: int, j: int):
    """
    Check if i and j are connected (irrespective of direction)
    """
    return dag.has_edge(i, j) or dag.has_edge(j, i)


def type_of(dag, node: int):
    """
    Get the type of a node

    """
    return dag.nodes[node]["type"]


def _orient(dag, n1: int, n2: int):
    """
    Orients all edges from type(node1) to type(node2). If types are the same, simply orient the edge between the nodes.

    """
    t1 = type_of(dag, n1)
    t2 = type_of(dag, n2)

    # Case: orient intra-type edges (as if not typed)
    if t1 == t2:
        if not _has_both_edges(dag, n1, n2):
            print(f"Edge {n1}-{n2} is already oriented. Not touching it.")
        else:
            logging.debug(f"... Orienting {n1} (t{t1}) -> {n2} (t{t2}) (intra-type)")
            dag.remove_edge(n2, n1)

    # Case: orient t-edge
    else:
        logging.debug(f"Orienting t-edge: {t1} --> {t2}")
        for _n1, _n2 in permutations(dag.nodes(), 2):
            if type_of(dag, _n1) == t1 and type_of(dag, _n2) == t2 and _has_both_edges(dag, _n1, _n2):
                logging.debug(f"... Orienting {_n1} (t{t1}) -> {_n2} (t{t2})")
                dag.remove_edge(_n2, _n1)
            elif (
                type_of(dag, _n1) == t1
                and type_of(dag, _n2) == t2
                # CPDAG contains at least one edge with t2 -> t1, while it should be t1 -> t2.
                and dag.has_edge(_n2, _n1)
                and not dag.has_edge(_n1, _n2)
            ):
                raise Exception(
                    f"State of inconsistency. CPDAG contains edge {_n2} (t{t2}) -> {_n1} (t{t1}), while the t-edge should be t{t1} -> t{t2}."
                )
