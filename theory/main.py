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
import os
import pandas as pd
import sys

from causaldag import DAG
from itertools import combinations, permutations
from joblib import delayed, Parallel
from os import makedirs
from sampler import sample_t_dag
from time import time

from tmec import tmec_enumeration, typed_meek


def compare_tmec_size(
    folder_path: str,
    factor: str,
    n_type_list: list,
    n_node_list: list,
    probs_inter_list: list,
    probs_intra_list: list,
    n_graph: int = 100,
    n_cpu: int = -1,
    verbose: bool = False,
):
    """
    Compare the size of the MEC and the t-MEC for random t-DAGs.
    :param folder_path: name of the folder where the results will be saved
    :param factor: factor that will vary (type, node, density)
    :param n_type_list: list of number of types considered
    :param n_node_list: list of number of nodes considered
    :param probs_inter_list: list of prob_inter considered
    :param probs_intra_list: list of prob_intra considered
    :param n_graph: number of graphs sampled to compare the MECs
    :param n_cpu: number of cpus to use (-1 = all)
    :param verbose: if True, print messages for each exp completed
    """
    type_no_sources = False  # We don't make the assumption that there are no root types

    results = Parallel(n_jobs=n_cpu)(
        delayed(doit)(n_nodes, n_types, prob_inter, prob_intra, type_no_sources, graph_i, verbose)
        for graph_i in range(n_graph)
        for n_types in n_type_list
        for n_nodes in n_node_list
        for prob_inter in probs_inter_list
        for prob_intra in probs_intra_list
    )

    # Create output path
    makedirs(folder_path, exist_ok=True)
    if factor != "":
        pd.DataFrame(results).to_csv(folder_path + f"/results_{factor}.csv")
    else:
        pd.DataFrame(results).to_csv(folder_path + "/results.csv")


def count_unoriented_tedges(g: np.ndarray, types: list):
    """
    Return the number of t-edges that are not oriented in the graph g.
    :param g: graph with possibly some edges not oriented
    :param types: list of types for every node in g
    """
    # Collect t-edge orientations (some may be unoriented)
    n_types = len(set(types))
    t_edges = np.zeros((n_types, n_types))
    for i, j in permutations(range(g.shape[0]), 2):
        if types[i] == types[j]:
            continue
        if g[i, j] and not g[j, i]:  # For every oriented edge
            assert t_edges[types[i], types[j]] + t_edges[types[j], types[i]] < 2
            t_edges[types[i], types[j]] = 1
            t_edges[types[j], types[i]] = 0

        # Unoriented edge, means unoriented t-edge
        elif g[i, j] and g[j, i]:
            # Assert the t-edge is not already oriented
            assert t_edges[types[i], types[j]] + t_edges[types[j], types[i]] != 1
            t_edges[types[i], types[j]] = 1
            t_edges[types[j], types[i]] = 1

    count = 0
    for i, j in combinations(range(t_edges.shape[0]), 2):
        if t_edges[i, j] and t_edges[j, i]:
            count += 1
    return count


def doit(
    n_nodes: int, n_types: int, prob_inter: float, prob_intra: float, type_no_sources: bool, graph_i: int, verbose: bool
):
    """
    :param n_nodes: Number of nodes
    :param n_types: Number of types
    :prob_inter: Probability of adding an edge between different types
    :prob_intra: Probability of adding an edge between the same type
    :param type_no_sources: If True, all types will have at least one parent type. Otherwise, no constraint.
    :param graph_i: index of the graph
    :param verbose: if True, print messages for each exp completed
    """
    if verbose:
        print(
            f"<start job> nodes: {n_nodes}, types: {n_types}, p_inter: {prob_inter}, p_intra: {prob_intra}, type_no_sources: {type_no_sources}, graph_i: {graph_i}"
        )
    start_time = time()

    dag, type_graph, types = sample_t_dag(
        n_nodes, n_types, prob_inter=prob_inter, prob_intra=prob_intra, type_no_sources=type_no_sources
    )

    # calculate MEC size
    dag = DAG.from_amat(dag)
    mec_size = len(dag.cpdag().all_dags())  # mec_size()
    cpdag = dag.cpdag().to_amat()[0]

    # calculate t-MEC size
    pre_tess_g, g_compatibility = typed_meek(cpdag, types)
    all_dags = tmec_enumeration(pre_tess_g, types, g_compatibility)
    tess_g = (sum(all_dags) > 0).astype(int)
    n_unoriented_tedges = count_unoriented_tedges(tess_g, types)

    # Sanity check: true DAG is in enumerated DAGs
    assert hash(bytes(dag.to_amat()[0])) in [
        hash(bytes(x)) for x in all_dags
    ], "The true DAG is not in the enumeration."

    # Sanity check: all enumerated DAGs are Markov equivalent to the true DAG
    for d in all_dags:
        if not dag.markov_equivalent(DAG.from_amat(d)):
            raise Exception("t-MEC contains DAGs that are not Markov equivalent.")

    tmec_size = len(all_dags)

    assert mec_size >= tmec_size, "T-MEC cannot be larger than MEC (by definition)"

    runtime = time() - start_time
    if verbose:
        print(
            f"<end job> nodes: {n_nodes}, types: {n_types}, p_inter: {prob_inter}, p_intra: {prob_intra}, type_no_sources: {type_no_sources}, graph_i: {graph_i}, MEC: {mec_size}, t-MEC: {tmec_size}, unoriented t-edges: {n_unoriented_tedges}, runtime: {runtime}"
        )

    return dict(
        mec_size=mec_size,
        tmec_size=tmec_size,
        unoriented_tedges=n_unoriented_tedges,
        n_nodes=n_nodes,
        n_types=n_types,
        prob_inter=prob_inter,
        prob_intra=prob_intra,
        graph_i=graph_i,
        runtime=runtime,
    )
