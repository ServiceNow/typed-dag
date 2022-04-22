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
import argparse
from itertools import combinations, permutations
import json
import logging
import networkx as nx
import numpy as np
import os
import pandas as pd
import pcalg

from causaldag import DAG, PDAG
from copy import deepcopy as copy
from functools import partial
from pathlib import Path

from ci import ci_test_dis, ci_test_fcit, ci_test_partialcorr
from pc import pc_meek_rules
from tmec import to_tessential_graph
from typed_pc import orient_forks_naive, orient_forks_majority_top1
from visualization import create_graph_viz


BASE_DATA_PATH = Path("/dcdr/data/semi_simulated/")

# TODO: describe each function


def load_dataset(path: str):
    """Load synthetic datasets that are in multiple files, containing:
    the dag, types, variables' names, types' names, and known t-edges
    """
    path = Path(path)
    types = np.load(path / "test_classes_1.npy")
    dag = np.load(path / "test_dag_1.npy")
    if os.path.exists(path / "variable_names.npy"):
        variable_names = np.load(path / "variable_names.npy", allow_pickle=True)
    else:
        variable_names = [f"x{i}" for i in range(dag.shape[0])]
    if os.path.exists(path / "type_names.npy"):
        type_names = np.load(path / "type_names.npy", allow_pickle=True)
    else:
        type_names = [f"t{i}" for i in range(len(np.unique(types)))]
    if os.path.exists(path / "known_tedges.npy"):
        known_tedges = np.load(path / "known_tedges.npy")
    else:
        known_tedges = []
    data = np.load(path / "test_data_1.npy")

    # TODO: Add known t-edges and the rest of the stuff that could exist and file existance checks
    return data, dag, types, variable_names, type_names, known_tedges


def load_semi_simulated_dataset(
    name: str = "sachs", type: str = "exact", n_per_type: int = 2, n_samples: int = None, dataset_id: int = 1
):
    """Load pseudoreal datasets that are in multiple files, containing:
    the dag, types, variables' names, types' names, and known t-edges
    """
    data, dag, types, variable_names, type_names, known_tedges = load_dataset(
        BASE_DATA_PATH / f"{name}_{type}_{n_per_type}_n10000/data{dataset_id}"
    )
    if n_samples is None:
        n_samples = data.shape[0]
    return data[:n_samples], dag, types, variable_names, type_names, known_tedges


###############
# PC variants
###############


def pc(skeleton, separating_sets):
    out = pcalg.estimate_cpdag(skel_graph=skeleton, sep_set=separating_sets)
    return nx.adjacency_matrix(out).todense()


def tpc_naive(skeleton, separating_sets):
    out = pc_meek_rules(orient_forks_naive(skeleton=skeleton, sep_sets=separating_sets))
    return nx.adjacency_matrix(out).todense()


def tpc_majority_top1(skeleton, separating_sets):
    out = pc_meek_rules(orient_forks_majority_top1(skeleton=skeleton, sep_sets=separating_sets))
    return nx.adjacency_matrix(out).todense()


def get_metrics(g_true, g_est, types):
    from metrics import edge_errors, shd, shd_tess

    metrics = {}
    metrics.update(dict(zip(["fn", "fp", "rev"], edge_errors(g_est, g_true))))
    metrics["shd"] = shd(g_est, g_true)
    metrics["shd_tess"] = shd_tess(g_est, g_true, types)
    return metrics


def sanity_type_consistency(cpdag: np.ndarray, types: np.ndarray) -> bool:
    # Collect t-edge orientations (some may be unoriented)
    n_types = len(set(types))
    t_edges = np.zeros((n_types, n_types))
    for i, j in permutations(range(cpdag.shape[0]), 2):
        if cpdag[i, j] and not cpdag[j, i]:  # For every oriented edge
            t_edges[types[i], types[j]] = 1

    # Check if some oriented edges caused t-edges to be oriented in both directions
    for i, j in permutations(range(n_types), 2):
        if t_edges[i, j] and t_edges[j, i]:
            return False

    return True


def sanity_all_two_type_forks_oriented(cpdag: np.ndarray, types: np.ndarray) -> bool:
    for i, j, k in permutations(range(cpdag.shape[0]), 3):
        # Criterion 1: i,j are of the same type, k is not of the same type
        if types[i] == types[j] and types[i] != types[k]:

            # Criterion 2: i - k - j with i -/-j
            if (
                (cpdag[i, k] == 1 or cpdag[k, i] == 1)
                and (cpdag[j, k] == 1 or cpdag[k, j] == 1)
                and (cpdag[i, j] == cpdag[j, i] == 0)
            ):
                if (
                    cpdag[i, k] + cpdag[k, i] > 1  # Unoriented i-k
                    or cpdag[j, k] + cpdag[k, j] > 1  # Unoriented j-k
                    or cpdag[i, k] != cpdag[j, k]  # Inconsistent orientation
                    or cpdag[k, i] != cpdag[k, j]  # Inconsistent orientation
                ):
                    return False
    return True


def sanity_check(cpdag: np.ndarray, skeleton: np.ndarray, types: np.ndarray) -> bool:
    checks = {}
    checks["type_consistency"] = sanity_type_consistency(cpdag, types)
    checks["two_type_forks"] = sanity_all_two_type_forks_oriented(cpdag, types)

    print()
    print("Sanity checking solution:")
    print("---------------------------")
    print("... Type consistency:", checks["type_consistency"])
    print("... Two-type forks oriented:", checks["two_type_forks"])

    # Hard tests (must pass)
    assert (
        nx.adjacency_matrix(nx.from_numpy_array(cpdag, create_using=nx.DiGraph).to_undirected()).sum() == skeleton.sum()
    ), "The orientation broke some connections."
    print("... No broken connections: True")

    print()

    return {k: int(v) for k, v in checks.items()}


def process_output(method, cpdag, skeleton, variable_names, true_dag, types):
    """
    Save to disk and calculate metrics

    """
    create_graph_viz(cpdag, variable_names, true_dag, types, method)
    np.save(f"{str(method)}.npy", cpdag)
    metrics = get_metrics(true_dag, cpdag, types)
    metrics.update(sanity_check(cpdag, skeleton, types))
    return {method: metrics}


def save_ground_truth(true_dag, types, variable_names):
    true_cpdag = DAG.from_amat(true_dag).cpdag().to_amat()[0]
    true_tessG = to_tessential_graph(true_cpdag, types)[0]
    create_graph_viz(true_dag, variable_names, true_dag, types, "gt")
    create_graph_viz(true_cpdag, variable_names, true_cpdag, types, "gt-cpdag")
    create_graph_viz(true_tessG, variable_names, true_tessG, types, "gt-tess")

    # Save these just for convenience
    np.save("true_dag.npy", true_dag)
    np.save("true_cpdag.npy", true_cpdag)
    np.save("true_tess.npy", true_tessG)
