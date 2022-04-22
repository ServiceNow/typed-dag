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

>> Visualize graphs

"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from graphviz import Graph, Digraph


def create_skeleton_viz(graph, coeff, var_names, stat_tests, types=None):
    """ Create a graph from a skeleton in graphviz for visualization """
    colors = ["green", "red"]
    g = Graph("G", filename="skeleton.gv")

    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i, j] == 1:
                if stat_tests[i, j]:  # == 1 if statistically significative
                    g.edge(var_names[i], var_names[j], color=colors[0], label=str(coeff[i, j]))
                else:
                    g.edge(var_names[i], var_names[j], color=colors[0])

    return g


def rgb2hex(rgb: list) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def create_graph_viz(dag: np.ndarray, var_names: list, stat_tests: np.ndarray, types: list, fname="cpdag"):
    """ Create a graph from a CPDAG in graphviz for visualization """
    edge_colors = ["green", "red"]

    if len(np.unique(types)) <= 12:
        cmap = matplotlib.cm.get_cmap("Set3", 12)
        node_colors = cmap(np.linspace(0, 1, 12))
    else:
        node_colors = plt.get_cmap("hsv")(np.linspace(0, 1.0, len(np.unique(types))))
    node_colors = [rgb2hex(c) for c in node_colors]

    g = Digraph("G", filename=f"figures/{fname}", format="png")

    for i in range(dag.shape[0]):
        color = node_colors[types[i]]
        g.node(var_names[i], fillcolor=color, style="filled")

    for i in range(dag.shape[0]):
        for j in range(dag.shape[1]):
            if dag[i, j] == 1:
                if stat_tests[i, j]:  # == 1 if statistically significative
                    g.edge(
                        var_names[i],
                        var_names[j],
                        color=edge_colors[0],
                        label="",
                    )
                else:
                    g.edge(var_names[i], var_names[j], color=edge_colors[1])

    g.render()

    return g


def show_data(data, dag, only_child=True):
    n_nodes = data.shape[1]

    fig, axs = plt.subplots(n_nodes, n_nodes, figsize=(15, 15))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if only_child and dag[i, j] == 1:
                axs[i, j].scatter(data[:, i], data[:, j], s=1)
            elif not only_child:
                axs[i, j].scatter(data[:, i], data[:, j], s=1)
