from tmec import tmec_enumeration, typed_meek
from sampler import sample_t_dag
from causaldag import DAG
import numpy as np
from utils import boxplot


def compare_tmec_size(factor: str, n_graph: int = 100):
    """
    Compare the size of the MEC and the t-MEC
    by plotting the results.
    :param factor: factor that will vary (type, node, density)
    :param n_graph: number of graphs sampled to compare the MECs
    """

    # default values
    n_type_list = [10]
    n_node_list = [50]
    probs_list = [0.2]

    # values specific to each factor
    if factor == "type":
        n_type_list = [2, 5, 10, 15, 20, 30, 40, 50]
        x = n_type_list
        x_label = "Number of types"
        label = "types"
    elif factor == "vertice":
        n_node_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        x = n_node_list
        x_label = "Number of vertices"
        label = "vertices"
    elif factor == "density":
        probs_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        x = probs_list
        x_label = "Density"
        label = "density"
    else:
        raise ValueError("This factor is not implemented")

    len_factor = len(x)

    mec_size = np.zeros((n_graph, len_factor))
    tmec_size = np.zeros((n_graph, len_factor))
    i = 0

    for prob in probs_list:
        for n_node in n_node_list:
            for n_type in n_type_list:
                for graph in range(n_graph):

                    print(f"nodes: {n_node}, types: {n_type}, p: {prob}")
                    dag, class_dag, types = sample_t_dag(n_node, n_type, prob=prob)

                    # calculate MEC size
                    dag = DAG.from_amat(dag)
                    mec_size[graph, i] = dag.cpdag().mec_size()
                    cpdag = dag.cpdag().to_amat()[0]

                    # calculate t-MEC size
                    t_essential_graph, g_compatibility = typed_meek(cpdag, types)
                    print("Start t-essential graph enumeration")
                    all_dags = tmec_enumeration(t_essential_graph, types, g_compatibility)
                    tmec_size[graph, i] = len(all_dags)
                i += 1

    # Plots the boxplot figure and save it
    boxplot(x, mec_size.T, tmec_size.T, x_label, label)

    # Save the raw results
    np.save(f"MEC_size_by_{label}.npy", mec_size)
    np.save(f"tMEC_size_by_{label}.npy", tmec_size)


def main():
    compare_tmec_size("vertice")
    compare_tmec_size("type")
    compare_tmec_size("density")


main()
