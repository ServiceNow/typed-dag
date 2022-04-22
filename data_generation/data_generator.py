import os
import warnings
import numpy as np
import networkx as nx
import mechanisms as mech
import pickle
import json
from typing import Tuple
from causaldag import DAG
from tmec import typed_meek, tmec_enumeration


def is_acyclic(adjacency: np.ndarray) -> bool:
    """Check if adjacency matrix is
    acyclic return True if so"""
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0:
            return False
    return True


def is_consistent_tdag(adjacency: np.ndarray, types: list) -> bool:
    """Check if the adjacency matrix with the given types
    form a consistent t-DAG
    :param adjacency: adjacency matrix
    :param types: type for each node
    returns: True, if consistent
    """

    # Check if it is at least a DAG
    if not is_acyclic(adjacency):
        print("Graph is cyclic")
        return False

    # For if t-edges are consistents
    n_nodes = len(types)
    t_edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if types[i] != types[j] and adjacency[i, j]:
                if (types[i], types[j]) not in t_edges:
                    t_edges.append((types[i], types[j]))
                if (types[j], types[i]) in t_edges:
                    print("t-edges are not consistent")
                    return False

    return True


def count_undirected_edges(adjacency: np.ndarray) -> int:

    undirected_edges = 0
    for i in range(adjacency.shape[0]):
        for j in range(i, adjacency.shape[0]):
            if adjacency[i, j] and adjacency[j, i]:
                undirected_edges += 1

    return undirected_edges


def get_mec_size(adjacency: np.ndarray) -> Tuple[int, int]:
    """ Return size of the MEC and the number of undirected edges"""
    dag = DAG.from_amat(adjacency)
    mec_size = dag.cpdag().mec_size()
    cpdag = dag.cpdag().to_amat()[0]

    return mec_size, count_undirected_edges(cpdag)


def get_tmec_size(adjacency: np.ndarray, types: list) -> Tuple[int, int]:
    """ Return size of the tMEC and the number of undirected edges"""
    dag = DAG.from_amat(adjacency)
    cpdag = dag.cpdag().to_amat()[0]
    t_ess, type_g = typed_meek(cpdag, types)

    all_dags = tmec_enumeration(t_ess, types, type_g)
    tmec_size = len(all_dags)

    return tmec_size, count_undirected_edges(t_ess)


class DataGenerator:
    """
    Generate a Structural Causal Model(SCM) where nodes belongs to types.
    Metadata for each nodes is returned and data sampled from the SCM.

    :param n_nodes: number of nodes in the graph
    :param n_types: number of possible types
    :param prob_inter: Probability of having an edge between nodes of different types
    :param prob_dropping: Probability setting to 0 prob_inter for some pair of types
    :param prob_intra: Probability of having an edge between nodes of the same type
    :param type_distr: distribution of the types
    :param entity_distr: distribution of the entities given its type
    :param metadata_distr: distribution of the metadata given the entity
    :param mech_type: type of mechanisms [linear, anm, nn]
    :param root_distr: type of distribution for root node [gaussian]
    :param noise_coeff: coefficients used to mix noise
    :param noise_distr: distribution used as noise [gaussian]
    :param intervention: if True, generate testing data with intervention
    :param n_intervention: Number of interventional targets
    :param n_targets: Number of targeted nodes per interventional target
    :param intervention_type: Type of intervention (perfect or imperfect)
    :param perfect_interv_distr: Type of distribution used when a perfect intervention is performed
    """

    def __init__(
        self,
        n_nodes: int,
        n_types: int,
        prob_inter: float,
        prob_dropping: float,
        prob_intra: float,
        type_distr: object,
        entity_distr: object,
        metadata_distr: object,
        mech_type: str,
        root_distr: str,
        noise_coeff: float,
        noise_distr: str,
        intervention: bool,
        n_interventions: int,
        n_targets: int,
        intervention_type: str,
        perfect_interv_distr: str,
    ):

        self.n_nodes = n_nodes
        self.n_types = n_types
        self.type_distr = type_distr
        self.prob_inter = prob_inter
        self.prob_dropping = prob_dropping
        self.prob_intra = prob_intra
        self.entity_distr = entity_distr
        self.metadata_distr = metadata_distr
        self.mechanisms = mech_type
        self.root_distr = root_distr
        self.noise_coeff = noise_coeff
        self.noise_distr = noise_distr
        self.intervention = intervention
        self.n_interventions = n_interventions
        self.n_targets = n_targets
        self.intervention_type = intervention_type
        self.perfect_interv_distr = perfect_interv_distr

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample the complete generative model:
        sample node's types, attributes, graph and mechanims

        :param n: total sample size
        :returns: dag, metadata and data
        """
        # Sample node's types, entities and attributes
        self.types = self.type_distr(self.n_nodes)
        self.entities = self.entity_distr(self.types)
        self.n = n

        if self.metadata_distr is not None:
            self.metadata = self.metadata_distr(self.entities)
        else:
            self.metadata = None

        # Sample consistent t-DAG
        self.dag, self.order, self.type_matrix = self.sample_tdag(
            self.n_nodes, self.types, self.prob_inter, self.prob_dropping, self.prob_intra
        )

        self.mec_size, self.mec_undirected_edges = get_mec_size(self.dag)
        print(f"mec_size:{self.mec_size}, {self.mec_undirected_edges}")
        self.tmec_size, self.tmec_undirected_edges = get_tmec_size(self.dag, self.types)
        print(f"tmec_size:{self.tmec_size}, {self.tmec_undirected_edges}")
        # self.undirected_edges_mec = undirected_edges_mec(self.dag)
        # self.undirected_edges_tmec = undirected_edges_tmec(self.dag, self.types)

        # Sample SCM (mechanisms)
        self.scm = SCM(
            self.dag,
            self.n_nodes,
            self.order,
            self.mechanisms,
            self.root_distr,
            self.noise_coeff,
            self.noise_distr,
            self.intervention,
            self.n_interventions,
            self.n_targets,
            self.intervention_type,
            self.perfect_interv_distr,
        )

        # Generate data
        div = self.n_interventions + 1
        n_per_interv = [n // div + (1 if x < n % div else 0) for x in range(div)]
        cumul_n = np.cumsum([0] + n_per_interv)

        self.data = np.zeros((n, self.n_nodes))
        self.data_interv = np.zeros((n, self.n_nodes))
        self.data_regime = np.zeros((n))

        for i, interv_target in enumerate(self.scm.interv_family):
            mask = np.ones((self.n_nodes))
            a = cumul_n[i]
            b = cumul_n[i + 1]

            self.data[a:b] = self.scm.sample_data(i, n_per_interv[i])
            for target in interv_target:
                mask[target] = 0
            self.data_interv[a:b] = mask
            self.data_regime[a:b] = i

        return self.dag, self.metadata, self.data

    def sample_tdag(
        self, n_nodes: int, types: list, prob_inter: float, prob_dropping: float, prob_intra: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a t-DAG of 'n_nodes' nodes
        :param n_nodes: number of nodes
        :param types: types of each node
        :param prob_inter: Probability of having an edge between nodes of different types
        :param prob_dropping: Probability setting to 0 prob_inter for some pair of types
        :param prob_intra: Probability of having an edge between nodes of the same type
        :returns: 1) adjacency matrix of the sampled t-DAG
                  2) order of the nodes
                  3) type matrix

        Note: the sampling make sure at least one representative
        of each type is present in the t-DAG
        """
        n_nodes = len(types)
        n_types = len(np.unique(np.array(types)))

        # shuffle order of nodes
        order = np.arange(n_nodes)
        np.random.shuffle(order)

        # sample matrix of type interaction
        type_matrix = np.zeros((n_types, n_types))
        for i in range(n_types):
            for j in range(i, n_types):
                if i != j:
                    if np.random.rand() > prob_dropping:
                        if np.random.rand() < 0.5:
                            type_matrix[i, j] = prob_inter
                        else:
                            type_matrix[j, i] = prob_inter
                else:
                    type_matrix[i, j] = prob_intra

        # sample t-DAG
        dag = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i):
                if np.random.rand() < type_matrix[types[order[j]], types[order[i]]]:
                    dag[order[j], order[i]] = 1

        if not is_consistent_tdag(dag, types):
            raise ValueError("Could not sample a consistent tDAG")

        return dag, order, type_matrix

    def save(self, path: str, data_type: str, idx: int):
        """
        Save all the sampled objects as numpy files
        and the DataGenerator as a pickle object

        :param path: path to the folder where the files will be saved
        :param data_type: type of data set (train or test)
        :param idx: index of the dag, used for the files names
        """
        np.save(os.path.join(path, f"{data_type}_classes_{idx}.npy"), self.types)
        np.save(os.path.join(path, f"{data_type}_entities_{idx}.npy"), self.entities)
        np.save(os.path.join(path, f"{data_type}_eta_{idx}.npy"), self.type_matrix)
        np.save(os.path.join(path, f"{data_type}_metadata_{idx}.npy"), self.metadata)
        np.save(os.path.join(path, f"{data_type}_dag_{idx}.npy"), self.dag)
        np.save(os.path.join(path, f"{data_type}_data_{idx}.npy"), self.data)

        # save the size of the MEC and tMEC
        ec_size = {"mec size": self.mec_size, "tmec size": self.tmec_size, "number edges": np.sum(self.dag)}
        with open(os.path.join(path, f"{data_type}_ecsize_{idx}.json"), "w") as file:
            json.dump(ec_size, file, indent=4)

        # save interventional targets and regimes
        if self.intervention:
            np.save(os.path.join(path, f"{data_type}_interventions_{idx}.npy"), self.data_interv)
            np.save(os.path.join(path, f"{data_type}_regimes_{idx}.npy"), self.data_regime)

        # with open(os.path.join(path, f"{data_type}_datagenerator_{idx}.pkl"), "wb") as f:
        #     pickle.dump(self, f)


class SCM:
    """
    Structural Causal Model, includes the dag with its associated mechanisms
    :param dag: adjacency matrix of the DAG
    :param n_nodes: number of nodes
    :param causal_order: causal order of the nodes in the DAG
    :param mech_type: type of mechanisms [linear, anm, nn]
    :param root_distr: type of distribution for root node [gaussian]
    :param noise_coeff: coefficients used to mix noise
    :param noise_distr: distribution used as noise [gaussian]
    :param intervention: if True, generate testing data with intervention
    :param n_intervention: Number of interventional targets
    :param n_targets: Number of targeted nodes per interventional target
    :param intervention_type: Type of intervention (perfect or imperfect)
    :param perfect_interv_distr: Type of distribution used when a perfect intervention is performed
    """

    def __init__(
        self,
        dag: np.ndarray,
        n_nodes: int,
        causal_order: np.ndarray,
        mech_type: str,
        root_distr: str,
        noise_coeff: str,
        noise_distr: str,
        intervention: bool,
        n_interventions: int,
        n_targets: int,
        intervention_type: str,
        perfect_interv_distr: str,
    ):

        self.dag = dag
        self.n_nodes = n_nodes
        self.causal_order = causal_order
        self.mech_type = mech_type
        self.root_distr = root_distr
        self.noise_coeff = noise_coeff
        self.noise_distr = noise_distr
        self.intervention = intervention
        self.n_interventions = n_interventions
        self.n_targets = n_targets
        self.intervention_type = intervention_type
        self.perfect_interv_distr = perfect_interv_distr

        if intervention:
            self.interv_family = self._sample_interventions()
        else:
            self.interv_family = [[]]
        self.mech_dict = self._sample_mechanisms(self.interv_family)

        # TODO: check if ok, if intervention == False

    def _sample_interventions(self) -> list:
        """
        Sample interventional targets, for now uniformly
        :returns: interv_family, a list of list containing the interventional targets
        """
        if self.n_interventions <= self.n_nodes:
            interv_family = np.random.choice(np.arange(self.n_nodes), self.n_interventions, replace=False)
        else:
            interv_family = np.random.choice(np.arange(self.n_nodes), self.n_interventions, replace=True)
        interv_family = [[i] for i in interv_family]

        # add observational setting
        interv_family.insert(0, [])

        return interv_family

    def _sample_mechanisms(self, interv_family: list) -> dict:
        """
        Sample mechanisms for each node
        :param interv_family: list of interventional targets
        :returns: mech_dict containing the sample mechanisms
        """
        mech_dict = {}

        for i, node in enumerate(self.causal_order):
            parents = np.where(self.dag[:, node] == 1)[0]
            n_interventions = len(interv_family)

            mech_dict[node] = mech.Mechanism(
                node=node,
                n_k=n_interventions,
                interv_family=interv_family,
                interv_type="perfect",
                n_causes=len(parents),
                noise_coeff=self.noise_coeff,
                noise_distr=self.noise_distr,
                mech_type=self.mech_type,
                root_distr_type=self.root_distr,
                interv_distr_type="gaussian",
            )

        return mech_dict

    def sample_data(self, kth: int, n: int) -> np.ndarray:
        """
        Sample n data points from the SCM
        :param kth: index of the interventional setting
        :param n: number of data points
        :returns: data
        """
        data = np.zeros((n, self.n_nodes))

        for node in self.causal_order:
            parents = np.where(self.dag[:, node] == 1)[0]

            # if an intervention is performed on this node, select
            # the interventional mechanism
            if kth in self.mech_dict[node].mechanisms:
                k = kth
            else:
                k = 0

            if len(parents) == 0:
                # Node without parent
                data[:, node] = self.mech_dict[node](n, k)
            else:
                data[:, node] = self.mech_dict[node](n, k, data[:, parents])

        return data
