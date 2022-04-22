"""
This has been copied from DCDI

"""
import networkx as nx

from causaldag import DAG
from cdt.metrics import retrieve_adjacency_matrix

from tmec import to_tessential_graph


def edge_errors(pred, target):
    """
    Counts all types of edge errors (false negatives (fn), false positives
    (fp), reversed edges (rev))

    :parm pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    :param target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns: fn, fp, rev
    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)

    diff = true_labels - predictions

    rev = (((diff + diff.transpose()) == 0) & (diff != 0)).sum() / 2

    undirected_fn = ((diff + diff.transpose()) == 2).sum() / 2
    undirected_fp = ((diff + diff.transpose()) == -2).sum() / 2

    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev
    fn -= undirected_fn
    fp -= undirected_fp

    return fn, fp, rev


def shd(pred, target):
    """
    Calculates the structural hamming distance (SHD)

    :param pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    :param target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns: shd
    """
    return sum(edge_errors(pred, target))


def shd_tess(pred, target, types):
    """
    Calculates the SHD between pred and the t-essential graph of target
    :param pred: np.ndarray, shape=(nodes, nodes)
        The adjacency matrix of the predicted graph (CPDAG).
    :param target: np.ndarray, shape=(nodes, nodes)
        The adjacency matrix of the true DAG
    :param types: np.ndarray, shape=(nodes,)
        The type of each variable

    """
    target = to_tessential_graph(DAG.from_amat(target).cpdag().to_amat()[0], types)[0]
    return int(shd(pred, target))
