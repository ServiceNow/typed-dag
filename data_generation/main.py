import argparse
import json
import numpy as np
import os
import warnings

import distribution
from data_generator import DataGenerator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mechanism", type=str, default="linear", help="Type of mechanism use to generate the data(linear, anm, nn)"
    )
    parser.add_argument(
        "--prob-inter", type=float, default=0.3, help="Probability of connection between nodes of different types"
    )
    parser.add_argument(
        "--prob-dropping", type=float, default=0.01, help="Probability of connection between nodes of different types"
    )
    parser.add_argument(
        "--prob-intra", type=float, default=0, help="Probability of connection between nodes of the same types"
    )

    parser.add_argument("--root-distr", type=str, default="gaussian", help="Distribution of root variables")
    parser.add_argument("--noise-distr", type=str, default="gaussian", help="Distribution of noises")
    parser.add_argument("--noise-coeff", type=float, default=0.1, help="Noise coefficient")
    parser.add_argument("--n-datasets", type=int, default=10, help="Total number of datasets to generate")
    parser.add_argument("--n-types", type=int, default=4, help="Number of node types")
    parser.add_argument("--entities-dim", type=int, default=5, help="Dimension of the entities")

    # Training datasets
    parser.add_argument("--n-dags-train", type=int, default=1, help="Number of DAGs to generate train dataset from")
    parser.add_argument("--n-train", type=int, default=1000, help="Number of points")
    parser.add_argument("--n-nodes-train", type=int, default=10, help="Number of nodes for each DAGs")

    # Testing datasets
    parser.add_argument("--n-dags-test", type=int, default=1, help="Number of DAGs to generate dataset from")
    parser.add_argument("--n-test", type=int, default=100, help="Number of points for test dataset")
    parser.add_argument("--n-nodes-test", type=int, default=20, help="Number of nodes for DAGs used in testing")

    # Intervention
    parser.add_argument(
        "--intervention-train", action="store_true", help="if True, generate training data with intervention"
    )
    parser.add_argument(
        "--intervention-test", action="store_true", help="if True, generate testing data with intervention"
    )
    parser.add_argument("--n-interventions-train", type=int, default=0, help="Number of interventional targets")
    parser.add_argument("--n-interventions-test", type=int, default=0, help="Number of interventional targets")
    parser.add_argument(
        "--n-targets-train", type=int, default=1, help="Number of targeted nodes per interventional target"
    )
    parser.add_argument(
        "--n-targets-test", type=int, default=1, help="Number of targeted nodes per interventional target"
    )
    parser.add_argument(
        "--intervention-type", type=str, default="perfect", help="Type of intervention (perfect or imperfect)"
    )
    parser.add_argument(
        "--perfect-interv-distr",
        type=str,
        default="gaussian",
        help="Type of marginal distribution used when a perfect intervention is performed",
    )

    parser.add_argument("--seed", type=int, default=None, help="Seed for RNG")
    parser.add_argument(
        "--suffix", type=str, default="linear", help="Suffix that will be added at the end of the directory name"
    )
    args = parser.parse_args()

    return args


# parse the arguments
args = parse_args()

# create a directory to hold all datasets
base_data_path = f"simulated_data/data_p{args.n_nodes_test}_n{args.n_test}_{args.suffix}"
os.makedirs(base_data_path, exist_ok=True)

# save the arguments in a JSON file
path = os.path.join(base_data_path, "settings.json")
with open(path, "w") as f:
    json.dump(vars(args), f, indent=4)

# set seed
if args.seed is not None:
    np.random.seed(args.seed)

# create models and data
for i_dataset in range(args.n_datasets):
    print(f"Generating data set #{i_dataset + 1}")

    # create a directory per dataset
    ds_path = f"{base_data_path}/data{i_dataset + 1}"
    os.makedirs(ds_path, exist_ok=True)

    type_distr = distribution.Covering_Categorial(args.n_types)
    entity_distr = distribution.GaussianList(args.n_types, args.entities_dim)
    metadata_distr = None

    # Generate meta-training example
    for i_dag in range(args.n_dags_train):
        # TODO: no interv for now
        generator_train = DataGenerator(
            args.n_nodes_train,
            args.n_types,
            args.prob_inter,
            args.prob_dropping,
            args.prob_intra,
            type_distr,
            entity_distr,
            metadata_distr,
            args.mechanism,
            args.root_distr,
            args.noise_coeff,
            args.noise_distr,
            args.intervention_train,
            args.n_interventions_train,
            args.n_targets_train,
            args.intervention_type,
            args.perfect_interv_distr,
        )
        generator_train.sample(args.n_train)
        generator_train.save(ds_path, "train", i_dag + 1)

    # Generate meta-testing example
    for i_dag in range(args.n_dags_test):
        generator_test = DataGenerator(
            args.n_nodes_test,
            args.n_types,
            args.prob_inter,
            args.prob_dropping,
            args.prob_intra,
            type_distr,
            entity_distr,
            metadata_distr,
            args.mechanism,
            args.root_distr,
            args.noise_coeff,
            args.noise_distr,
            args.intervention_test,
            args.n_interventions_test,
            args.n_targets_test,
            args.intervention_type,
            args.perfect_interv_distr,
        )
        generator_test.sample(args.n_test)
        generator_test.save(ds_path, "test", i_dag + 1)
