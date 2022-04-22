import copy
import numpy as np
import torch as th


class Linear:
    """
    Linear mechanism, where Effect = coeff * Cause + Noise.
    :param n_causes: number of causes/parents of the given variable
    :param noise_coeff: coefficients used to mix noise
    :param noise_distr: distribution used as noise [gaussian]
    :param param: parameters of the mechanism
    :param low_val_obs: lower value to sample param
    :param high_val_obs: upper value to sample param
    :param low_val: lower value to sample param with interv
    :param high_val: upper value to sample param with interv
    """

    def __init__(
        self,
        n_causes: int,
        noise_distr: str,
        noise_coeff: float = 0.4,
        param: list = None,
        low_val_obs: float = 0.25,
        high_val_obs: float = 1,
        low_val: float = 2,
        high_val: float = 5,
    ):
        super().__init__()
        self.n_causes = n_causes
        self.noise_distr = noise_distr
        self.noise_coeff = noise_coeff

        self.low_val_obs = low_val_obs
        self.high_val_obs = high_val_obs
        self.low_val = low_val
        self.high_val = high_val

        self._init_mechanism(param)

    def _init_mechanism(self, obs_param=None):
        """Init the mechanism by sampling parameters"""
        self.param = []
        if obs_param is None:
            # observational case
            for i in range(self.n_causes):
                param = np.random.uniform(self.low_val_obs, self.high_val_obs)
                # sample the sign of the coeff
                if np.random.randint(2) == 0:
                    param *= -1

                self.param.append(param)
        else:
            # interventional cases
            for i in range(self.n_causes):
                change = np.random.uniform(self.low_val, self.high_val)
                if obs_param[i] > 0:
                    param = obs_param[i] + change
                else:
                    param = obs_param[i] - change

                # sample the sign of the coeff
                if np.random.randint(2) == 0:
                    param *= -1

                self.param.append(param)

    def __call__(self, n, causes):
        """Apply the mechanism to the causes"""
        effect = np.zeros(n)
        noise = self.noise_coeff * self.noise_distr(n)

        # Compute each cause's contribution
        for parent in range(causes.shape[1]):
            effect = effect + self.param[parent] * causes[:, parent]
        effect = effect + noise

        return effect


class ANM:
    """
    Additive noise model, where Effect = f(Cause) + Noise
    and f is a neural network

    :param n_causes: number of causes/parents of the given variable
    :param noise_coeff: coefficients used to mix noise
    :param noise_distr: distribution used as noise [gaussian]
    :param param: neural network parameters
    :param n_hidden: number of hidden units in the neural network
    """

    def __init__(
        self, n_causes: int, noise_distr: str, noise_coeff: float = 0.4, param: th.Tensor = None, n_hidden: int = 10
    ):
        super().__init__()
        self.n_causes = n_causes
        self.noise_coeff = noise_coeff
        self.noise_distr = noise_distr
        self.n_hidden = n_hidden
        self._init_mechanism(param)

    def weight_init(self, model):
        if isinstance(model, th.nn.modules.Linear):
            th.nn.init.normal_(model.weight.data, mean=0.0, std=1)

    def apply_nn(self, x):
        data = x.astype("float32")
        data = th.from_numpy(data)

        return np.reshape(self.param(data).data, (x.shape[0],))

    def _init_mechanism(self, obs_param=None):
        if obs_param is None:
            layers = []

            layers.append(th.nn.modules.Linear(self.n_causes, self.n_hidden))
            layers.append(th.nn.PReLU())
            layers.append(th.nn.modules.Linear(self.n_hidden, 1))

            layers = th.nn.Sequential(*layers)
            layers.apply(self.weight_init)
            self.param = layers
        else:
            self.param = copy.deepcopy(obs_param)
            for i, layer in enumerate(obs_param):
                if isinstance(layer, th.nn.modules.Linear) and i > 0:
                    with th.no_grad():
                        layer.weight += th.empty_like(layer.weight).normal_(mean=0, std=1)

    def __call__(self, n, causes):
        """Apply the mechanism to the causes"""
        effect = np.zeros(n)
        noise = self.noise_coeff * self.noise_distr(n)

        effect = self.apply_nn(causes).numpy()
        effect = effect + noise

        return effect


class NN:
    """
    General nonlinear mechanism, where Effect = f(Cause, Noise)
    and f is a neural network

    :param n_causes: number of causes/parents of the given variable
    :param noise_coeff: coefficients used to mix noise
    :param noise_distr: distribution used as noise [gaussian]
    :param param: neural network parameters
    :param n_hidden: number of hidden units in the neural network
    """

    def __init__(
        self, n_causes: int, noise_distr: str, noise_coeff: float = 0.4, param: th.Tensor = None, n_hidden: int = 20
    ):
        """Init the mechanism."""
        super().__init__()
        self.n_causes = n_causes
        self.noise_coeff = noise_coeff
        self.noise_distr = noise_distr
        self.n_hidden = n_hidden
        self._init_mechanism(param)

    def weight_init(self, model):
        if isinstance(model, th.nn.modules.Linear):
            th.nn.init.normal_(model.weight.data, mean=0.0, std=1)

    def apply_nn(self, x):
        data = x.astype("float32")
        data = th.from_numpy(data)

        return np.reshape(self.param(data).data, (x.shape[0],))

    def _init_mechanism(self, obs_param=None):
        if obs_param is None:
            layers = []

            layers.append(th.nn.modules.Linear(self.n_causes + 1, self.n_hidden))
            layers.append(th.nn.Tanh())
            layers.append(th.nn.modules.Linear(self.n_hidden, 1))

            layers = th.nn.Sequential(*layers)
            layers.apply(self.weight_init)
            self.param = layers
        else:
            self.param = copy.deepcopy(obs_param)
            for i, layer in enumerate(obs_param):
                if isinstance(layer, th.nn.modules.Linear) and i > 0:
                    with th.no_grad():
                        layer.weight += th.empty_like(layer.weight).normal_(mean=0, std=1)

    def __call__(self, n, causes):
        """Run the mechanism."""
        effect = np.zeros(n)
        noise = self.noise_coeff * self.noise_distr(n)

        mix = np.hstack((causes, noise[:, np.newaxis]))
        effect = self.apply_nn(mix).numpy()

        return effect


class Mechanism:
    """
    The base class for causal mechanisms of a variable (observational and under
    intervention)

    :param n_k: total number of interventional target/context
    :param interv_family: list of interventional targets
    :param interv_type: type of intervention [perfect, imperfect]
    :param n_causes: number of causes/parents of the given variable
    :param noise_coeff: coefficients used to mix noise
    :param noise_distr: distribution used as noise [gaussian]
    :param mech_type: type of mechanisms [linear, anm, nn]
    :param root_distr: type of distribution for root node [gaussian]
    :param interv_distr_type: type of distr when applying a perfect interv
    """

    mech_dict = {"linear": Linear, "anm": ANM, "nn": NN}

    def __init__(
        self,
        node,
        n_k: int,
        interv_family: list,
        interv_type: str,
        n_causes: int,
        noise_coeff: float,
        noise_distr: str,
        mech_type: str,
        root_distr_type: str,
        interv_distr_type: str,
    ):
        super().__init__()

        if mech_type not in self.mech_dict:
            raise ValueError(f"{mech_type} is not a valid mechanism")

        self.mechanisms = {}
        self.node = node
        self.n_k = n_k
        self.interv_family = interv_family
        self.n_causes = n_causes
        self.noise_coeff = [noise_coeff] * n_k
        self.interv_type = interv_type

        self.noise_distr = distr_init(noise_distr, "noise")
        self.interv_distr = distr_init(interv_distr_type, "interv")

        if n_causes > 0:
            self.mechanism = self.mech_dict[mech_type]
            self._init_mechanisms()
        else:
            self.root_distr = distr_init(root_distr_type, "root")
            self._init_root_mechanisms()

    def __call__(self, n: int, k: int, causes: np.ndarray = None) -> np.ndarray:
        if k not in self.mechanisms:
            raise ValueError("Mechanism not initiated!")
        return self.mechanisms[k](n, causes)

    def _init_mechanisms(self):
        for k in range(self.n_k):
            if k == 0:
                self.mechanisms[0] = self.mechanism(self.n_causes, self.noise_distr, self.noise_coeff[0])
            elif self.node in self.interv_family[k]:
                if self.interv_type == "imperfect":
                    param = self.mechanisms[0].param
                    self.mechanisms[k] = self.mechanism(self.n_causes, self.noise_distr, self.noise_coeff[k], param)
                elif self.interv_type == "perfect":
                    self.mechanisms[k] = self.interv_distr

    def _init_root_mechanisms(self):
        for k in range(self.n_k):
            if k == 0:
                self.mechanisms[k] = self.root_distr
            elif self.node in self.interv_family[k]:
                self.mechanisms[k] = self.interv_distr


def distr_init(name: str, distr_type: str) -> object:
    """
    Initialize distributions with predefined parameters
    :param name: name of the distribution [gaussian, uniform]
    :param distr_type: type of the distr in the SCM [noise, root, interv]
    """
    assert name in ["gaussian", "uniform"], f"{name} is not a valid distribution"
    assert distr_type in ["noise", "root", "interv"], f"{distr_type} is not a valid distribution type"

    if distr_type == "root" or distr_type == "noise":
        if name == "gaussian":
            distr = Gaussian(0, 1, 1, 2)
        elif name == "uniform":
            distr = Uniform()
    elif distr_type == "interv":
        if name == "gaussian":
            distr = Gaussian(2, 1)
        elif name == "uniform":
            distr = Uniform()

    return distr


class Gaussian:
    """
    Gaussian distribution. Used for root distr, noise and perfect interventions

    :param mu: mu parameter
    :param sigma: sigma parameter
    :param sigma_min: if not None, used as the lower bound value to sample sigma
    :param sigma_max: if not None, used as the upper bound value to sample sigma
    """

    def __init__(self, mu=0, sigma=1, sigma_min=None, sigma_max=None):
        self.mu = mu
        if sigma_min is None and sigma_max is None:
            self.sigma = sigma
        else:
            self.sigma = np.random.uniform(sigma_min, sigma_max)

    def __call__(self, n: int, causes=None):
        return np.random.normal(self.mu, self.sigma, size=n)


class Uniform:
    """
    Uniform distribution. Used for root distr, noise and perfect interventions
    :param _min: lower bound of the uniform
    :param _max: upper bound of the uniform
    """

    def __init__(self, _min=-1, _max=1):
        self._min, self._max = _min, _max

    def __call__(self, n: int, causes=None):
        return np.random.uniform(self._min, self._max, size=n)
