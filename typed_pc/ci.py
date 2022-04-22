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

>> Conditional independence tests

"""
import numpy as np

from fcit import fcit
from gsq.ci_tests import ci_test_bin, ci_test_dis as _ci_test_dis
import pandas as pd
from pingouin import partial_corr


def ci_test_dis(data: np.array, x: int, y: int, sep: list, pval_accumulator: dict, **kwargs) -> float:
    """
    :param data: np.array
    :param x: index of the variable x in data
    :param y: index of the variable y in data
    :param sep: list of indices that are covariates
    :param pval_accumulator: a dictionnary to store the outcome of all CI tests (p-values)
    :returns: p-value
    """
    p_value = _ci_test_dis(data, x, y, sep, **kwargs)
    if len(sep) > 0:
        print(x, y, sep)
    pval_accumulator[(x, y, *sep)] = p_value
    assert not np.isnan(p_value), f"Got a NaN p-value. pv={p_value}"
    return p_value


def ci_test_partialcorr(data: np.array, x: int, y: int, sep: list, pval_accumulator: dict, **kwargs) -> float:
    """
    Wrap the partial correlation test from the pingouin
    module to be in the pcalg format
    :param data: np.array
    :param x: index of the variable x in data
    :param y: index of the variable y in data
    :param sep: list of indices that are covariates
    :param pval_accumulator: a dictionnary to store the outcome of all CI tests (p-values)
    :returns: p-value of the partial correlation
    """
    df = pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    x_name = str(x)
    y_name = str(y)
    covar_list = [str(i) for i in sep]
    result = partial_corr(data=df, x=x_name, y=y_name, covar=covar_list)
    pval_accumulator[(x, y, *sep)] = result["p-val"][0]
    assert not np.isnan(result["p-val"][0]), f"Got a NaN p-value. pv={result['p-val'][0]}"
    return result["p-val"][0]


def ci_test_fcit(data: np.ndarray, x: int, y: int, sep: list, pval_accumulator: dict, **kwargs) -> float:
    """
    Fast conditional independence test based on decision trees (Chalupka et al., 2018)
    :param data: np.array
    :param x: index of the variable x in data
    :param y: index of the variable y in data
    :param sep: list of indices that are covariates
    :param pval_accumulator: a dictionnary to store the outcome of all CI tests (p-values)
    :returns: p-value
    """
    p_value = fcit.test(x=data[:, [x]], y=data[:, [y]], z=data[:, list(sep)] if len(sep) > 0 else None, verbose=False)
    pval_accumulator[(x, y, *sep)] = p_value
    # If the ratio used in the fcit is exactly 1, the t-test returns nan. This special case means that there is no
    # difference in the MSE if we add X to the conditioning set to predict Y. This means we should interpret that
    # as independence.
    p_value = 1 if np.isnan(p_value) else p_value
    assert not np.isnan(p_value), f"Got a NaN p-value. pv={p_value}"
    return p_value
