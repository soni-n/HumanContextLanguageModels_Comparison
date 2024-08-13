import numpy as np
import pandas as pd
import argparse
import os
import glob
from scipy.stats import combine_pvalues
from sklearn.metrics import f1_score
from itertools import combinations, product
from math import factorial


def permutation_test(x, y, func='x_mean != y_mean', method='exact',
                     num_rounds=1000, seed=None, paired=False):
    """
    Nonparametric permutation test (pulled from mlxtend library)
    Parameters
    -------------
    x : list or numpy array with shape (n_datapoints,)
        A list or 1D numpy array of the first sample
        (e.g., the treatment group).
    y : list or numpy array with shape (n_datapoints,)
        A list or 1D numpy array of the second sample
        (e.g., the control group).
    func : custom function or str (default: 'x_mean != y_mean')
        function to compute the statistic for the permutation test.
        - If 'x_mean != y_mean', uses
          `func=lambda x, y: np.abs(np.mean(x) - np.mean(y)))`
           for a two-sided test.
        - If 'x_mean > y_mean', uses
          `func=lambda x, y: np.mean(x) - np.mean(y))`
           for a one-sided test.
        - If 'x_mean < y_mean', uses
          `func=lambda x, y: np.mean(y) - np.mean(x))`
           for a one-sided test.
    method : 'approximate' or 'exact' (default: 'exact')
        If 'exact' (default), all possible permutations are considered.
        If 'approximate' the number of drawn samples is
        given by `num_rounds`.
        Note that 'exact' is typically not feasible unless the dataset
        size is relatively small.
    paired : bool
        If True, a paired test is performed by only exchanging each
        datapoint with its associate.
    num_rounds : int (default: 1000)
        The number of permutation samples if `method='approximate'`.
    seed : int or None (default: None)
        The random seed for generating permutation samples if
        `method='approximate'`.
    Returns
    ----------
    p-value under the null hypothesis
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/
    """

    if method not in ('approximate', 'exact'):
        raise AttributeError('method must be "approximate"'
                             ' or "exact", got %s' % method)

    if isinstance(func, str):

        if func not in (
                'x_mean != y_mean', 'x_mean > y_mean', 'x_mean < y_mean'):
            raise AttributeError('Provide a custom function'
                                 ' lambda x,y: ... or a string'
                                 ' in ("x_mean != y_mean", '
                                 '"x_mean > y_mean", "x_mean < y_mean")')

        elif func == 'x_mean != y_mean':
            def func(x, y):
                return np.abs(np.mean(x) - np.mean(y))

        elif func == 'x_mean > y_mean':
            def func(x, y):
                return np.mean(x) - np.mean(y)

        else:
            def func(x, y):
                return np.mean(y) - np.mean(x)

    rng = np.random.RandomState(seed)

    m, n = len(x), len(y)

    if paired:
        if m != n:
            raise ValueError('x and y must have the same'
                             ' length if `paired=True`')
        sample_x = np.empty(m)
        sample_y = np.empty(n)

    else:
        combined = np.hstack((x, y))

    at_least_as_extreme = 0.
    reference_stat = func(x, y)

    # Note that whether we compute the combinations or permutations
    # does not affect the results, since the number of permutations
    # n_A specific objects in A and n_B specific objects in B is the
    # same for all combinations in x_1, ... x_{n_A} and
    # x_{n_{A+1}}, ... x_{n_A + n_B}
    # In other words, for any given number of combinations, we get
    # n_A! x n_B! times as many permutations; hoewever, the computed
    # value of those permutations that are merely re-arranged combinations
    # does not change. Hence, the result, since we divide by the number of
    # combinations or permutations is the same, the permutations simply have
    # "n_A! x n_B!" as a scaling factor in the numerator and denominator
    # and using combinations instead of permutations simply saves computational
    # time

    if method == "exact":

        if paired:
            for flip in product([True, False], repeat=m):
                for i, f in enumerate(flip):
                    if f:
                        sample_x[i], sample_y[i] = y[i], x[i]
                    else:
                        sample_x[i], sample_y[i] = x[i], y[i]

                diff = func(sample_x, sample_y)
                if diff > reference_stat or np.isclose(diff, reference_stat):
                    at_least_as_extreme += 1.0

            num_rounds = 2 ** n

        else:
            for indices_x in combinations(range(m + n), m):
                indices_y = [i for i in range(m + n) if i not in indices_x]
                diff = func(combined[list(indices_x)], combined[indices_y])

                if diff > reference_stat or np.isclose(diff, reference_stat):
                    at_least_as_extreme += 1.0
            num_rounds = factorial(m + n) / (factorial(m) * factorial(n))

    else:
        if paired:
            for i in range(num_rounds):
                flip = rng.randn(m) > 0.

                for i, f in enumerate(flip):
                    if f:
                        sample_x[i], sample_y[i] = y[i], x[i]
                    else:
                        sample_x[i], sample_y[i] = x[i], y[i]

                diff = func(sample_x, sample_y)
                if diff > reference_stat or np.isclose(diff, reference_stat):
                    at_least_as_extreme += 1.

            # To cover the actual experiment results
            at_least_as_extreme += 1.
            num_rounds += 1.

        else:
            for i in range(num_rounds):
                rng.shuffle(combined)
                diff = func(combined[:m], combined[m:])

                if diff > reference_stat or np.isclose(diff, reference_stat):
                    at_least_as_extreme += 1.

            # To cover the actual experiment results
            at_least_as_extreme += 1.
            num_rounds += 1.

    return at_least_as_extreme / num_rounds

if __name__ == '__main__':

    argparse = argparse.ArgumentParser(description="Set Params") 
    argparse.add_argument('--pred_dir', type=str, help="Path containing all predictions of main model")
    argparse.add_argument('--alt_dir', type=str, help="Path containing all predictions of alt model")

    args = argparse.parse_args()
    
    pred_dfs = []
    alt_dfs = []
    
    pred_csvs = sorted(glob.glob(os.path.join(args.pred_dir, "*.csv")))
    alt_csvs = sorted(glob.glob(os.path.join(args.alt_dir, "*.csv")))
   
    print('pred csv: ', pred_csvs)
    print('alt csv: ', alt_csvs)
    for p_csv,a_csv in zip(pred_csvs,alt_csvs):
        pred_dfs.append(pd.read_csv(p_csv, header=None))
        alt_dfs.append(pd.read_csv(a_csv, header=None))

    if len(pred_dfs) !=  len(alt_dfs):
        print(f'Non-matching number of pred and alt files: {len(pred_dfs)} and {len(alt_dfs)}, respectively')
        sys.exit(0)

    if len(pred_dfs) == 1:
        preds = sorted(pred_dfs[0].iloc[:,-1].values.tolist())
        alts = sorted(alt_dfs[0].iloc[:,-1].values.tolist())

        p = permutation_test(preds, alts, num_rounds=10000, paired=True, method='approximate', seed=1337, func='x_mean != y_mean')
        print(f"p-val: {p}")
    else:
        p_t_vals = []
        ps = []
        for idx in range(len(pred_dfs)):
            preds = sorted(pred_dfs[idx].iloc[:,-1].values.tolist())
            alts = sorted(alt_dfs[idx].iloc[:,-1].values.tolist())

            print(len(preds))
            print(len(alts))
            
            p = permutation_test(preds, alts, num_rounds=10000, paired=True, method='approximate', seed=1337, func='x_mean != y_mean')
            print(f'p-val: {p}')
            ps.append(p)
        
        dist, combined_p = combine_pvalues(ps)
        print(f'scipy fisher method, chi-dist: {dist} combine_pvalue: {combined_p}')
