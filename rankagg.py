import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colormaps
import os
import pickle
import warnings
from tqdm import tqdm
import pickle
import time

import ranky

import pymc3 as pm
import arviz as az

from scipy.stats import norm, multivariate_normal
from scipy.stats import rankdata

plt.style.use('bmh')


def bordaRanking(df: pd.DataFrame, name_column: str, list_columns: list,
                 name_label:str, weights=None, save_path=None):
    # BORDA METHOD
    # df: pandas dataframe containing the outcomes
    # name_column: which column contains treatment names
    # list_columns: list of all the columns that should be considered outcomes
    # name_label: name of the column containing treatments in the output table
    # weights: list of weights. If None they are all equal
    # save_path: where to save the final csv (name included). If None, nothing is saved
    L = len(list_columns)
    if weights is None:
        weights = np.ones(L)
    else:
        weights = np.array(weights)
        weights = L * weights / np.sum(weights)
    if len(weights) != L:
        raise Exception('Different number of d and weight columns!')
    lists = np.column_stack([df[l] for l in list_columns])
    lists = np.nan_to_num(lists, nan=-np.inf)
    ranks = rankdata(lists, method='min', axis=0) - 1
    scores = np.array([np.dot(r, weights) for r in ranks])
    agg_ranks = rankdata(-scores, method='min')
    adf = pd.DataFrame(df[name_column], columns=[name_label])
    adf['agg_ranks'] = agg_ranks
    adf['score'] = scores
    adf = adf.sort_values('agg_ranks').reset_index(drop=True)
    if save_path is not None:
        adf.to_csv(save_path, index=None)
    return adf


def mc4Ranking(df: pd.DataFrame, name_column: str, list_columns: list,
               name_label: str, save_path=None, eps=1e-4, weights=None):
    # MARKOV CHAIN 4 METHOD
    # df: pandas dataframe containing the outcomes
    # name_column: which column contains treatment names
    # list_columns: list of all the columns that should be considered outcome scores
    # name_label: name of the column containing treatments in the output table
    # weights: list of weights. If None they are all equal
    # eps: edge weight for non-edges, non-zero to avoid issues while calculating eigenvectors
    # save_path: where to save the final csv (name included). If None, nothing is saved
    N = df.shape[0]
    tidx = np.arange(N)
    A = np.zeros((N, N))
    if weights is None:
        weights = np.ones(len(list_columns)).astype(int)
    else:
        weights = np.array(weights).astype(int)
    repcols = []
    for l, w in zip(list_columns, weights):
        repcols.extend([l for _ in range(w)])
    thresh = len(repcols) / 2
    for ti in tidx:
        for tj in tidx:
            if ti != tj:
                check = []
                for l in repcols:
                    if df[l].iloc[ti] != float('nan') and df[l].iloc[tj] != float('nan'):
                        check.append(df[l].iloc[ti] <= df[l].iloc[tj])
                if np.sum(check) >= thresh:
                    A[ti][tj] = 1 / N - eps
                else:
                    A[ti][tj] = eps
        A[ti][ti] = 1 - np.sum(A[ti])
    evals, pis = np.linalg.eig(A.T)
    pi = pis[:, np.isclose(evals, 1)][:, 0]
    pi = np.real(pi / np.sum(pi))
    agg_ranks = rankdata(-pi, method='min')
    adf = pd.DataFrame(df[name_column], columns=[name_label])
    adf['agg_ranks'] = agg_ranks
    adf['score'] = pi
    adf = adf.sort_values('agg_ranks').reset_index(drop=True)
    if save_path is not None:
        adf.to_csv(save_path, index=None)
    return adf


def roidRanking(df: pd.DataFrame, name_column: str, list_columns: list,
                name_label: str, weights=None, save_path=None):
    # GRAPH BASED (ROID) METHOD
    # df: pandas dataframe containing the outcomes
    # name_column: which column contains treatment names
    # list_columns: list of all the columns that should be considered outcome scores
    # name_label: name of the column containing treatments in the output table
    # weights: list of weights. If None they are all equal
    # save_path: where to save the final csv (name included). If None, nothing is saved
    L = len(list_columns)
    if weights is None:
        weights = np.ones(L)
    weights = np.array(weights)
    if len(weights) != L:
        raise Exception('Different number of d and weight columns!')
    N = df.shape[0]
    tidx = np.arange(N)
    A = np.zeros((N, N)) + np.eye(N)
    for ti in tidx:
        for tj in tidx:
            if ti != tj:
                check = []
                weis = []
                for li in range(L):
                    l = list_columns[li]
                    if df[l].iloc[ti] != float('nan') and df[l].iloc[tj] != float('nan'):
                        check.append(df[l].iloc[ti] <= df[l].iloc[tj])
                        weis.append(weights[li])
                weis = np.array(weis)
                A[ti][tj] = np.sum(weis[check])
    roids = np.array([np.sum(A[:, ti]) / np.sum(A[ti, :]) for ti in tidx])
    agg_ranks = rankdata(-roids, method='min')
    adf = pd.DataFrame(df[name_column], columns=[name_label])
    adf['agg_ranks'] = agg_ranks
    adf['score'] = roids
    adf = adf.sort_values('agg_ranks').reset_index(drop=True)
    if save_path is not None:
        adf.to_csv(save_path, index=None)
    return adf


def genPscoreRanking(df: pd.DataFrame, name_column: str, d_columns: list,
                     se_columns: list, name_label: str, save_path=None):
    # GENERALIZED P-SCORES METHOD
    # df: pandas dataframe containing the outcomes
    # name_column: which column contains treatment names
    # d_columns: list of all the columns that should be considered efficacy scores
    # se_columns: list of all the columns that should be considered st-errs of efficacy scores
    # name_label: name of the column containing treatments in the output table
    # save_path: where to save the final csv (name included). If None, nothing is saved
    N = df.shape[0]
    L = len(d_columns)
    if L != len(se_columns):
        raise Exception('Different number of d and se columns!')
    # covariances is calculated from d_columns. The code should be modified if
    # the user wants to use covariances from multivariate analyses
    corrs = np.corrcoef(np.nan_to_num(df[d_columns].values.T, nan=np.nanmax(df[d_columns])))
    pscores = []
    for i in range(N):
        pscore = 0
        for j in range(N):
            if i != j:
                normeans = []
                cidx = []
                for l in range(L):
                    mean = df[d_columns[l]].iloc[j] - df[d_columns[l]].iloc[i]
                    if not np.isnan(mean):
                        sigma = np.sqrt(df[se_columns[l]].iloc[j]**2 +\
                                        df[se_columns[l]].iloc[i]**2)
                        if sigma == 0:
                            normeans.append(mean)
                        else:
                            normeans.append(mean / sigma)
                        cidx.append(l)
                if len(normeans) > 0:
                    corrs_denanned = corrs.copy()[cidx]
                    corrs_denanned = corrs_denanned[:, cidx]
                    pscore += multivariate_normal.cdf(normeans, mean=np.zeros(len(normeans)),
                                                      cov=corrs_denanned)
        pscores.append(pscore / (N - 1))
    pscores = np.array(pscores)
    agg_ranks = rankdata(-pscores, method='min')
    adf = pd.DataFrame(df[name_column], columns=[name_label])
    adf['agg_ranks'] = agg_ranks
    adf['score'] = pscores
    adf = adf.sort_values('agg_ranks').reset_index(drop=True)
    if save_path is not None:
        adf.to_csv(save_path, index=None)
    return adf


def mcNmaRanking(df: pd.DataFrame, name_column: str, d_columns: list, se_columns: list,
                 name_label: str,  weights=None, stay_param=0.5, save_path=None):
    # MARKOV CHAIN NMA METHOD
    # df: pandas dataframe containing the outcomes
    # name_column: which column contains treatment names
    # d_columns: list of all the columns that should be considered efficacy scores
    # se_columns: list of all the columns that should be considered st-errs of efficacy scores
    # name_label: name of the column containing treatments in the output table
    # weights: list of weights. If None they are all equal
    # stay_parameter: value of the stay parameter
    # save_path: where to save the final csv (name included). If None, nothing is saved
    N = df.shape[0]
    L = len(d_columns)
    if weights is None:
        weights = np.ones(L)
    weights = np.array(weights)
    if len(weights) != L:
        raise Exception('Different number of d and weight columns!')
    if len(se_columns) != L:
        raise Exception('Different number of d and se columns!')
    if hasattr(stay_param, '__len__'):
        if len(stay_param) != N:
            raise Exception('if stay_param is a list, its lenght has'
                            'to be the number of elements')
    else:
        stay_param = np.ones(N) * stay_param
    A = np.zeros((N, N)) + np.diag(stay_param)
    for i in range(N):
        for j in range(N):
            if i != j:
                probs = []
                weis = []
                for l in range(L):
                    mean = df[d_columns[l]].iloc[j] - df[d_columns[l]].iloc[i]
                    if not np.isnan(mean):
                        sigma = np.sqrt(df[se_columns[l]].iloc[j]**2 +\
                                        df[se_columns[l]].iloc[i]**2)
                        probs.append(norm.cdf(mean / sigma))
                        weis.append(weights[l])
                weis = np.array(weis) / np.sum(weis)
                A[j][i] = np.dot(probs, weis)
    for i in range(N):
        A[i] = A[i] / np.sum(A[i])
    evals, pis = np.linalg.eig(A.T)
    pi = pis[:, np.isclose(evals, 1)][:, 0]
    pi = np.real(pi / np.sum(pi))
    agg_ranks = rankdata(-pi, method='min')
    adf = pd.DataFrame(df[name_column], columns=[name_label])
    adf['agg_ranks'] = agg_ranks
    adf['score'] = pi
    adf = adf.sort_values('agg_ranks').reset_index(drop=True)
    if save_path is not None:
        adf.to_csv(save_path, index=None)
    return adf


def bayesianRanking(df: pd.DataFrame, name_column: str, list_columns: list, name_label: str,
                    bounded=True, upper_bound=1, lower_bound=0, tau_max=5, niters=5000,
                    burn_in=5000, chains=4, cores=None, bayesian_save=None, aggrank_save=None,
                    bayes_show=False, weights=None):
    # BAYESIAN INFERENCE METHOD
    # df: pandas dataframe containing the outcomes
    # name_column: which column contains treatment names
    # list_columns: list of all the columns that should be considered outcome scores
    # name_label: name of the column containing treatments in the output table
    # bounded: if the score distribution has bounds
    # upper_bound: upper bound of the score distribution
    # lower_bound: lower bound of the score distribution
    # tau_max: upper bound of the heterogeneity distribution (lower is zero)
    # niters: number of MC iterations
    # burn_in: tuning steps of MCMC
    # chains: number of MCMC chains
    # cores: number of cores used by the function (if None is all)
    # bayesian_save: here to save the bayesian output csv (name included). If None, nothing is saved
    # aggrank_save: where to save the final csv (name included). If None, nothing is saved
    # bayes_show: show bayesian results?
    # weights: list of weights. If None they are all equal
    L = len(list_columns)
    if weights is None:
        weights = np.ones(L).astype(int)
    else:
        weights = np.array(weights).astype(int)
    if len(weights) != L:
        raise Exception('Different number of d and weight columns!')
    if not np.issubdtype(weights.dtype, np.integer):
        raise Exception('bayesianRanking only takes integer weights')
    N = df.shape[0]
    if not bounded:
        upper_bound = np.inf
        lower_bound = -np.inf
    model = pm.Model()
    with model:
        tau = pm.Uniform("tau", lower=0, upper=tau_max)
        aggvars = []
        if bounded:
            for i in range(N):
                aggvars.append(pm.Uniform(df[name_label].iloc[i],
                                          lower=lower_bound, upper=upper_bound))
        else:
            for i in range(N):
                aggvars.append(pm.Normal(df[name_label].iloc[i], mu=0, sigma=1e4))
        for li in range(L):
            l = list_columns[li]
            for w in range(weights[li]):
                for i in range(N):
                    if not np.isnan(df[l].iloc[i]):
                        single_rank = pm.TruncatedNormal(f"{l}_{w}_{i}",mu=aggvars[i],sigma=tau, 
                                                         lower=lower_bound, upper=upper_bound,
                                                         observed=df[l].iloc[i])
        startime = time.time()
        print('Starting to sample (this will take a while, especially after the prog-bar)...')
        trace = pm.sample(niters, tune=burn_in, cores=cores, chains=chains,
                          return_inferencedata=True)
    print('Done!')
    elapstime = (time.time() - startime) // 60
    print(f"\nBayesian sampling is over! [tte = {elapstime} mins]")
    bdf = az.summary(trace, var_names=[*df[name_label].values, 'tau'], hdi_prob=0.90)
    if bayes_show:
        display(bdf)
    if bayesian_save is not None:
        bdf.to_csv(bayesian_save, index=None)
    agg_ranks = rankdata(-bdf['mean'].iloc[:-1], method='min')
    adf = pd.DataFrame(list(bdf.index)[:-1], columns=[name_label])
    adf['agg_ranks'] = agg_ranks
    adf['score'] = bdf['mean'].values[:-1]
    adf = adf.sort_values('agg_ranks').reset_index(drop=True)
    if aggrank_save is not None:
        adf.to_csv(aggrank_save, index=None)
    return adf


def spieRanking(df: pd.DataFrame, name_column: str, list_columns: list, name_label: str,
                list_names: list, weights=None, aggrank_save=None, plotprefix_save=None,
                cmap='viridis', plotname_column=None, show=False):
    # SPIE CHARTS METHOD
    # df: pandas dataframe containing the outcomes
    # name_column: which column contains treatment names
    # list_columns: list of all the columns that should be considered outcome scores
    # name_label: name of the column containing treatments in the output table
    # weights: list of weights. If None they are all equal
    # aggrank_save: where to save the final csv (name included). If None, nothing is saved
    # plotprefix_path: folder where to save the spie chart. If None, nothing is saved
    # list_names: list of outcome labels
    # cmap: colormap of the spie chart sectors
    # plotname_column: how to show the element name in the spie chart
    # show: show the spie chart?
    if plotprefix_save is not None:
        plotprefix_save = f"{plotprefix_save}_charts/"
        if not os.path.exists(plotprefix_save):
            os.makedirs(plotprefix_save)
    df = df.fillna(0)
    N = df.shape[0]
    L = len(list_columns)
    if weights is None:
        weights = np.ones(L)
    if len(weights) != L:
        raise Exception('Different number of d and weight columns!')
    weights = 2 * np.pi * np.array(weights) / np.sum(weights)
    aggscores = np.zeros(N)
    mappa = colormaps.get_cmap(cmap)
    if plotname_column is None:
        plotname_column = name_column
    for n, line in df.iterrows():
        aggscores[n] = 0.5 * np.dot(weights, line[list_columns]**2) / np.pi
        figa, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={'projection': 'polar'})
        cumweights = np.concatenate(([0], np.cumsum(weights)))
        for i in range(L):
            xvals = np.arange(cumweights[i], cumweights[i+1], 0.005)
            center = 0.5 * (cumweights[i] + cumweights[i+1])
            ax.fill_between(xvals, line[list_columns[i]], zorder=i+10,
                            color=mappa(0.5 * cumweights[i] / np.pi), label=list_names[i])
            ax.plot([xvals[0], *xvals, xvals[-1]],
                    np.concatenate(([0], [line[list_columns[i]]
                                          for _ in range(len(xvals))], [0])),
                    color='xkcd:black', zorder=200, linewidth=1)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.16), ncols=4, fontsize=12)
        ax.set_ylim(0, 1.)
        ax.set_xticks(cumweights)
        ax.set_yticks(np.arange(0, 1.01, 0.1), minor=True)
        ax.set_yticks(np.arange(0, 1.01, 0.2), minor=False)
        ax.grid(linestyle=':', which='minor')
        ax.grid(linestyle='-', which='major')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"{line[plotname_column].capitalize()} ({np.round(aggscores[n], 3)})",
                     fontsize=14, pad=14)
        if plotprefix_save is not None:
            plt.savefig(f"{plotprefix_save}spie-chart_{line[name_column]}.pdf",
                        bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(figa)
    agg_ranks = rankdata(-aggscores, method='min')
    adf = pd.DataFrame(df[name_column], columns=[name_label])
    adf['agg_ranks'] = agg_ranks
    adf['score'] = aggscores
    adf = adf.sort_values('agg_ranks').reset_index(drop=True)
    if aggrank_save is not None:
        adf.to_csv(aggrank_save, index=None)
    return adf


def kendallRanking(df: pd.DataFrame, name_column: str, list_columns: list,
                   name_label: str, save_path=None, weights=None):
    # KEMENY OPTIMIZATION (KENDALL) METHOD
    # df: pandas dataframe containing the outcomes
    # name_column: which column contains treatment names
    # list_columns: list of all the columns that should be considered outcome scores
    # name_label: name of the column containing treatments in the output table
    # weights: list of weights. If None they are all equal
    # save_path: where to save the final csv (name included). If None, nothing is saved
    if weights is None:
        weights = np.ones(len(list_columns)).astype(int)
    else:
        weights = np.array(weights).astype(int)
    repcols = []
    for l, w in zip(list_columns, weights):
        repcols.extend([l for _ in range(w)])
    lists = np.column_stack([df[l] for l in repcols])
    lists = np.nan_to_num(lists, nan=-np.inf)
    ranks = rankdata(-lists, method='min', axis=0)
    aggscores = ranky.center(ranks, method='kendalltau', verbose=False)
    agg_ranks = rankdata(aggscores, method='min')
    adf = pd.DataFrame(df[name_column], columns=[name_label])
    adf['agg_ranks'] = agg_ranks
    adf['score'] = aggscores
    adf = adf.sort_values('agg_ranks').reset_index(drop=True)
    if save_path is not None:
        adf.to_csv(save_path, index=None)
    return adf


def spearmanRanking(df: pd.DataFrame, name_column: str, list_columns: list,
                    name_label: str, save_path=None, weights=None):
    # KEMENY OPTIMIZATION (SPEARMAN) METHOD
    # df: pandas dataframe containing the outcomes
    # name_column: which column contains treatment names
    # list_columns: list of all the columns that should be considered outcome scores
    # name_label: name of the column containing treatments in the output table
    # weights: list of weights. If None they are all equal
    # save_path: where to save the final csv (name included). If None, nothing is saved
    if weights is None:
        weights = np.ones(len(list_columns)).astype(int)
    else:
        weights = np.array(weights).astype(int)
    repcols = []
    for l, w in zip(list_columns, weights):
        repcols.extend([l for _ in range(w)])
    lists = np.column_stack([df[l] for l in repcols])
    lists = np.nan_to_num(lists, nan=-np.inf)
    ranks = rankdata(-lists, method='min', axis=0)
    aggscores = ranky.center(ranks, method='spearman', verbose=False)
    agg_ranks = rankdata(aggscores, method='min')
    adf = pd.DataFrame(df[name_column], columns=[name_label])
    adf['agg_ranks'] = agg_ranks
    adf['score'] = aggscores
    adf = adf.sort_values('agg_ranks').reset_index(drop=True)
    if save_path is not None:
        adf.to_csv(save_path, index=None)
    return adf


def kendallDistance(ranks_a, ranks_b, normalize=False):
    # Calculates kendall tau
    T = len(ranks_a)
    if T != len(ranks_b):
        raise Exception('Rankings have different lenghts!')
    ranks_a, ranks_b = np.array(ranks_a), np.array(ranks_b)
    distance = np.sum(np.abs(ranks_a - ranks_b))
    distance = 0
    for i in range(T):
        for j in range(i):
            if (ranks_a[i] - ranks_a[j]) * (ranks_b[i] - ranks_b[j]) < 0:
                distance += 1
    if normalize:
        return 2 * distance / (T**2 - T)
    return distance


def spearmanDistance(ranks_a, ranks_b):
    # calculates spearman's footrule distance
    if len(ranks_a) != len(ranks_b):
        raise Exception('Rankings have different lenghts!')
    ranks_a, ranks_b = np.array(ranks_a), np.array(ranks_b)
    distance = np.sum(np.abs(ranks_a - ranks_b))
    return distance


def spearmanCorrelation(ranks_a, ranks_b, scores_a=None, scores_b=None,
                        reverse_b_scores=False, replace_nan=0):
    # calculates both Spearman and Pearson correlation
    if len(ranks_a) != len(ranks_b):
        raise Exception('Rankings have different lenghts!')
    if scores_a is None or scores_b is None:
        return np.corrcoef(ranks_a, ranks_b)[0][1], None
    scores_a = np.nan_to_num(scores_a, nan=replace_nan)
    if reverse_b_scores:
        scores_b = np.nan_to_num(-scores_b, nan=replace_nan)
    else:
        scores_b = np.nan_to_num(scores_b, nan=replace_nan)
    return np.corrcoef(ranks_a, ranks_b)[0][1], np.corrcoef(scores_a, scores_b)[0][1]