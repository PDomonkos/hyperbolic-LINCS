import numpy as np
import pandas as pd

import statistics
from statsmodels.formula.api import ols

import networkx as nx

import sklearn
import sklearn.metrics


def get_scale_free_soft_treshold_power(similarity_matrix, powers = np.arange(1, 11), RsquaredCut=0.9, MeanCut=100, nBreaks=10, verbose = False):
    """
        Code is adapted from WGCNA:
            https://github.com/mortazavilab/PyWGCNA/blob/main/PyWGCNA/wgcna.py
            MIT License
            Copyright (c) 2022 Narges Rezaie
    """
    
    nGenes = similarity_matrix.shape[1]
    nPowers = len(powers)

    colname = ["Power", "SFT.R.sq", "slope", "mean(k)"]
    sft = pd.DataFrame(np.full((len(powers), len(colname)), None), columns=colname, dtype=object)
    sft['Power'] = powers

    powers1 = [0]
    powers1.extend(powers[:-1])
    powerSteps = powers - powers1
    uniquePowerSteps = np.unique(powerSteps)
    corrPowers = pd.DataFrame()
    for p in uniquePowerSteps:
        corrPowers[p] = [similarity_matrix ** p]

    datk = np.zeros((nGenes, len(powers)))
    corrPrev = np.ones(similarity_matrix.shape)
    for j in range(nPowers):
        corrCur = corrPrev * corrPowers[powerSteps[j]][0]
        datk[:, j] = np.nansum(corrCur, axis=0) - 1
        corrPrev = corrCur

    for i in range(len(powers)):

        khelp = datk[:, i]
        df = pd.DataFrame({'data': khelp})

        df['discretized_k'] = pd.cut(df['data'], nBreaks)

        dk = df.groupby('discretized_k', observed=False).mean()
        dk = pd.DataFrame(dk.reset_index())
        dk.columns = ['discretized_k', 'dk']

        p_dk = df['discretized_k'].value_counts() / len(khelp)
        p_dk = pd.DataFrame(p_dk.reset_index())
        p_dk.columns = ['discretized_k', 'p_dk']

        breaks1 = np.linspace(start=min(khelp), stop=max(khelp), num=nBreaks + 1)
        y, edges = np.histogram(df['data'], bins=breaks1)
        dk2 = 0.5 * (edges[1:] + edges[:-1])

        df = pd.merge(dk, p_dk, on='discretized_k')
        if df['dk'].isnull().values.any():
            df.loc[df['dk'].isnull().values, 'dk'] = dk2[df['dk'].isnull().values]
        if np.any(df['dk'] == 0):
            df.loc[df['dk'] == 0, 'dk'] = dk2[df['dk'] == 0]
        if df['p_dk'].isnull().values.any():
            df.loc[df['p_dk'].isnull().values, 'p_dk'] = 0

        df['log_dk'] = np.log10(df['dk'] + 1e-09)
        df['log_p_dk'] = np.log10(df['p_dk'] + 1e-09)

        model1 = ols(formula='log_p_dk ~ log_dk', data=df).fit()
        SFT1 = pd.DataFrame({'Rsquared.SFT': [model1.rsquared],
                                'slope.SFT': [model1.params.values[1]]})
        sft.loc[i, 'SFT.R.sq'] = SFT1.loc[0, 'Rsquared.SFT']
        sft.loc[i, 'slope'] = SFT1.loc[0, 'slope.SFT']
        sft.loc[i, 'mean(k)'] = statistics.mean(khelp)

    sft["signed R^2"] = -1 * np.sign(sft['slope']) * sft['SFT.R.sq']

    ind = np.logical_and(sft['SFT.R.sq'] > RsquaredCut, sft['mean(k)'] <= MeanCut)
    if np.sum(ind) > 0:
        powerEstimate = np.min(powers[ind])
        if verbose:
            print(f"Selected power to have scale free network is {str(powerEstimate)}.")
    else:
        ind = np.argsort(sft['SFT.R.sq']).tolist()
        powerEstimate = powers[ind[-1]]
        if verbose:
            print(f"No power detected to have scale free network!\nFound the best given power which is {str(powerEstimate)}.")

    return sft, powerEstimate


def generate_synthetic_euclidean_network(N, r, R = 1):

    """ Sample nodes """
    sampled_rho = R * np.sqrt(np.random.rand(N))
    sampled_phi = np.random.rand(N) * 2 * np.pi

    sampled_x = sampled_rho * np.cos(sampled_phi)
    sampled_y = sampled_rho * np.sin(sampled_phi)
    samples = np.vstack([sampled_x, sampled_y]).T

    coordinates = {}
    for i, coord in enumerate(samples):
        coordinates[i] = coord

    """ Sample edges """
    distances = sklearn.metrics.pairwise.euclidean_distances(samples)
    adjacency_matrix = (distances < r).astype(int) - np.eye(N)
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())

    """ Create network """
    G = nx.empty_graph(N)
    G.add_edges_from(edges)

    return G, coordinates, distances, adjacency_matrix


def hyperbolic_distance(polar_coordinates):
  distance_matrix = np.zeros((polar_coordinates.shape[0], polar_coordinates.shape[0]))
  for i in range(polar_coordinates.shape[0]-1):
    for j in range(i+1, polar_coordinates.shape[0]):
      u = polar_coordinates[i]
      v = polar_coordinates[j]
      duv = np.arccosh(np.cosh(u[0])*np.cosh(v[0])- np.sinh(u[0])*np.sinh(v[0])*np.cos(v[1]-u[1]))
      distance_matrix[i,j] = duv
      distance_matrix[j,i] = duv
  return distance_matrix


def generate_synthetic_hyperbolic_network(N, R = 1, r = 1):

    """ 
        Sample nodes 
            https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere
            PDF for the radial coordinate in: Hyperbolic geometry of complex networks: eq 7. https://arxiv.org/pdf/1006.5169
            Integrate PDF to get CDF, then use inverse uniform sampling.
    """
    random_directions = np.random.normal(size=(2, N))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = np.arccosh(np.random.rand(N) * (np.cosh(R) - 1) + 1)
    sampled_cartesian = (random_directions * random_radii).T

    sampled_rho = np.sqrt(sampled_cartesian[:,0]**2 + sampled_cartesian[:,1]**2)
    sampled_phi = np.arctan2(sampled_cartesian[:,1], sampled_cartesian[:,0])
    sampled_polar = np.vstack([sampled_rho, sampled_phi]).T

    coordinates = {}
    for i, coord in enumerate(sampled_cartesian):
        coordinates[i] = coord

    """ Sample edges """
    distances = hyperbolic_distance(sampled_polar)
    adjacency_matrix = (distances < r).astype(int) - np.eye(N)
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())

    """ Create network """
    G = nx.empty_graph(N)
    G.add_edges_from(edges)

    return G, coordinates, distances, adjacency_matrix