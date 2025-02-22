import torch
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr


def euclidean_distance(x):
    torch.set_default_tensor_type('torch.DoubleTensor')
    nx = x.size(0)
    x = x.contiguous()
    
    x = x.view(nx, -1)

    norm_x = torch.sum(x ** 2, 1, keepdim=True).t()
    ones_x = torch.ones(nx, 1)

    xTx = torch.mm(ones_x, norm_x)
    xTy = torch.mm(x, x.t())
    
    d = (xTx.t() + xTx - 2 * xTy)
    d[d < 0] = 0

    return d 


def poincare_distance(x):
    """ 
        Poincaré distance matrix
        Code is adapted from Poincaré Maps:
            https://github.com/facebookresearch/PoincareMaps/blob/main/poincare_maps.py
            Creative Commons Attribution-NonCommercial 4.0 International Public License
    """
    x = torch.DoubleTensor(x)
    eps = 1e-5
    boundary = 1 - eps
    
    nx = x.size(0)
    x = x.contiguous()
    x = x.view(nx, -1)
    
    norm_x = torch.sum(x ** 2, 1, keepdim=True)
    sqdist = euclidean_distance(x) * 2    
    squnorm = 1 - torch.clamp(norm_x, 0, boundary)

    x = (sqdist / torch.mm(squnorm, squnorm.t())) + 1
    z = torch.sqrt(torch.pow(x, 2) - 1)
    
    dist_mat = torch.log(x + z)
    
    return dist_mat.detach().cpu().numpy()


def evaluate_global(distance_matrix, global_distance_matrix, verbose, n_repeats, n_pairs, n_triplets):
    
    metrics = {}
    metrics["Spearman correlation"] = []
    metrics["Spearman p_value"] = []
    metrics["Random triplet accuracy"] = []

    for i in range(n_repeats):
        pair_indices = []
        a_indices = np.random.choice(np.arange(distance_matrix.shape[0]-1), size=n_pairs, replace=True, p=None)
        a_indices, a_counts = np.unique(a_indices, return_counts = True)
        for a, counts in zip(a_indices, a_counts):
            b_indices = np.random.choice(np.arange(a+1, distance_matrix.shape[0]), size=min(counts, distance_matrix.shape[0]-a-1), replace=False, p=None)
            for b in b_indices:
                pair_indices += [[a,b]]
        pair_indices = tuple(np.array(pair_indices).T.tolist())
        corr = spearmanr(distance_matrix[pair_indices], global_distance_matrix[pair_indices])
        metrics["Spearman correlation"] += [corr.correlation]
        metrics["Spearman p_value"] += [corr.pvalue]
         
        accuracy = 0
        for j in range(n_triplets):
            triplet_indices = np.random.choice(np.arange(distance_matrix.shape[0]), size=3, replace=False, p=None)
            order = np.argsort([distance_matrix[triplet_indices[0], triplet_indices[1]],
                      distance_matrix[triplet_indices[0], triplet_indices[2]],
                      distance_matrix[triplet_indices[1], triplet_indices[2]]])
            original_order = np.argsort([global_distance_matrix[triplet_indices[0], triplet_indices[1]],
                      global_distance_matrix[triplet_indices[0], triplet_indices[2]],
                      global_distance_matrix[triplet_indices[1], triplet_indices[2]]])
            if np.array_equal(order, original_order):
                accuracy += 1/n_triplets
        metrics["Random triplet accuracy"] += [accuracy]

    if verbose:
        print("Spearman correaltion\t" + str(round(np.mean(metrics["Spearman correlation"]),4)) + "\t~  " + str(round(np.std(metrics["Spearman correlation"]),4)))
        print("p-value\t\t\t" + str(round(np.mean(metrics["Spearman p_value"]),4)) + "\t~  " + str(round(np.std(metrics["Spearman p_value"]),4)))
        print("Random triplet accuracy\t" + str(round(np.mean(metrics["Random triplet accuracy"]),4)) + "\t~  " + str(round(np.std(metrics["Random triplet accuracy"]),4)))

    return metrics


def evaluate_knn(distance_matrix, verbose, top_n, target_feature, df_meta, n_splits, n_neighbors, weights):

    top_n_classes = df_meta[target_feature].value_counts().nlargest(top_n).index.values
    extra = 0
    if "unknown" in top_n_classes:
        extra += 1
    if "NA" in top_n_classes:
        extra += 1
    if "-" in top_n_classes:
        extra += 1
    top_n_classes = df_meta[target_feature].value_counts().nlargest(top_n+extra).index.values
    top_n_classes = np.delete(top_n_classes, np.where(top_n_classes == "unknown"))
    top_n_classes = np.delete(top_n_classes, np.where(top_n_classes == "NA"))
    top_n_classes = np.delete(top_n_classes, np.where(top_n_classes == "-"))
        
    df_top_n_classes = df_meta[df_meta[target_feature].isin(top_n_classes)]
    df_top_n_classes["target"] = pd.factorize(df_top_n_classes[target_feature])[0]
    y = df_top_n_classes["target"].values
    top_n_indices = df_meta.index.get_indexer(df_top_n_classes.index)
    distance_matrix = distance_matrix[:,top_n_indices][top_n_indices,:]
    if verbose:
        print(top_n_classes.shape)
        print(top_n_classes)
        print()

    metrics = {}
    for metric in ["accuracy", "top_3_accuracy", "top_5_accuracy", "precision", "recall", "f1", "roc_auc"]:
        metrics[metric] = []

    kf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
    for train_indices, val_indices in kf.split(distance_matrix,y):
        X_train = distance_matrix[train_indices,:][:,train_indices]
        y_train = y[train_indices]
        X_valid = distance_matrix[val_indices,:][:,train_indices]
        y_valid = y[val_indices]

        KNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric='precomputed', n_jobs=None)
        KNN.fit(X_train, y_train)
        pred_probas = KNN.predict_proba(X_valid)
        preds = KNN.predict(X_valid)

        metrics["accuracy"] += [sklearn.metrics.accuracy_score(y_valid, preds)]
        metrics["precision"] += [sklearn.metrics.precision_score(y_valid, preds, average = 'macro')]
        metrics["recall"] += [sklearn.metrics.recall_score(y_valid, preds, average = 'macro')]
        metrics["f1"] += [sklearn.metrics.f1_score(y_valid, preds, average = 'macro')]
        if pred_probas.shape[1] == 2:
            metrics["top_3_accuracy"] += [0]
            metrics["top_5_accuracy"] += [0]
            metrics["roc_auc"] += [sklearn.metrics.roc_auc_score(y_valid, pred_probas[:,1])]
        else:
            metrics["top_3_accuracy"] += [sklearn.metrics.top_k_accuracy_score(y_valid, pred_probas, k=3)]
            metrics["top_5_accuracy"] += [sklearn.metrics.top_k_accuracy_score(y_valid, pred_probas, k=5)]
            metrics["roc_auc"] += [sklearn.metrics.roc_auc_score(y_valid, pred_probas, multi_class="ovr", average='macro')]

    if verbose:
        for k, v in metrics.items():
            print(k + "\t\t" + str(round(np.mean(v),4)) + "\t~  " + str(round(np.std(v),4)))
            
    return metrics


def evaluate(embeddings, distance, df_meta,
             knn_n_splits = 5, knn_n_neighbors = 5, knn_weights = "distance", knn_targets = [(20, "MOA"),(20, "Cell"),(2, "Time")],
             global_n_repeats = 5, global_n_pairs = 5000, global_n_triplets = 5000, global_distance_matrix = None):

    if distance == "euclidean":
        distance_matrix = sklearn.metrics.pairwise.euclidean_distances(embeddings)
    elif distance == "cosine":
        distance_matrix = 1 - sklearn.metrics.pairwise.cosine_similarity(embeddings) 
        np.fill_diagonal(distance_matrix,0)
    elif distance == "poincaré":
        distance_matrix = poincare_distance(embeddings)
    elif distance == "canberra":
        distance_matrix = sklearn.metrics.pairwise_distances(embeddings, metric = "canberra")

    metrics = {}
    for top_n, target_feature in knn_targets:
        metric = evaluate_knn(distance_matrix, verbose = False,
                             top_n = top_n, target_feature = target_feature, df_meta = df_meta,
                             n_splits = knn_n_splits, n_neighbors = knn_n_neighbors, weights = knn_weights)
        metrics["TOP_" + str(top_n) + "_" + target_feature] = metric
            
    if global_distance_matrix is not None:     
        global_metrics = evaluate_global(distance_matrix, global_distance_matrix, 
                                         False, global_n_repeats, global_n_pairs, global_n_triplets)
        metrics["global"] = {}
        for k, v in global_metrics.items():
            metrics["global"][k] = v

    return metrics