import argparse
import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import umap
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
import tools.loader
import tools.folder

from tqdm import tqdm
from termcolor import colored
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clustering(model, data_loader, args):
    encoder_features = []
    model_clustering = model.encoder_q
    encoder = model_clustering.eval()

    with torch.no_grad():
        for image, _, __ in tqdm(data_loader, desc='Extracting features'):
            image = image.cuda(args.gpu)
            features = encoder.fc.in_features(image)
            batch_size = features.size(0)
            features = features.cpu().numpy()
            for j in range(0, batch_size):
                encoder_features.append(features[j])

    #features = np.concatenate(features)
    print(np.array(encoder_features).shape)
    sys.exit()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        umap_reducer = umap.UMAP(n_components=64, random_state=args.seed)
        reduced_features = umap_reducer.fit_transform(encoder_features)
    
    n_components_range = range(7, 15)
    BIC = []

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=args.seed)
        gmm.fit(reduced_features)
        BIC.append(gmm.bic(reduced_features))

    best_BIC_idx = np.argmin(BIC)
    best_BIC_clusters = n_components_range[best_BIC_idx]

    print(f'Best number of clusters according to BIC: {best_BIC_clusters}')

    gmm = GaussianMixture(n_components=best_BIC_clusters, random_state=args.seed)
    gmm.fit(reduced_features)
    labels = gmm.predict(reduced_features)

    silhouette = silhouette_score(reduced_features, labels)
    calinski_harabasz = calinski_harabasz_score(reduced_features, labels)
    davies_bouldin = davies_bouldin_score(reduced_features, labels)

    return silhouette, calinski_harabasz, davies_bouldin, best_BIC_clusters

