import torch
import numpy as np
import time
from sklearn.cluster import KMeans, MiniBatchKMeans
import sys

def k_means(x, H, device, mini_batch_kmeans=False):
    divergent = False
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], -1)



    Mu = torch.zeros((H, x.shape[1]))
    Sd = torch.zeros((H,))

    if not mini_batch_kmeans:
        try:
            print("Kmeans started ...")
            start = time.time()
            kmeans = KMeans(n_clusters=H, random_state=0).fit(x)
            end = time.time()
            print("Centres found after %.2f seconds ..." % (end - start))
            mu = kmeans.cluster_centers_
            labels = np.array(kmeans.labels_)
            sd = []
            for h in range(H):
                index = np.argwhere(labels == h)
                data = x[np.reshape(index, (index.shape[0],))]
                distance = (np.sum((data - mu[h]) ** 2, axis=1)) ** (0.50)
                maximum_distance = np.max(distance)
                if maximum_distance > 0:
                    sd.append(maximum_distance / float(3))
                else:
                    sd.append(sys.float_info.epsilon)
            sd = np.array(sd)

            Sd = torch.tensor(sd, dtype=torch.float32)
            Mu = torch.tensor(mu, dtype=torch.float32)

            Sd = Sd.to(device)
            Mu = Mu.to(device)

            print('K-means is Done!')
        except ValueError:
            divergent = True

    else:

        try:
            print("Minibatch-Kmeans started ...")
            kmeans = MiniBatchKMeans(n_clusters=H, random_state=0, batch_size=H+1).fit(x)
            print("Centres found ...")
            mu = kmeans.cluster_centers_
            labels = np.array(kmeans.labels_)
            sd = []
            for h in range(H):
                index = np.argwhere(labels == h)
                data = x[np.reshape(index, (index.shape[0],))]
                distance = (np.sum((data - mu[h]) ** 2, axis=1)) ** (0.50)
                maximum_distance = np.max(distance)
                if maximum_distance > 0:
                    sd.append(maximum_distance / float(3))
                else:
                    sd.append(sys.float_info.epsilon)
            sd = np.array(sd)

            Sd = torch.tensor(sd, dtype=torch.float32)
            Mu = torch.tensor(mu, dtype=torch.float32)

            Sd = Sd.to(device)
            Mu = Mu.to(device)

            print('Minibatch-Kmeans is Done!')
        except ValueError:
            divergent = True


    return Mu, Sd, divergent