import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import defaultdict
#import seaborn as sb
from utils import customSoftmax
import faiss
import torch.multiprocessing as mp

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, labels = -1,
        normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """

        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        # Hardcoded shadow labels
        self.shadowLabels = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        self.noOfInformative = 1
        self.noOfShadow = 4
        self.dots = 0
        #self.pool = multiprocessing.Pool()
        mp.set_start_method('spawn', force=True)

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            self.dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None

        else:
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
            )

    def parallelProcessingFunc(self, labels, x):
        labels_flatten = labels #.flatten().cpu().numpy()
        x_numpy = np.ascontiguousarray(np.transpose(x))
        numpyCentroids = self.centroids.detach().cpu().numpy()
        #res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(self.dim) #faiss.GpuIndexFlatL2(res, self.dim)
        index.add(numpyCentroids)
        _, clusterAssignment = index.search(x_numpy, 1)
        clusters = defaultdict(list)

        for label, point, clusterId in zip(labels_flatten, x_numpy, clusterAssignment):
            clusters[clusterId[0]].append((point, label))

        for l in range(self.num_clusters):
            if l not in clusters.keys():
                clusters[l] = []

        for clusterId, elements in clusters.items():
            # Collect the informative and shadow centroids in the cluster
            informativeInCluster = []
            shadowInCluster = []
            for point, label in elements:
                if label in self.shadowLabels:
                    shadowInCluster.append(point)
                else:
                    informativeInCluster.append(point)
       
            # Adjustments as described above
            if not informativeInCluster:
                informativeInCluster.append(np.zeros(512))
            if not shadowInCluster:
                shadowInCluster.append(np.zeros(512))

                    # Finding the centroids"
            informativeCentroids = np.array(informativeInCluster)
            if len(informativeInCluster) > self.noOfInformative:
                nbrs = NearestNeighbors(n_neighbors=self.noOfInformative, algorithm='ball_tree').fit(informativeInCluster)
                _, indices = nbrs.kneighbors(numpyCentroids[clusterId].reshape(1, -1))
                informativeCentroids = informativeCentroids[list(indices[0])]

            shadowCentroids = np.array(shadowInCluster)
            if len(shadowInCluster) > self.noOfShadow:
                nbrs = NearestNeighbors(n_neighbors=self.noOfShadow, algorithm='ball_tree').fit(shadowInCluster)
                _, indices = nbrs.kneighbors(numpyCentroids[clusterId].reshape(1, -1))
                shadowCentroids = shadowCentroids[list(indices[0])]

            clusters[clusterId] = [shadowCentroids, informativeCentroids]

        semanticWeights = np.zeros((x_numpy.shape[0], self.num_clusters))
        for p, input in enumerate(x_numpy):
            weight = np.zeros(self.num_clusters)
            for t in range(self.num_clusters):
                if len(clusters[t]) == 2:
                    weight[t] = customSoftmax(input, clusters[t][0], clusters[t][1])
                else:
                    weight[t] = 0.1 #arbitrarily entered 0.1 here
                semanticWeights[p] =  weight # weight
        return np.transpose(semanticWeights)

    def forward(self, x, y = np.empty(1)):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x_flatten = x.view(N, C, -1)

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        weights = torch.ones(soft_assign.shape)

        if np.any(y):
            # Convert to numpy versions for computation
            '''
            for i, labels in enumerate(y):
                labels_flatten = labels #.flatten().cpu().numpy()
                x_numpy = np.ascontiguousarray(x_flatten[i].squeeze().cpu().detach().numpy().T)
                numpyCentroids = self.centroids.detach().cpu().numpy()
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatL2(res, self.dim)
                index.add(numpyCentroids)
                _, clusterAssignment = index.search(x_numpy, 1)
                clusters = defaultdict(list)
                for label, point, clusterId in zip(labels_flatten, x_numpy, clusterAssignment):
                    clusters[clusterId[0]].append((point, label))

                for l in range(self.num_clusters):
                    if l not in clusters.keys():
                        clusters[l] = []

                for clusterId, elements in clusters.items():
                # Collect the informative and shadow centroids in the cluster
                    informativeInCluster = []
                    shadowInCluster = []
                    for point, label in elements:
                        if label in self.shadowLabels:
                            shadowInCluster.append(point)
                        else:
                            informativeInCluster.append(point)

                    # Adjustments as described above
                    if not informativeInCluster:
                        informativeInCluster.append(np.zeros(512))
                    if not shadowInCluster:
                        shadowInCluster.append(np.zeros(512))

                    # Finding the centroids"
                    informativeCentroids = np.array(informativeInCluster)
                    if len(informativeInCluster) > self.noOfInformative:
                        nbrs = NearestNeighbors(n_neighbors=self.noOfInformative, algorithm='ball_tree').fit(informativeInCluster)
                        _, indices = nbrs.kneighbors(numpyCentroids[clusterId].reshape(1, -1))
                        informativeCentroids = informativeCentroids[list(indices[0])]

                    shadowCentroids = np.array(shadowInCluster)
                    if len(shadowInCluster) > self.noOfShadow:
                        nbrs = NearestNeighbors(n_neighbors=self.noOfShadow, algorithm='ball_tree').fit(shadowInCluster)
                        _, indices = nbrs.kneighbors(numpyCentroids[clusterId].reshape(1, -1))
                        shadowCentroids = shadowCentroids[list(indices[0])]

                    clusters[clusterId] = [shadowCentroids, informativeCentroids]

                semanticWeights = np.zeros((x_numpy.shape[0], self.num_clusters))
                for p, input in enumerate(x_numpy):
                    weight = np.zeros(self.num_clusters)
                    for t in range(self.num_clusters):
                        if len(clusters[t]) == 2:
                            weight[t] = customSoftmax(input, clusters[t][0], clusters[t][1])
                        else:
                            weight[t] = 0.1 #arbitrarily entered 0.1 here
                    semanticWeights[p] =  weight # weight

                attentionWeights = torch.tensor(semanticWeights).T
                weights[i] = attentionWeights #.T # purposeful error- you are not supposed to transpose here
            '''
            pool = mp.Pool()
            poolInput = [(label, x) for label, x in zip(y, x_flatten.squeeze().cpu().detach().numpy())]

            weights = pool.starmap(self.parallelProcessingFunc, poolInput)
            weights = torch.from_numpy(np.array(weights))
            del poolInput
            weights = weights.to(x.device)

        # Code used to test the attention map
        # imgTest = cv2.imread("/content/002998_pitch1_yaw12.jpg")
        # fig, ax = plt.subplots(figsize=(16,12))
        # sb.heatmap(semanticWeights,ax=ax, alpha = 0.5, zorder = 2,  cmap="viridis", cbar = True)
        # ax.imshow(imgTest, aspect = ax.get_aspect(),
        #           extent = ax.get_xlim() + ax.get_ylim(),
        #           zorder = 1)
        # ax.axis('off')

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2) * weights[:,C:C+1,:].unsqueeze(2)


        vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad
