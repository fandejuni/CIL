# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 02:03:32 2019
"""

import numpy as np
from scipy.signal import convolve2d
from skimage.morphology import square
from scipy import ndimage
from heapq import heappush, heappop, heapify

def find_branch_points(skel):
    X = np.array([[1, 1, 1],
                   [1, 10, 1],
                   [1, 1, 1]], np.int)
    bp = convolve2d(skel.astype(np.int), X, mode="same")
    return (bp >= 13).astype(np.bool)

def find_end_points(skel):
    X = np.array([[1, 1, 1],
                   [1, 10, 1],
                   [1, 1, 1]], np.int)
    bp = convolve2d(skel.astype(np.int), X, mode="same")
    return (bp == 11).astype(np.bool)

def find_vertex_points(skel):
    X = np.array([[1, 1, 1],
                   [1, 10, 1],
                   [1, 1, 1]], np.int)
    vp = convolve2d(skel.astype(np.int), X, mode="same")
    return np.logical_or(vp==11, vp >= 13)
"""
def find_more_vertex_points(skel):
    vp = find_vertex_points(skel)
    vp[10:-10,10:-10] = find_vertex_points(skel[10:-10,10:-10])
    return vp
"""
def get_points(arr):
    return np.array(np.where(arr)).transpose()

class SkelGraph:
    
    def __init__(self, skel):
        self.h, self.w = skel.shape
        
        self.mask = skel.astype(np.bool).copy()
        
        self.vertex_labels = (find_vertex_points(skel)).astype(np.int)
        self.size = ndimage.label(self.vertex_labels, structure=square(3), output=self.vertex_labels)
        self.vertices = [[] for _ in range(self.size)]
        
        for y,x in get_points(self.vertex_labels):
            self.vertices[self.vertex_labels[y,x]-1].append((y,x))
            
        self.Adj = [[] for _ in range(self.size)]
        
        self.compute_graph()
    
    def get_neighbour_pixels(self, pixel):
        (y,x) = pixel
        for dy in (0,-1,1):
            for dx in (0,-1,1):
                if (dy == dx == 0):
                    continue
                if not ((0 <= y+dy < self.h) and (0 <= x+dx < self.w)):
                    continue
                if self.mask[(y+dy, x+dx)]:
                    yield (y+dy, x+dx)
    
    def compute_graph(self):
        
        edge_mask = np.logical_xor(self.mask, self.vertex_labels > 0)
        self.edge_labels, nr_edges = ndimage.label(edge_mask, structure=square(3))
        
        edges_inc = [[] for _ in range(nr_edges)]
        
        for u in range(self.size):
            for pixel in self.vertices[u]:
                for n_pixel in self.get_neighbour_pixels(pixel):
                    label = self.edge_labels[n_pixel]-1
                    if label >= 0:
                        edges_inc[label].append(u)
        
        for i in range(1,nr_edges+1):
            if len(edges_inc[i-1]) != 2:
                continue
            edge = (self.edge_labels == i)
            length = edge.sum()
            u,v  = edges_inc[i-1]
            self.Adj[u].append((v,i,length))
            self.Adj[v].append((u,i,length))
            
    def trim(self, threshold):
        candidates = [(self.Adj[u][0][2],u) for u in range(self.size) if len(self.Adj[u])==1]
        heapify(candidates)
        area = self.mask.sum()
        deleted_area = 0
        size = sum((1 for u in range(self.size) if len(self.Adj[u])>0))
        while candidates != []:
            if size <= 2:
                break
            if deleted_area >= 0.25*area:
                break
            d, u = heappop(candidates)
            if self.Adj[u] == []:
                continue
            v,i,l = self.Adj[u][0]
            if l < threshold:
                size -= 1
                deleted_area += l+1
                self.mask = np.logical_xor(self.mask, self.edge_labels == i)
                self.mask = np.logical_xor(self.mask, self.vertex_labels == u+1)
                self.Adj[u] = []
                for j in range(len(self.Adj[v])):
                    if self.Adj[v][j][1] == i:
                        self.Adj[v] = self.Adj[v][:j] + self.Adj[v][j+1:]
                        if len(self.Adj[v])==1:
                            heappush(candidates, (self.Adj[v][0][2],v))
                        break
                    
    def get_num_endpoints(self):
        return sum((1 for u in range(self.size) if len(self.Adj[u])==1))
    
    def get_longest_path(self):
        if self.size == 0:
            return 0, []
        u = 0
        while(self.Adj[u] == []):
            u += 1
        dist, _ = self.bfs(u)
        
        for i in range(self.size):
            if (dist[i] != float("inf")) and (dist[i] > dist[u]):
                u = i
                
        dist, pred = self.bfs(u)
        v = u
        for i in range(self.size):
            if (dist[i] != float("inf")) and (dist[i] > dist[v]):
                v = i
        d = dist[v]
        
        path = (self.vertex_labels == v+1).astype(np.bool)
        while pred[v][0] >= 0:
            u,i,l = pred[v]
            path += (self.edge_labels == i)
            path += (self.vertex_labels == u+1)
            v = u
        
        return d, path
        
    def bfs(self, u):
        dist = [float("inf") for _ in range(self.size)]
        pred = [(-1,0,0) for _ in range(self.size)]
        
        pred[u] = (-2,0,0)
        dist[u] = 0
        
        stack = [u]
        while stack != []:
            u = stack.pop()
            for v,i,l in self.Adj[u]:
                if pred[v][0] == -1:
                    pred[v]  = u,i,l
                    dist[v] = dist[u] + l + 1
                    stack.append(v)
        return dist, pred
    
    
    
