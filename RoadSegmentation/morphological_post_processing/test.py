import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import medial_axis, square, disk, binary_dilation
from skimage.filters.rank import gradient
from skimage.filters import gaussian
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema

def get_points(arr):
    return np.array(np.where(arr)).transpose()

def minimal_skeleton(skel, dist_trans, min_distance = 20):
    h,w = skel.shape
    min_skel = skel.copy()
    peaks = peak_local_max(dist_trans, min_distance=min_distance)

    peaksy, peaksx = peaks.transpose()
    min_skel[peaksy, peaksx] = 0
    min_skel[peaksy, peaksx] = 0
    
    edges, num_edges = ndimage.label(min_skel, square(3))
    degree = [0 for _ in range(num_edges+1)]
    
    for y, x in peaks:
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not (0 <= y+dy < h) and (0 <= x+dx < w):
                    continue
                l = edges[y+dy,x+dx]
                if l > 0:
                    degree[l] += 1
                    for dy2 in (-1,0,1):
                        for dx2 in (-1,0,1):
                            if not (0 <= y+dy+dy2 < h) and (0 <= x+dx+dx2 < w):
                                continue
                            if edges[y+dy+dy2,x+dx+dx2] == l:
                                edges[y+dy+dy2,x+dx+dx2] = 0              
    for i in range(1, num_edges+1):
        if degree[i] < 2:
            min_skel[edges == i] = 0
    min_skel[peaksy, peaksx] = 1
    
    return min_skel

def breaking_segments(skel, dist_trans, threshold = 0.5):
    yskel, xskel = np.where(skel)
    distances = dist_trans[yskel,xskel]
    min_dist, max_dist = distances.min(), distances.max()
    
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(distances.reshape(-1,1))
    s = np.linspace(min_dist, max_dist)
    e = kde.score_samples(s.reshape(-1,1))
    
    
    plt.figure(figsize=(6,6))
    plt.hist(distances, bins = 50)
    plt.xlabel('Distance to border in pixels')
    plt.ylabel('Number of pixels')
    plt.title('Histogram of distances to border')
    plt.show()
    
    plt.figure(figsize=(6,6))
    plt.plot(s, e)
    plt.xlabel('Distance to border in pixels')
    plt.ylabel('Density')
    plt.title('Estimated density of distances to border')
    plt.show()
    
    mi = argrelextrema(e, np.less)[0]
    
    if len(mi) < 1:
        return np.zeros(skel.shape)
    
    overlap = np.zeros(skel.shape, np.int)
    overlap += binary_dilation((dist_trans < s[mi[0]]), square(3))
    for i in range(1,len(mi)):
        overlap += binary_dilation((s[mi[i-1]] < dist_trans)*(s[mi[i]] > dist_trans), square(3))
    overlap += binary_dilation((dist_trans > s[mi[-1]]), square(3))
    
    return skel*(overlap > 1)
    
    smooth_dist_trans = gaussian(dist_trans, sigma=10)
    grad = gradient(smooth_dist_trans/smooth_dist_trans.max(), disk(3))
    grad = (skel*grad).astype(np.float32)/(smooth_dist_trans+1)
    return (grad > 0.5)
    
def reconstruct_from_skel(skel, dist_trans, num_labels = 1, smooth=0.2):
    
    mean_widths = [0 for _ in range(num_labels)]
    if smooth > 0:
        for label in range(1,num_labels+1):
            seg = (skel==label).astype(np.float32)
            mean_widths[label-1] = (seg*dist_trans).sum()/seg.sum()
    
    h, w = skel.shape
    reconstr = np.zeros(skel.shape, np.int)
    for y,x in get_points(skel):
        label = skel[y,x]
        d = int(smooth*mean_widths[label-1]+(1-smooth)*dist_trans[y,x]+0.5)
        if d<0:
            continue
        left_cut = max(0,d-x)
        right_cut = max(0,x+d-w+1)
        up_cut = max(0,d-y)
        down_cut = max(0,y+d-h+1)
        ball = disk(d)[up_cut:2*d+1-down_cut, left_cut:2*d+1-right_cut]*label
        reconstr[max(0,y-d):y+d+1,max(0,x-d):x+d+1] = np.maximum(ball,reconstr[max(0,y-d):y+d+1,max(0,x-d):x+d+1])
    return reconstr

def skeletal_dismembering(mask):
    
    plt.figure(figsize=(12,12))
    plt.imshow(mask)
    plt.show()
    
    skel, dist_trans = medial_axis(mask, return_distance=True)
    
    plt.figure(figsize=(12,12))
    plt.imshow(skel)
    plt.show()
    
    plt.figure(figsize=(12,12))
    plt.imshow(dist_trans)
    plt.show()
    
    min_skel = minimal_skeleton(skel, dist_trans)
    
    plt.figure(figsize=(12,12))
    plt.imshow(min_skel)
    plt.show()
    
    breaks = breaking_segments(min_skel, dist_trans)
    
    min_skel[breaks] = 0
    min_skel_parts, _ = ndimage.label(min_skel, square(3))
    
    plt.figure(figsize=(12,12))
    plt.imshow(min_skel_parts)
    plt.show()
    
    skel[breaks] = 0
    skel_parts, num_skel_parts = ndimage.label(skel, square(3))
    
    plt.figure(figsize=(12,12))
    plt.imshow(skel_parts)
    plt.show()
    
    reconstr = reconstruct_from_skel(skel_parts, dist_trans, num_skel_parts, smooth=0)
    
    plt.figure(figsize=(12,12))
    plt.imshow(reconstr)
    plt.show()
    
    min_skel = min_skel*skel
    return reconstr

def main():
    img = mpimg.imread("test.png")
    mask = (img > 0.5)[:,:,0]
    skeletal_dismembering(mask)
    
if __name__ == "__main__":
    main()