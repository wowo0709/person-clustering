import os
import shutil

from matplotlib import pyplot as plt
import numpy as np

from . import calc as ic

pj = os.path.join


def plot_clusters(clusters, images, max_csize=None, mem_limit=1024**3):
    """Plot `clusters` of images in `images`.

    For interactive work, use :func:`visualize` instead.

    Parameters
    ----------
    clusters : see :func:`~imagecluster.calc.cluster`
    images : see :func:`~imagecluster.io.read_images`
    max_csize : int
        plot clusters with at most this many images
    mem_limit : float or int, bytes
        hard memory limit for the plot array (default: 1 GiB), increase if you
        have (i) enough memory, (ii) many clusters and/or (iii) large
        max(csize) and (iv) max_csize is large or None
    """
    assert len(clusters) > 0, "`clusters` is empty"
    stats = ic.cluster_stats(clusters)
    if max_csize is not None:
        stats = stats[stats[:,0] <= max_csize, :]
    # number of clusters
    ncols = stats[:,1].sum()
    # csize (number of images per cluster)
    nrows = stats[:,0].max()
    shape = images[list(images.keys())[0]].shape[:2]
    mem = nrows * shape[0] * ncols * shape[1] * 3
    if mem > mem_limit:
        raise Exception(f"size of plot array ({mem/1024**2} MiB) > mem_limit "
                        f"({mem_limit/1024**2} MiB)")
    # uint8 has range 0..255, perfect for images represented as integers, makes
    # rather big arrays possible
    arr = np.ones((nrows*shape[0], ncols*shape[1], 3), dtype=np.uint8) * 255
    icol = -1
    for csize in stats[:,0]:
        for cluster in clusters[csize]:
            icol += 1
            for irow, filename in enumerate(cluster):
                image = images[filename]
                arr[irow*shape[0]:(irow+1)*shape[0],
                    icol*shape[1]:(icol+1)*shape[1], :] = image
    print(f"plot array ({arr.dtype}) size: {arr.nbytes/1024**2} MiB")
    fig,ax = plt.subplots()
    ax.imshow(arr)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig,ax


def visualize(*args, **kwds):
    """Interactive wrapper of :func:`plot_clusters`. Just calls ``plt.show`` at
    the end. Doesn't return ``fig,ax``.
    """
    plot_clusters(*args, **kwds)
    plt.show()


def make_links(clusters, cluster_dr):
    """In `cluster_dr`, create nested dirs with symlinks to image files
    representing `clusters`.

    Parameters
    ----------
    clusters : see :func:`~imagecluster.calc.cluster`
    cluster_dr : str
        path
    """
    print("cluster dir: {}".format(cluster_dr))
    if os.path.exists(cluster_dr):
        shutil.rmtree(cluster_dr)
    for csize, group in clusters.items():
        for iclus, cluster in enumerate(group):
            dr = pj(cluster_dr,
                    'cluster_with_{}'.format(csize),
                    'cluster_{}'.format(iclus))
            for fn in cluster:
                link = pj(dr, os.path.basename(fn))
                os.makedirs(os.path.dirname(link), exist_ok=True)
                os.symlink(os.path.abspath(fn), link)


def merge_clusters(cluster_dict, fingerprints, FACE_THRESHOLD=0.18, CLOTH_THRESHOLD=0.12, iteration=1) -> dict:
    '''
    parameters:
        cluster_dict: calc.cluster() 의 return 값 (dict / key=cluster_size(int), value=clusters(2d-array))
        fingerprints: feature vector dictionary (key=filepath, value=feature vector)
        iteration: merge 반복 횟수
        FACE_THRESHOLD
        CLOTH_THRESHOLD
    return:
        merged_clusters: calc.cluster() 의 return 값과 동일한 형태 (dict / key=cluster_size(int), value=clusters(2d-array))
    '''

    for _ in range(iteration):
        cluster_list = sorted([[key, value] for key, value in cluster_dict.items()], key=lambda x:x[0], reverse=True)
        cluster_fingerprints = [] # [(face, cloth), ...]
        cluster_cnt = 0

        for cluster_with_num in cluster_list:
            num, clusters = cluster_with_num
            for idx, cluster in enumerate(clusters):
                cluster_face_fingerprint = np.zeros((128,))
                cluster_cloth_fingerprint = np.zeros((128,))
                i = 0
                for person in cluster:
                    encoding = fingerprints[person]
                    face, cloth = encoding[:128], encoding[128:]
                    cluster_face_fingerprint += face
                    cluster_cloth_fingerprint += cloth
                    i += 1
                assert i > 0, 'cluster is empty!'
                cluster_face_fingerprint /= i
                cluster_cloth_fingerprint /= i

                cluster_fingerprints.append([(num, idx), (cluster_face_fingerprint, cluster_cloth_fingerprint)])
                cluster_cnt += 1

        merged = []
        merged_clusters = dict()

        for i in range(cluster_cnt):
            if cluster_fingerprints[i][0] in merged:
                continue
            big_num, big_idx = cluster_fingerprints[i][0]
            person_list = cluster_dict[big_num][big_idx]
            merged_num = big_num
            for j in range(i+1, cluster_cnt):
                cluster_face_norm = round(np.linalg.norm(cluster_fingerprints[i][1][0] - cluster_fingerprints[j][1][0]),3)
                cluster_cloth_norm = round(np.linalg.norm(cluster_fingerprints[i][1][1] - cluster_fingerprints[j][1][1]),3)
                if cluster_face_norm < FACE_THRESHOLD or cluster_cloth_norm < CLOTH_THRESHOLD:
                    small_num, small_idx = cluster_fingerprints[j][0]
                    merged_num += small_num
                    person_list += cluster_dict[small_num][small_idx]
                    merged.append(cluster_fingerprints[j][0])
                    
            merged_clusters[merged_num] = merged_clusters.get(merged_num, [])
            merged_clusters[merged_num].append(person_list)

        cluster_dict = merged_clusters

    return merged_clusters