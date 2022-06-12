import os
from collections import OrderedDict

import numpy as np
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.models import Model
# import tensorflow.keras as keras
import torchvision.transforms as transforms
import torchvision.models as models
import torch

from PIL import Image

data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

pj = os.path.join


def get_model():

    '''pytorch - resnet50
    base_model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(
        *(list(base_model.children())[:6]), # conv3_x, (28, 28, 512)
        torch.nn.AdaptiveAvgPool2d((1,1)),
        torch.nn.Flatten()
    )
    '''
    
    '''pytorch - resnet18'''
    base_model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(
        *(list(base_model.children())[:6]), # extract from conv3_x block (28, 28, 128)
        torch.nn.AdaptiveAvgPool2d((1,1)),
        torch.nn.Flatten()
    )
    

    return model


def fingerprint(image, model):
    """Run image array (3d array) run through `model` (``keras.models.Model``).

    Parameters
    ----------
    image : 3d array
        (3,x,y) or (x,y,3), depending on
        ``keras.preprocessing.image.img_to_array`` and ``image_data_format``
        (``channels_{first,last}``) in ``~/.keras/keras.json``, see
        :func:`~imagecluster.io.read_images`
    model : ``keras.models.Model`` instance

    Returns
    -------
    fingerprint : 1d array
    """

    pil_image = Image.fromarray((image).astype(np.uint8))# .convert('RGB')# .resize((224,224), resample=3)
    
    arr3d = data_transform(pil_image)
    
    arr4d_tt = arr3d.unsqueeze(0)

    ret = model.forward(arr4d_tt)[0, :].detach().numpy()

    return ret


# Cannot use multiprocessing (only tensorflow backend tested, rumor has it that
# the TF computation graph is not built multiple times, i.e. pickling (what
# multiprocessing does with _worker) doen't play nice with Keras models which
# use Tf backend). The call to the parallel version of fingerprints() starts
# but seems to hang forever. However, Keras with Tensorflow backend runs
# multi-threaded (model.predict()), so we can sort of live with that. Even
# though Tensorflow has not the best scaling on the CPU, on low core counts
# (2-4), it won't matter that much. Also, TF was built to run on GPUs, not
# scale out multi-core CPUs.
#
##def _worker(image, model):
##    print(fn)
##    return fn, fingerprint(image, model)
##
##
##def fingerprints(images, model):
##    _f = functools.partial(_worker, model=model)
##    with mp.Pool(int(mp.cpu_count()/2)) as pool:
##        ret = pool.map(_f, images.items())
##    return dict(ret)

def fingerprints(images, model):
    """Calculate fingerprints for all image arrays in `images`.

    Parameters
    ----------
    images : see :func:`~imagecluster.io.read_images`
    model : see :func:`fingerprint`

    Returns
    -------
    fingerprints : dict
        {filename1: array([...]),
         filename2: array([...]),
         ...
         }
    """
    fingerprints = {}
    for fn,image in images.items():
        print(fn)
        fingerprints[fn] = fingerprint(image, model)
    return fingerprints


def pca(fingerprints, n_components=0.9, **kwds):
    """PCA of fingerprints for dimensionality reduction.

    Parameters
    ----------
    fingerprints : see :func:`fingerprints`
    n_components, kwds : passed to :class:`sklearn.decomposition.PCA`

    Returns
    -------
    dict
        same format as in :func:`fingerprints`, compressed fingerprints, so
        hopefully lower dim 1d arrays
    """
    if 'n_components' not in kwds.keys():
        kwds['n_components'] = n_components
    # Yes in recent Pythons, dicts are ordered in CPython, but still.
    _fingerprints = OrderedDict(fingerprints)
    X = np.array(list(_fingerprints.values()))
    Xp = PCA(**kwds).fit(X).transform(X)
    return {k:v for k,v in zip(_fingerprints.keys(), Xp)}


def cluster(fingerprints, sim=0.5, timestamps=None, alpha=0.3, method='average',
            metric='euclidean', extra_out=False, print_stats=True, min_csize=2):
    """Hierarchical clustering of images based on image fingerprints,
    optionally scaled by time distance (`alpha`).

    Parameters
    ----------
    fingerprints: dict
        output of :func:`fingerprints`
    sim : float 0..1
        similarity index
    timestamps: dict
        output of :func:`~imagecluster.io.read_timestamps`
    alpha : float
        mixing parameter of image content distance and time distance, ignored
        if timestamps is None
    method : see :func:`scipy.cluster.hierarchy.linkage`, all except 'centroid' produce
        pretty much the same result
    metric : see :func:`scipy.cluster.hierarchy.linkage`, make sure to use
        'euclidean' in case of method='centroid', 'median' or 'ward'
    extra_out : bool
        additionally return internal variables for debugging
    print_stats : bool
    min_csize : int
        return clusters with at least that many elements

    Returns
    -------
    clusters [, extra]
    clusters : dict
        We call a list of file names a "cluster".

        | keys = size of clusters (number of elements (images) `csize`)
        | value = list of clusters with that size

        ::

            {csize : [[filename, filename, ...],
                      [filename, filename, ...],
                      ...
                      ],
            csize : [...]}
    extra : dict
        if `extra_out` is True
    """
    assert 0 <= sim <= 1, "sim not 0..1"
    assert 0 <= alpha <= 1, "alpha not 0..1"
    assert min_csize >= 1, "min_csize must be >= 1"
    files = list(fingerprints.keys())
    # array(list(...)): 2d array
    #   [[... fingerprint of image1 (4096,) ...],
    #    [... fingerprint of image2 (4096,) ...],
    #    ...
    #    ]
    dfps = distance.pdist(np.array(list(fingerprints.values())), metric)
    if timestamps is not None:
        # Sanity error check as long as we don't have a single data struct to
        # keep fingerprints and timestamps, as well as image data. This is not
        # pretty, but at least a safety hook.
        set_files = set(files)
        set_tsfiles = set(timestamps.keys())
        set_diff = set_files.symmetric_difference(set_tsfiles)
        assert len(set_diff) == 0, (f"files in fingerprints and timestamps do "
                                    f"not match: diff={set_diff}")
        # use 'files' to make sure we have the same order as in 'fingerprints'
        tsarr = np.array([timestamps[k] for k in files])[:,None]
        dts = distance.pdist(tsarr, metric)
        dts = dts / dts.max()
        dfps = dfps / dfps.max()
        dfps = dfps * (1 - alpha) + dts * alpha
    # hierarchical/agglomerative clustering (Z = linkage matrix, construct
    # dendrogram), plot: scipy.cluster.hierarchy.dendrogram(Z)
    Z = hierarchy.linkage(dfps, method=method, metric=metric)
    # cut dendrogram, extract clusters
    # cut=[12,  3, 29, 14, 28, 27,...]: image i belongs to cluster cut[i]
    cut = hierarchy.fcluster(Z, t=dfps.max()*(1.0-sim), criterion='distance')
    cluster_dct = dict((iclus, []) for iclus in np.unique(cut))
    for iimg,iclus in enumerate(cut):
        cluster_dct[iclus].append(files[iimg])
    # group all clusters (cluster = list_of_files) of equal size together
    # {number_of_files1: [[list_of_files], [list_of_files],...],
    #  number_of_files2: [[list_of_files],...],
    # }
    clusters = {}
    for cluster in cluster_dct.values():
        csize = len(cluster)
        if csize >= min_csize:
            if not (csize in clusters.keys()):
                clusters[csize] = [cluster]
            else:
                clusters[csize].append(cluster)
    if print_stats:
        print_cluster_stats(clusters)
    if extra_out:
        extra = {'Z': Z, 'dfps': dfps, 'cluster_dct': cluster_dct, 'cut': cut}
        return clusters, extra
    else:
        return clusters


def cluster_stats(clusters):
    """Count clusters of different sizes.

    Returns
    -------
    2d array
        Array with column 1 = csize sorted (number of images in the cluster)
        and column 2 = cnum (number of clusters with that size).

        ::

            [[csize, cnum],
             [...],
            ]
    """
    return np.array([[k, len(clusters[k])] for k in
                     np.sort(list(clusters.keys()))], dtype=int)


def print_cluster_stats(clusters):
    """Print cluster stats.

    Parameters
    ----------
    clusters : see :func:`cluster`
    """
    print("#images : #clusters")
    stats = cluster_stats(clusters)
    for csize,cnum in stats:
        print(f"{csize} : {cnum}")
    if stats.shape[0] > 0:
        nimg = stats.prod(axis=1).sum()
    else:
        nimg = 0
    print("#images in clusters total: ", nimg)
