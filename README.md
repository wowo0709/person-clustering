# person-clustering

![image](https://user-images.githubusercontent.com/70505378/173368969-ec9a75b7-7bd0-4f41-86b2-be70b25c1806.png)

## Contents
- Introduction
- Related Works
- Project Description
- Contribute
- References


<br>

## Introduction
- Overall Service [[Github]](https://github.com/boostcampaitech3/final-project-level3-cv-10/tree/main)

When working on the final project at 'Naver Boost Camp AITech', I had to implement the function of providing users with key characters in the video.

The following two conditions must be met:

- It have to show **good performance** for various videos. 
- To be applied to various videos, the **'training process' is eliminated.** (Unsupervised method)

To meet the above two conditions, I used the following two methods.

- Use **Face landmark + Clothing image** feature together
- Use **unsupervised clustering** method HAC (Hierarchical Agglomeratie Clustering)

<br>

In other words, the project was implemented to effectively cluster the characters that appear in one video.

<br>

When the clustering performance using Face, Cloth, and Face+Cloth was quantified, it was confirmed that the performance was greatly improved when all Face+Cloth features were used.

NMI was used as a metric, and for an explanation of NMI, see [here](https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf).

![image](https://user-images.githubusercontent.com/70505378/173325606-7787bd81-c2bf-4601-82f9-234f147956dd.png)

<br>

Additionally, if post-processing such as **cluster merging** or **min_csize deletion** applied, clustering shows **up to 0.9 or higher NMI** for **_key characters_**.

<br>

## Related Works

**Clustering Method**
> Clustering techniques can be divided into a supervised method and an unsupervised method. <br><br> **Supervised clustering** has a better performance, but has the problem of securing learning data, and **Unsupervised clustering** is generally applicable because it does not use learning data, but has a relatively low performance and is sensitive to hyperparameters. <br><br> In this project, unsupervised clustering technique is used for general application, but combined two features, face and clothes, to improve performance.

**Face feature extraction**
> Face landmark is used to extract face feature values for face distinction. It uses at least 5 points (eyes, nose, corners of the mouth), and at most more than 50 points. <br><br> The 'dlib' library makes it easy to perform from face detection to feature extraction.

**Image feature extraction**
> Image clustering is an area where research is being conducted steadily. The image itself is clustered using the feature map obtained by passing through the deep learning model. <br><br> Although high-performance clustering techniques have recently emerged, person clustering is difficult because it often requires training in part and uses the entire image. <br><br> reference: [[Papers with code]](https://paperswithcode.com/task/image-clustering)


<br>

As far as I know, there is no github repository that has previously attempted to extract **Face and Clothing features separately, concatenate them, and then clustering**. So it is hoped that people who find similar ideas or need them can use the project well.

<br>

## Project Description

Typical clustering tasks include **face clustering** and **image clustering**. Face clustering uses face landmark, and image clustering uses image feature map.

In this project, **face landmark and image feature map information are used simultaneously to improve performance.**

<br>

First, **face detection** is performed on the frame image, and a **128-d face landmark feature vector** is extracted using the 'dlib' library.

At the same time, the **clothing location** is calculated from the face location and the **128-d clothing feature vector** is extracted. In this case, 'ResNet-18' is used as the feature extractor, and the feature map of `conv3_x block` is extracted and GAP (Global Average Pooling) is applied to extract a 128-d feature vector.

Next, the extracted two 128-dimensional feature vectors are **normalized** to a size of 1, and then **concatenated** and used as feature values for one character.

<br>

Here, I used unsupervised clustering method for general application to various videos.

Among techniques such as K-Means, HAC (Hierarchical Agglomerative Clustering), and DBSCAN, HAC is selected because the number of clusters doesn't need to be determined in advance and it shows the best performance.

In addition, the clustering results were further enhanced by cluster merging or post-processing that eliminated clusters with sizes less than 10.


### Project Structure

```bash
├── clustering
│   ├── calc.py
│   ├── exceptions.py
│   ├── icio.py
│   └── postproc.py
│
├── outdated_files
│   ├── face_classifier.py
│   └── person_clustering.ipynb
│
├── demo.ipynb
└── face_extractor.py
``` 

- `clustering`: Files Required for Clustering
    - `calc.py`: clustering, feature extraction 등의 핵심 연산 함수들이 정의되어 있는 파일
    - `exceptions.py`: File with key operational functions such as clustering, feature extraction, etc. defined.
    - `icio.py`: File with IO (Input/Output) related functions defined
    - `postproc.py`: File with Post-processing functions defined
- `outdated_files`: Files used in previous versions
- `demo.ipynb`: Demo file
- `face_extractor.py`: Main code file (frame extraction, face detection, cloth detection, feature extraction, etc.)

### Installation

`pip install -r requirements.txt`

### How to use
- run `demo.ipynb` 

**Create `Face Extractor` Instance**
```python
# Create FaceExtractor instance
video_num = 0
video_path = video_paths[video_num]
result_dir = video_path[:-4] + '_result'

face_extractor = FaceExtractor(
    video_path = video_path,
    result_dir = result_dir,
)
```

**Run `Face Extractor`** <br>

parameters
- face_cnt: Number of faces to be extracted from the image
- frame_batch_size: Number of frames to process at a time
- face_cloth_weights: Ratio of face feature:cloth feature 
- use_scene_detection: If True, use scene detection when extracting frames
- capture_interval: If `use_scene_detection=False`, extract frames from the video every `capture_interval` seconds (0 < )
- stop_sec: If > 0, the video stops after `stop_sec` seconds
- skip_sec: If > 0, skip 'skip_sec' seconds at the beginning of the video
- resizing_resolution: Resize video with width greater than `resizing_resolution`
- resizing_ratio: resize ratio ([0, 1])
```python
# Run face extractor
fingerprints = face_extractor.run(
    face_cnt=350,
    frame_batch_size=16,
    face_cloth_weights=[1.0, 1.0],
    use_scene_transition=False,
    capture_interval_sec=3,
    stop_sec=0,
    skip_sec=0,
    resizing_resolution=1000,
    resizing_ratio=None
)
```

**Clustering & Print result**

parameters
- sim: similarity threshold (the higher, the stricter)
- min_csize: Eliminate clusters with sizes less than `min_csize`

```python
# Print clustering result
clusters = calc.cluster(fingerprints, sim=0.63, min_csize=6) # the higher, the stricter

postproc.make_links(clusters, os.path.join(result_dir, 'imagecluster/clusters'))

images = icio.read_images(result_dir, size=(224,224))

fig, ax = postproc.plot_clusters(clusters, images)
fig.savefig(os.path.join(result_dir, 'imagecluster/_cluster.png'))
postproc.plt.show()
```

![image](https://user-images.githubusercontent.com/70505378/173352906-19afbf5c-c5d3-43e1-a65b-7758046826da.png)

**Cluster merging & Print result**

```python
# After merging
FACE_THRESHOLD_HARD = 0.18
CLOTH_THRESHOLD_HARD = 0.12
merged_clusters = postproc.merge_clusters(clusters, fingerprints, FACE_THRESHOLD_HARD, CLOTH_THRESHOLD_HARD, iteration=1)

postproc.make_links(merged_clusters, os.path.join(result_dir, 'imagecluster/merged_clusters'))

images = icio.read_images(result_dir, size=(224,224))

fig, ax = postproc.plot_clusters(merged_clusters, images)
fig.savefig(os.path.join(result_dir, 'imagecluster/_merged_cluster.png'))
postproc.plt.show()
```

![image](https://user-images.githubusercontent.com/70505378/173352971-4b48f81f-3c22-422f-86c7-f005cc8f62ac.png)


<br>

## Contribute
- how to contribute?

**Make issue & Create new branch**

- Create issue that contains the content you want to contribute (`assignees`, `labels` 지정)
- Issue title have to start with one of the tags below
    - `[DOCS]`: Working with documents
    - `[DEV]`: Development (Same function as before, but performance may differ)
    - `[FEAT]`: Add or Request new feature
    - `[REFACT]`: Update code that does not affect results/performance
    - `[EXT]`: Etc. 

![image](https://user-images.githubusercontent.com/70505378/173353991-35f33c49-b984-4dde-8767-274b647b07f9.png)

- If you want to commit after writing the code yourself, create a branch with the name [tag]/[issue_num] as shown below and push
    - Ex. dev/13, feat/20, ext/27

![image](https://user-images.githubusercontent.com/70505378/173355030-7d59c543-3d57-49e6-b535-f54b22d78c0c.png)

**Commit convention**

- Start with one of the tags below, same as when creating an Issue
    - Use the same tag if you have a corresponding issue
 - Make the title clear and understandable
     - Ex. [FEAT] Add scene transition code
 - When you push, push the branch you worked on
     - Ex. `git push origin dev/13`

**Pull request**

- Create content according to template when creating pull request

<br>

## References
- Unknown Face Classifier [[Github]](https://github.com/ukayzm/opencv/tree/master/unknown_face_classifier)
- Image Cluster [[Github]](https://github.com/elcorto/imagecluster)

<br>

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:bee1e8,100:ddc7fc&height=200&section=footer)
