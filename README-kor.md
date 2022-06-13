# person-clustering
![header](https://capsule-render.vercel.app/api?type=waving&color=0:bee1e8,100:ddc7fc&height=250&section=header)

## Contents
- Introduction
- Related Works
- Project Description
- Contribute
- References


<br>

## Introduction
- Overall Service [[Github]](https://github.com/boostcampaitech3/final-project-level3-cv-10/tree/main)

`Naver 부스트캠프 AITech`에서 최종 프로젝트를 진행하며, 영상의 주요 등장 인물들을 사용자에게 제공하는 기능을 구현해야 했습니다.

이때 다음의 두 가지 조건을 충족해야 합니다. 

- 다양한 영상들에 대해 **좋은 성능**을 보여준다. 
- 다양한 영상에 적용하기 위해 **특정 인물을 대상으로 한 '학습 과정'을 없앤다.** (Unsupervised method)

위 두 가지 목표를 달성하기 위해 아래 두 가지 방법을 사용합니다. 

- Use **Face landmark + Clothing image** feature together
- Use **unsupervised clustering** method HAC (Hierarchical Agglomeratie Clustering)

<br>

즉, 해당 프로젝트는 한 영상 내에서 등장하는 인물들을 효과적으로 clustering 하기 위해 구현되었습니다. 

<br>

Face, Cloth, Face+Cloth를 이용한 clustering 성능을 정량 평가했을 때 Face+Cloth feature를 모두 사용했을 때 성능이 대폭 향상됨을 확인하였습니다. 

평가지표로는 NMI를 사용했으며, NMI에 대한 설명은 [here](https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf)를 참고해주세요. 

![image](https://user-images.githubusercontent.com/70505378/173325606-7787bd81-c2bf-4601-82f9-234f147956dd.png)

<br>

추가적으로 **cluster merging**, **min_csize deletion** 등의 후처리까지 적용하면 _'주요 등장 인물들에 대해'_ 0.9 이상의 NMI까지 기록합니다. 

<br>

## Related Works

**Clustering Method**
> Clustering 기법은 supervised method와   unsupervised method로 나눌 수 있다. <br><br> **Supervised clustering**은 그 성능이 더 뛰어나지만 학습 데이터를 확보해야 한다는 문제가 있고, **Unsupervised clustering** 은 학습 데이터를 사용하지 않아 일반적으로 적용할 수 있지만 성능이 비교적 낮고, 하이퍼파라미터에 민감하다는 문제가 있다. <br><br> 본 프로젝트에서는 일반적인 적용을 위해 unsupervised clustering 기법을 사용하되, 성능의 향상를 위해 얼굴과 옷의 두 가지 feature를 함께 사용한다. 
**Face feature extraction**
> 얼굴 구분을 위해 얼굴 feature 값을 추출할 때는 face landmark를 이용한다. 적게는 5 point(eyes, nose, corners of the mouth), 많게는 50 point 이상을 이용한다. <br><br> `dlib` 라이브러리를 이용하면 face detection 부터 feature extraction까지 쉽게 수행할 수 있다. 
**Image feature extraction**
> Image Clustering은 꾸준한 연구가 진행되고 있는 분야이다. 이미지 자체를 딥러닝 모델에 통과시켜 얻은 feature map을 사용하여 clustering 한다. <br><br> 최근 높은 성능의 clustering 기법들이 등장하고 있지만, 부분적으로 학습을 요구하는 경우가 많으며, 전체 이미지를 사용하기 때문에 person clustering은 어렵다. <br><br> reference: [[Papers with code]](https://paperswithcode.com/task/image-clustering)

<br>

작성자가 알기로 이전까지 **Face와 Clothing feature를 따로 추출하여 concatenate 한 뒤 clustering** 하는 방법을 시도한 github repository가 없어서, 이와 유사한 아이디어를 찾거나 해당 기능이 필요한 사람들이 이 프로젝트를 유용하게 사용할 수 있기를 바란다. 

<br>

## Project Description

대표적인 clustering task로 face clustering과 image clustering이 있다. Face clustering은 face landmark를 이용하고, image clustering은 image feature map을 이용한다. 

본 프로젝트에서는 성능의 향상을 위해 face landmark와 image feature map 정보를 동시에 활용한다. 

먼저 프레임 이미지에서 face detection을 수행하고, `dlib` 라이브러리를 이용하여 128 차원의 face landmark feature vector를 추출한다. 

동시에 face location으로부터 clothing location을 계산하고, clothing feature vector를 추출한다. 이때 feature extractor로는 `ResNet-18`을 사용하며, conv3_x block의 feature map을 추출하고 GAP(Global Average Pooling)을 적용하여 마찬가지로 128차원의 feature vector로 추출한다. 

추출된 두 128차원 feature vector를 크기가 1이 되도록 normalize 한 뒤 concat하여 한 인물에 대한 feature 값으로 사용한다. 

Clustering 기법은 여러 영상들에 대한 일반적인 적용을 위해 unsupervised method를 택한다. 

K-Means, HAC(Hierarchical Agglomerative Clustering), DBSCAN 등의 기법들 중 미리 cluster의 개수를 정하지 않아도 되고 가장 나은 성능을 보이는 HAC를 선택한다. 

추가적으로 clustering 결과에 cluster merging이나 10 미만의 크기를 갖는 cluster를 지우는 post-processing으로 더욱 성능을 끌어올렸다. 


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

- `clustering`: Clustering에 필요한 파일들
    - `calc.py`: clustering, feature extraction 등의 핵심 연산 함수들이 정의되어 있는 파일
    - `exceptions.py`: Clustering exception이 정의되어 있는 파일
    - `icio.py`: IO(Input/Output) 관련 함수들이 정의되어 있는 파일
    - `postproc.py`: Post-processing 함수들이 정의되어 있는 파일
- `outdated_files`: 과거 버전에서 사용된 파일들
- `demo.ipynb`: 데모 파일
- `face_extractor.py`: 메인 코드 파일 (frame extraction, face detection, cloth detection, feature extraction, etc.)

### Installation

`pip install -r requirements.txt`

### How to use
- `demo.ipynb` 파일 실행

**Face Extractor 인스턴스 생성**
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

**Face Extractor 실행** <br>
parameters
- face_cnt: 영상에서 추출할 얼굴 개수
- frame_batch_size: 한 번에 처리할 프레임 개수
- face_cloth_weights: face:cloth feature 비율
- use_scene_detection: True면 프레임 추출 시 scene detection 사용
- capture_interval: `use_scene_detection=False` 이면 영상에서 `capture_interval` 초마다 프레임 추출 (0 < )
- stop_sec: 0 초과의 수 지정 시 `stop_sec` 초까지만 영상 실행 후 중지
- skip_sec: 0 초과의 수 지정 시 영상의 초반 `skip_sec` 초 생략
- resizing_resolution: width가 `resizing_resolution` 이상인 영상 resize
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
- sim: similarity threshold (높일수록 엄격하게 clustering(더 유사해야 같은 cluster로 분류))
- min_csize: `min_csize` 미만의 크기를 가지는 클러스터는 삭제

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

- 기여하고 싶은 내용을 issue로 생성 (`assignees`, `labels` 지정)
- 제목은 아래 tag들 중 하나로 시작
    - `[DOCS]`: 문서 작업
    - `[DEV]`: 개발 내용 (기존과 기능은 동일하되, 성능에 차이가 발생할 수 있음)
    - `[FEAT]`: 새로운 기능 추가 및 요청
    - `[REFACT]`: 결과에는 영향을 주지 않는 코드 업데이트
    - `[EXT]`: 기타 등등

![image](https://user-images.githubusercontent.com/70505378/173353991-35f33c49-b984-4dde-8767-274b647b07f9.png)

- 만약 직접 코드 작성 후 commit하고 싶다면, 아래와 같이 [tag]/[issue_num] 이름으로 브랜치 생성 후 push
    - Ex. dev/13, feat/20, ext/27

![image](https://user-images.githubusercontent.com/70505378/173355030-7d59c543-3d57-49e6-b535-f54b22d78c0c.png)

**Commit convention**

- Issue 생성 시와 동일하게 아래 tag들 중 하나로 시작
    - 대응하는 issue가 있다면 동일한 tag 사용
 - 제목은 명확하고 이해할 수 있게 작성
     - Ex. [FEAT] Add scene transition code
 - push 할 때는 작업한 브랜치 그대로 push
     - Ex. `git push origin dev/13`

**Pull request**

- Pull request 생성 시 템플릿에 맞게 내용 작성

<br>

## References
- Unknown Face Classifier [[Github]](https://github.com/ukayzm/opencv/tree/master/unknown_face_classifier)
- Image Cluster [[Github]](https://github.com/elcorto/imagecluster)

<br>

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:bee1e8,100:ddc7fc&height=200&section=footer)