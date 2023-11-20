from typing import Literal, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from skimage.feature import SIFT, hog
from skimage.transform import resize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from tqdm import tqdm


def load_preprocess_nii(
    img: str,
    resize_shape: Tuple[int, int, int] = (256, 256, 24),
    slice_trunc: Optional[Tuple[int, int]] = (2, 10),  # top, bottom
) -> np.ndarray:
    img = nib.load(img).get_fdata()
    img = (img - img.min()) / img.max()
    img = resize(img, resize_shape)
    if slice_trunc is not None:
        img = img[..., slice_trunc[1]:(-slice_trunc[0])]

    return img


def load_preprocess_niis(
    imgs: Sequence[str],
    resize_shape: Tuple[int, int, int] = (256, 256, 24),
    slice_trunc: Optional[Tuple[int, int]] = (2, 10),  # top, bottom
) -> np.ndarray:
    return np.stack(
        [
            load_preprocess_nii(fn, resize_shape, slice_trunc)
            for fn in tqdm(imgs, desc="Load Images: ")
        ],
        axis=0,
    )


def hog_feature(img_pp: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [hog(img_pp[..., i]) for i in range(img_pp.shape[-1])]
    )


def sift_desc(img: np.ndarray) -> np.ndarray:
    detector = SIFT()
    try:
        detector.detect_and_extract(img)
    except RuntimeError:
        return None
    return detector.descriptors


class NiiHOG(BaseEstimator, TransformerMixin):
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        return self

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        assert X.ndim == 4
        return np.stack([hog_feature(X[i]) for i in range(X.shape[0])], axis=0)


class NiiSIFT(BaseEstimator, TransformerMixin):
    def __init__(
        self, nclusters: int = 20, random_state: Optional[int] = 0
    ) -> None:
        super().__init__()
        self._nclusters = nclusters
        self._kmeans_objs = None
        self._random_state = random_state

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        assert X.ndim == 4
        self._kmeans_objs = []
        for j in range(X.shape[-1]):
            descs = []
            for i in range(X.shape[0]):
                desc_ij = sift_desc(X[i, ..., j])
                if desc_ij is not None:
                    descs.append(desc_ij)
            descs = np.concatenate(descs, axis=0)
            self._kmeans_objs.append(
                MiniBatchKMeans(
                    self._nclusters,
                    n_init="auto",
                    random_state=self._random_state,
                ).fit(descs)
            )
        return self

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        assert X.ndim == 4
        res = np.zeros((X.shape[0], X.shape[-1], self._nclusters))
        for j in range(X.shape[-1]):
            for i in range(X.shape[0]):
                desc_ij = sift_desc(X[i, ..., j])
                if desc_ij is not None:
                    pred_ij = self._kmeans_objs[j].predict(desc_ij)
                    uni_ij, cnt_ij = np.unique(pred_ij, return_counts=True)
                    res[i, j, uni_ij] += cnt_ij / pred_ij.shape[0]
        return res.reshape(res.shape[0], -1)

    def fit_transform(self, X: np.ndarray, y=None, **fit_params) -> np.ndarray:
        assert X.ndim == 4
        self._kmeans_objs = []
        res = np.zeros((X.shape[0], X.shape[-1], self._nclusters))
        for j in range(X.shape[-1]):
            descs, nkp = [], []
            for i in range(X.shape[0]):
                desc_ij = sift_desc(X[i, ..., j])
                if desc_ij is not None:
                    descs.append(desc_ij)
                    nkp.append(desc_ij.shape[0])
                else:
                    nkp.append(0)
            descs = np.concatenate(descs, axis=0)
            km_obj = MiniBatchKMeans(
                self._nclusters, n_init="auto", random_state=self._random_state
            )
            pred_j = km_obj.fit_predict(descs)
            self._kmeans_objs.append(km_obj)
            pred_j = np.split(pred_j, np.cumsum(nkp)[:-1])
            for i, (nkp, pred_ij) in enumerate(zip(nkp, pred_j)):
                if nkp > 0:
                    uni_ij, cnt_ij = np.unique(pred_ij, return_counts=True)
                    res[i, j, uni_ij] += cnt_ij / nkp
        return res.reshape(res.shape[0], -1)


def get_tradition_model(predictor: Literal["svc", "rf"] = "svc") -> Pipeline:
    assert predictor in ["svc", "rf"]
    return Pipeline(
        [
            (
                "SIFTandHOG",
                FeatureUnion([("SIFT", NiiSIFT()), ("HOG", NiiHOG())]),
            ),
            ("PCA", PCA(n_components=100)),
            (
                "Predictor",
                SVC(probability=True)
                if predictor == "svc"
                else RandomForestClassifier(),
            ),
        ],
        verbose=True,
    )


# root = "/mnt/data1/tiantan/pp_SynthSeg/rm_skull/MS/"
# imgs = [
#     load_preprocess_nii(osp.join(root, fn)) for fn in os.listdir(root)[10:20]
# ]
# imgs = np.stack(imgs, axis=0)
# feature = NiiHOG().fit_transform(imgs)
# __import__('ipdb').set_trace()
# sift = NiiSIFT(20)
# features1 = sift.fit_transform(imgs)
# features2 = NiiSIFT(20).fit(imgs).transform(imgs)
