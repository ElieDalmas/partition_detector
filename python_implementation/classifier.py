import cv2
import numpy as np
from neighbors import KNeighborsClassifier

# level 1 categories
NOTEHEAD_CATS = {
    'noteheadBlackOnLine', 'noteheadBlackInSpace',
    'noteheadWholeOnLine', 'noteheadWholeInSpace',
    'noteheadHalfOnLine',  'noteheadHalfInSpace',
}
REST_CATS = {
    'restWhole', 'restHalf', 'restQuarter',
    'rest8th', 'rest16th', 'rest32nd',
}
CLEF_CATS = {'clefG', 'clefF', 'clefC'}

# level 2 notehead subtypes
NOTEHEAD_BLACK = {'noteheadBlackOnLine', 'noteheadBlackInSpace'}
NOTEHEAD_HALF  = {'noteheadHalfOnLine',  'noteheadHalfInSpace'}
NOTEHEAD_WHOLE = {'noteheadWholeOnLine', 'noteheadWholeInSpace'}


def level1_label(cat_name: str) -> str:
    if cat_name in NOTEHEAD_CATS:
        return 'notehead'
    if cat_name in REST_CATS:
        return 'rest'
    if cat_name in CLEF_CATS:
        return 'clef'
    return 'other'


def level2_label(cat_name: str) -> str | None:
    if cat_name in NOTEHEAD_BLACK:
        return 'black'
    if cat_name in NOTEHEAD_HALF:
        return 'half'
    if cat_name in NOTEHEAD_WHOLE:
        return 'whole'
    return None


def extract_features(blob: np.ndarray) -> np.ndarray:
    blob_resized = cv2.resize(blob, (32, 32))

    moments = cv2.moments(blob_resized)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    h, w = blob.shape
    density = np.sum(blob > 0) / (h * w + 1e-6)
    aspect = w / (h + 1e-6)

    ch, cw = h // 4, w // 4
    center = blob[h//2 - ch:h//2 + ch, w//2 - cw:w//2 + cw]
    center_density = np.sum(center > 0) / (center.size + 1e-6) if center.size > 0 else 0

    border = blob_resized.copy()
    inner = np.zeros_like(border)
    cv2.ellipse(inner, (16, 16), (8, 6), 0, 0, 360, 255, -1)
    border[inner > 0] = 0
    ring_ratio = np.sum(border > 0) / (np.sum(blob_resized > 0) + 1e-6)

    return np.concatenate([hu, [density, aspect, center_density, ring_ratio]])

def build_training_data(anns, id_to_name, img_bin):
    """
    Build (X, y_l1, y_l2) from ground truth annotations.
    y_l2 is None for non-notehead samples.
    """
    X, y_l1, y_l2 = [], [], []

    for ann in anns:
        cat = id_to_name.get(ann['cat_id'][0], '?')
        l1 = level1_label(cat)
        l2 = level2_label(cat)

        x1, y1, x2, y2 = [int(v) for v in ann['a_bbox']]
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4 or w > 200 or h > 200:
            continue

        blob = img_bin[y1:y2, x1:x2]
        if blob.size == 0:
            continue

        X.append(extract_features(blob))
        y_l1.append(l1)
        y_l2.append(l2)

    return np.array(X), y_l1, y_l2


class SymbolClassifier:
    """
    Two-level kNN classifier for music symbols.
    Level 1: notehead / rest / clef / other
    Level 2: black / half / whole  (noteheads only)
    """

    def __init__(self, k: int = 5):
        self.knn_l1 = KNeighborsClassifier(n_neighbors=k)
        self.knn_l2 = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X, y_l1, y_l2):
        self.knn_l1.fit(X, y_l1)

        # train level 2 only on notehead samples
        idx = [i for i, l in enumerate(y_l1) if l == 'notehead']
        X_nh  = X[idx]
        y_nh  = [y_l2[i] for i in idx]
        self.knn_l2.fit(X_nh, y_nh)

    def predict(self, X) -> list[tuple[str, str | None]]:
        """Returns list of (level1, level2) pairs."""
        l1_preds = self.knn_l1.predict(X)
        results = []
        for i, l1 in enumerate(l1_preds):
            if l1 == 'notehead':
                l2 = self.knn_l2.predict([X[i]])[0]
            else:
                l2 = None
            results.append((l1, l2))
        return results
