"""Classical copy-move detection: SIFT keypoints -> 2NN ratio match
(excluding spatially-near matches) -> hierarchical clustering of matched
pairs -> RANSAC affine fit per cluster -> dense region growing.

This is the path recommended to implement first in the blueprint: it is
explainable (the affine transform IS the evidence) and it is the only
branch that works when a region has been duplicated within the same
image, since duplicate regions share identical sensor/noise statistics
and are invisible to the noise-based deep branch.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


@dataclass
class CopyMoveMatch:
    cluster_id: int
    src_points: list[tuple[float, float]]
    dst_points: list[tuple[float, float]]
    transform: list[list[float]]   # 2x3 affine
    inlier_count: int
    src_bbox: tuple[int, int, int, int]
    dst_bbox: tuple[int, int, int, int]


@dataclass
class CopyMoveReport:
    detected: bool
    confidence: float
    matches: list[CopyMoveMatch] = field(default_factory=list)
    mask: np.ndarray | None = None
    notes: str = ""


def _bbox_of(points: np.ndarray) -> tuple[int, int, int, int]:
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    x1, y1 = points[:, 0].max(), points[:, 1].max()
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def detect_copy_move(
    image_bgr: np.ndarray,
    min_distance_px: float = 40.0,
    ratio_thresh: float = 0.5,
    cluster_dist_thresh: float = 60.0,
    min_cluster_size: int = 4,
    min_region_area: float = 400.0,
    max_clusters: int = 6,
    ransac_reproj_thresh: float = 3.0,
) -> CopyMoveReport:
    """Guardrails (min_region_area, max_clusters, geometric-consistency
    via RANSAC) exist specifically so this does not fire on legitimately
    repetitive textures (brick walls, crowds, tiled floors) — see the
    "nasty edge case" callout in the blueprint.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.03)
    kps, desc = sift.detectAndCompute(gray, None)

    if desc is None or len(kps) < 20:
        return CopyMoveReport(detected=False, confidence=0.0,
                               notes="Insufficient keypoints for self-matching.")

    bf = cv2.BFMatcher(cv2.NORM_L2)
    # Self-match: k=3 because the first neighbor of any point is itself.
    knn = bf.knnMatch(desc, desc, k=3)

    pts_a, pts_b = [], []
    for m_self, m1, m2 in knn:
        if m1.trainIdx == m_self.queryIdx:
            continue
        if m1.distance / (m2.distance + 1e-8) > ratio_thresh:
            continue
        p1 = np.array(kps[m1.queryIdx].pt)
        p2 = np.array(kps[m1.trainIdx].pt)
        if np.linalg.norm(p1 - p2) < min_distance_px:
            continue  # exclude near-duplicate matches from local texture
        pts_a.append(p1)
        pts_b.append(p2)

    if len(pts_a) < min_cluster_size:
        return CopyMoveReport(detected=False, confidence=0.0,
                               notes="No spatially-separated self-similar matches.")

    pts_a = np.array(pts_a)
    pts_b = np.array(pts_b)

    # Cluster matches by the midpoint + offset vector, so pairs
    # belonging to the same duplicated region group together.
    offsets = pts_b - pts_a
    feats = np.hstack([offsets])
    if len(feats) < 2:
        return CopyMoveReport(detected=False, confidence=0.0,
                               notes="Too few candidate pairs to cluster.")

    Z = linkage(feats, method="average", metric="euclidean")
    labels = fcluster(Z, t=cluster_dist_thresh, criterion="distance")

    matches: list[CopyMoveMatch] = []
    mask = np.zeros((h, w), dtype=np.uint8)

    unique_labels = [l for l in np.unique(labels)
                      if (labels == l).sum() >= min_cluster_size][:max_clusters]

    for cid, lbl in enumerate(unique_labels):
        idx = labels == lbl
        src = pts_a[idx]
        dst = pts_b[idx]

        M, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_thresh, confidence=0.99,
        )
        if M is None or inliers is None:
            continue
        inlier_count = int(inliers.sum())
        if inlier_count < min_cluster_size:
            continue

        src_in = src[inliers.ravel().astype(bool)]
        dst_in = dst[inliers.ravel().astype(bool)]
        src_bbox = _bbox_of(src_in)
        dst_bbox = _bbox_of(dst_in)

        src_area = src_bbox[2] * src_bbox[3]
        dst_area = dst_bbox[2] * dst_bbox[3]
        if max(src_area, dst_area) < min_region_area:
            continue  # too small to be a meaningful duplicated object

        cv2.rectangle(mask, (src_bbox[0], src_bbox[1]),
                       (src_bbox[0] + src_bbox[2], src_bbox[1] + src_bbox[3]), 255, -1)
        cv2.rectangle(mask, (dst_bbox[0], dst_bbox[1]),
                       (dst_bbox[0] + dst_bbox[2], dst_bbox[1] + dst_bbox[3]), 255, -1)

        matches.append(CopyMoveMatch(
            cluster_id=cid,
            src_points=[tuple(p) for p in src_in.tolist()],
            dst_points=[tuple(p) for p in dst_in.tolist()],
            transform=M.tolist(),
            inlier_count=inlier_count,
            src_bbox=src_bbox,
            dst_bbox=dst_bbox,
        ))

    if not matches:
        return CopyMoveReport(detected=False, confidence=0.0, mask=mask,
                               notes="Candidate matches did not survive "
                                     "geometric-consistency filtering.")

    total_inliers = sum(m.inlier_count for m in matches)
    confidence = float(np.clip(total_inliers / 60.0, 0.0, 1.0))

    return CopyMoveReport(
        detected=True,
        confidence=confidence,
        matches=matches,
        mask=mask,
        notes=f"{len(matches)} duplicated region(s) with consistent affine "
              f"transforms detected.",
    )
