"""Improved DinoV2-based change detector with nearest-neighbor matching.

This detector addresses the issue of slight viewpoint differences between
rendered and GT images by using nearest-neighbor matching instead of
direct positional correspondence.

Instead of comparing features at the same spatial location, it:
1. Builds a feature library from the GT image
2. For each rendered feature patch, finds the nearest GT feature
3. Computes similarity using the matched pair (not positional match)
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from dataclasses import dataclass
from typing import Optional
import faiss
import logging

from clsplats.change_detection.base_detector import BaseDetector
from clsplats.utils.custom_types import Image

log = logging.getLogger(__name__)


class NNMatchingDetector(BaseDetector):
    """DINOv2 detector with nearest-neighbor matching for viewpoint robustness."""

    def __init__(self, cfg: "NNMatchingChangeConfig"):
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True
        )
        self.model.eval()
        self.model.to(self.device)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _preprocess_image(self, image: Image) -> torch.Tensor:
        """Resize to DINO-compatible dimensions and apply ImageNet normalisation."""
        DINO_PATCH_SIZE = 14
        h, w, c = image.shape
        aligned_h = h - (h % DINO_PATCH_SIZE)
        aligned_w = w - (w % DINO_PATCH_SIZE)
        image_t = image.permute(2, 0, 1)
        resized = F.interpolate(
            image_t.unsqueeze(0),
            size=(aligned_h, aligned_w),
            mode="bilinear",
            align_corners=False,
        )
        normalised = self.normalize(resized.squeeze(0))
        return normalised

    def _dilate_mask(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Morphologically dilate a binary mask."""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        kernel = torch.ones(
            (1, 1, kernel_size, kernel_size), dtype=torch.float32, device=mask.device
        )
        mask = mask.float().unsqueeze(0)
        dilated = F.conv2d(mask, kernel, padding=kernel_size // 2)
        dilated = (dilated > 0).float()
        return dilated.squeeze(0).squeeze(0)

    def _build_faiss_index(self, features: torch.Tensor):
        """Build FAISS index from features.

        Args:
            features: (N, C) feature tensor

        Returns:
            faiss index
        """
        N, C = features.shape
        features_np = features.cpu().numpy().astype('float32')

        if self.cfg.metric == "cosine":
            # Normalize for cosine similarity search
            norms = np.linalg.norm(features_np, axis=1, keepdims=True)
            norms[norms == 0] = 1  # avoid division by zero
            features_np = features_np / norms

        if self.cfg.use_faiss_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, C, faiss.GpuIndexFlatConfig())
        else:
            index = faiss.IndexFlatL2(C)

        index.add(features_np)
        return index

    def _search_nn(self, index, query: torch.Tensor, k: int = 1):
        """Search nearest neighbors.

        Args:
            index: FAISS index
            query: (M, C) query tensor
            k: number of nearest neighbors

        Returns:
            distances: (M, k)
            indices: (M, k)
        """
        query_np = query.cpu().numpy().astype('float32')

        if self.cfg.metric == "cosine":
            norms = np.linalg.norm(query_np, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query_np = query_np / norms

        distances, indices = index.search(query_np, k)
        return distances, indices

    def predict_change_mask(
        self, rendered_image: Image, observation: Image
    ) -> torch.Tensor:
        """Return a change mask using nearest-neighbor matching.

        This method:
        1. Extracts DINOv2 features from both images
        2. Builds a feature library from observation (GT)
        3. For each rendered patch, finds the nearest GT feature
        4. Computes cosine similarity using matched pairs
        5. Thresholds to get change mask

        Args:
            rendered_image: Rendered image [H, W, 3]
            observation: Ground-truth observation [H, W, 3]

        Returns:
            Boolean mask [H, W] where True indicates a changed pixel.
        """
        rendered = self._preprocess_image(rendered_image)
        observed = self._preprocess_image(observation)

        rendered = rendered.unsqueeze(0)
        observed = observed.unsqueeze(0)

        with torch.no_grad():
            # Extract features
            (rendered_feats,) = self.model.get_intermediate_layers(
                rendered, reshape=True
            )
            (observed_feats,) = self.model.get_intermediate_layers(
                observed, reshape=True
            )

            # Shape: (B, C, H, W) -> (H, W, C)
            rendered_feats = rendered_feats.squeeze(0).permute(1, 2, 0)
            observed_feats = observed_feats.squeeze(0).permute(1, 2, 0)

            H, W, C = rendered_feats.shape
            H_obs, W_obs, C_obs = observed_feats.shape

            # Flatten for matching
            rendered_flat = rendered_feats.reshape(-1, C)  # (N, C)
            gt_flat = observed_feats.reshape(-1, C_obs)  # (N_obs, C)

            # Build FAISS index from GT features
            gt_index = self._build_faiss_index(gt_flat)

            # Find nearest GT match for each rendered patch
            _, nn_indices = self._search_nn(gt_index, rendered_flat, k=self.cfg.k_nn)

            # For k=1, nn_indices is (N, 1)
            if self.cfg.k_nn == 1:
                nn_indices = nn_indices.squeeze(-1)  # (N,)

            # Get matched GT features
            if self.cfg.k_nn == 1:
                matched_gt = gt_flat[nn_indices]  # (N, C)
            else:
                # Average k nearest neighbors
                matched_gt = gt_flat[nn_indices].mean(dim=1)  # (N, C)

            # Compute cosine similarity between rendered and matched GT
            cos_sim = self.cos(rendered_flat, matched_gt)  # (N,)
            cos_sim = cos_sim.reshape(H, W)

            # Threshold
            mask = cos_sim < self.cfg.threshold

            # Dilate if enabled
            if self.cfg.dilate_mask:
                mask = self._dilate_mask(mask, self.cfg.dilate_kernel_size)

            # Upsample to original size
            if self.cfg.upsample:
                orig_h, orig_w, _ = rendered_image.shape
                mask = mask.float().unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(
                    mask,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )
                mask = mask.squeeze(0).squeeze(0) > 0.5

            return mask


@dataclass
class NNMatchingChangeConfig:
    """Configuration for nearest-neighbor matching change detection."""

    threshold: float = 0.7
    dilate_mask: bool = True
    dilate_kernel_size: int = 5
    upsample: bool = True
    use_faiss_gpu: bool = True
    k_nn: int = 1
    metric: str = "cosine"  # "cosine" or "L2"


def create_nn_matching_detector(
    threshold: float = 0.7,
    dilate_mask: bool = True,
    dilate_kernel_size: int = 5,
    upsample: bool = True,
    use_faiss_gpu: bool = True,
    k_nn: int = 1,
    metric: str = "cosine",
) -> NNMatchingDetector:
    """Create a nearest-neighbor matching change detector."""
    cfg = NNMatchingChangeConfig(
        threshold=threshold,
        dilate_mask=dilate_mask,
        dilate_kernel_size=dilate_kernel_size,
        upsample=upsample,
        use_faiss_gpu=use_faiss_gpu,
        k_nn=k_nn,
        metric=metric,
    )
    return NNMatchingDetector(cfg)
