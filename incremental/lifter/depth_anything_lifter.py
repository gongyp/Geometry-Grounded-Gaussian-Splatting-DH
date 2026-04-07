"""Depth-Anything V2 lifter.

Estimates monocular depth using Depth-Anything V2 from HuggingFace and
lifts 2-D change masks into a per-Gaussian change mask using multi-view
depth back-projection and Gaussian proximity scoring.

This is the reference implementation from submodules/cl-splats, adapted for
use with GGSGaussianAdapter.
"""

import numpy as np
import torch
from PIL import Image as PILImage
from typing import List, Optional
from dataclasses import dataclass
import faiss
from huggingface_hub import snapshot_download


@dataclass
class LiftResult:
    """Result from the lifting operation."""

    positive_mask: torch.Tensor  # (N,) Boolean mask of changed Gaussians
    seed_score: torch.Tensor  # (N,) Score for each Gaussian
    neg_score: torch.Tensor  # (N,) Negative evidence score
    depth_consistency: Optional[torch.Tensor] = None  # (N,) Depth consistency score


class DepthAnythingLifter:
    """Lifter that uses Depth-Anything V2 for monocular depth estimation."""

    def __init__(
        self,
        depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
        local_model_path: Optional[str] = "./models/depth-anything-v2-small",
        k_nn: int = 8,
        local_radius_thresh: float = 2.5,
        depth_tol_abs: float = 0.05,
        depth_tol_rel: float = 0.05,
        lambda_seed: float = 2.0,
        lambda_neg: float = 0.25,
        min_visible_views: int = 2,
        min_positive_views: int = 2,
        min_seed_views: int = 1,
        min_positive_ratio: float = 0.3,
        final_thresh: float = 0.6,
    ):
        """Initialize the Depth-Anything lifter.

        Args:
            depth_model: HuggingFace model ID (used only to download on first run).
            local_model_path: Local directory to cache the model. Defaults to
                ``./models/depth-anything-v2-small``. On first call the model is
                downloaded from HuggingFace if the directory does not exist.
                Subsequent calls load directly from this local path.
        """
        from transformers import pipeline as hf_pipeline

        if local_model_path is not None:
            import os
            if not os.path.isdir(local_model_path):
                print(f"[DepthAnythingLifter] Downloading model to {local_model_path} ...")
                snapshot_download(
                    repo_id=depth_model,
                    local_dir=local_model_path,
                    endpoint="https://hf-mirror.com",
                )
            model_path = local_model_path
        else:
            # Download (if not already cached) and use the default cache location.
            model_path = snapshot_download(
                repo_id=depth_model,
                endpoint="https://hf-mirror.com",
            )

        self._pipe = hf_pipeline(task="depth-estimation", model=model_path, device=0, torch_dtype=torch.float16, low_cpu_mem_usage=True)

        # Lifting hyper-parameters
        self.k_nn = k_nn
        self.local_radius_thresh = local_radius_thresh
        self.depth_tol_abs = depth_tol_abs
        self.depth_tol_rel = depth_tol_rel
        self.lambda_seed = lambda_seed
        self.lambda_neg = lambda_neg
        self.min_visible_views = min_visible_views
        self.min_positive_views = min_positive_views
        self.min_seed_views = min_seed_views
        self.min_positive_ratio = min_positive_ratio
        self.final_thresh = final_thresh

    @torch.no_grad()
    def estimate_depth(self, observation: torch.Tensor) -> torch.Tensor:
        """Estimate depth from an observation image ``[H, W, 3]`` in ``[0, 1]``.

        Returns a depth map as a ``float32`` tensor ``[H, W]`` with metric depth
        in meters (from Depth-Anything `predicted_depth`).
        """
        obs_np = (observation.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
        pil_img = PILImage.fromarray(obs_np)

        # Use predicted_depth (metric depth in meters), not depth (uint8 [0-255])
        depth_tensor = self._pipe(pil_img)["predicted_depth"]
        # observation is [H, W, C], so use [:2] to get [H, W]
        obs_h, obs_w = observation.shape[:2]
        if depth_tensor.shape != (obs_h, obs_w):
            import torch.nn.functional as F
            depth_tensor = F.interpolate(
                depth_tensor.unsqueeze(0).unsqueeze(0), size=(obs_h, obs_w), mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)
        # Ensure positive depth values
        depth_tensor = torch.clamp(depth_tensor, min=0.0)
        return depth_tensor.float()

    def build_faiss_index(self, target):
        """Build a FAISS GPU index for kNN search.

        Call this once before calling search_with_index multiple times.

        Args:
            target: (N, D) tensor of target vectors (e.g., Gaussian positions)

        Returns:
            Tuple of (res, index) that should be passed to search_with_index
        """
        t = target.cpu().numpy().astype('float32')
        dim = t.shape[1]

        # Create GPU index
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, dim, faiss.GpuIndexFlatConfig())
        index.add(t)

        return res, index

    @torch.no_grad()
    def search_with_index(self, query, res, index, k=8):
        """Search kNN using a pre-built FAISS index.

        Args:
            query: (M, D) tensor of query vectors
            res: FAISS GPU resources (from build_faiss_index)
            index: FAISS GPU index (from build_faiss_index)
            k: number of nearest neighbors

        Returns:
            Tuple of (knn_dists, knn_idx) as tensors on query's device
        """
        q = query.cpu().numpy().astype('float32')
        dists, idx = index.search(q, k)
        knn_dists = torch.from_numpy(dists).to(query.device)
        knn_idx = torch.from_numpy(idx).to(query.device)
        return knn_dists, knn_idx

    @torch.no_grad()
    def faiss_knn(self, query, target, k=8):
        """Use GPU faiss for fast kNN (creates index each call)."""
        res, index = self.build_faiss_index(target)
        return self.search_with_index(query, res, index, k)

    @torch.no_grad()
    def lift(
        self,
        gaussians,  # GGSGaussianAdapter
        cameras: List,
        change_masks: List[torch.Tensor],
    ) -> LiftResult:
        """Multi-view lifting.

        For each view, estimates depth with Depth-Anything, back-projects
        changed pixels to 3-D, assigns evidence to nearby Gaussians, and
        accumulates multi-view positive/negative evidence.

        Returns:
            LiftResult with positive mask and scores
        """
        device = gaussians.get_positions().device
        N = gaussians.num_gaussians

        seed_score = torch.zeros(N, device=device)
        seed_votes = torch.zeros(N, device=device)
        neg_score = torch.zeros(N, device=device)
        neg_votes = torch.zeros(N, device=device)
        visible_views = torch.zeros(N, dtype=torch.int32, device=device)
        positive_views = torch.zeros(N, dtype=torch.int32, device=device)
        seed_views = torch.zeros(N, dtype=torch.int32, device=device)

        positions = gaussians.get_positions()  # (N, 3)
        scales = gaussians.get_scales()  # (N, 3)

        # Build FAISS index once for all kNN searches (reused across views)
        faiss_res, faiss_index = self.build_faiss_index(positions)

        for view_id, (cam, mask) in enumerate(zip(cameras, change_masks)):
            obs = cam.original_image.permute(1, 2, 0).contiguous()
            depth = self.estimate_depth(obs).to(device)  # [H, W]

            H, W = depth.shape
            mask = mask.to(device)

            # --- Positive pixels (changed) ---
            pos_pixels = (mask > 0.5) & torch.isfinite(depth) & (depth > 0)
            if pos_pixels.any():
                ys, xs = torch.nonzero(pos_pixels, as_tuple=True)
                # Sub-sample to avoid OOM in the kNN distance matrix (M×N).
                max_pos = min(ys.numel(), 2048)
                if ys.numel() > max_pos:
                    perm = torch.randperm(ys.numel(), device=device)[:max_pos]
                    ys = ys[perm]
                    xs = xs[perm]
                d = depth[ys, xs]

                # Back-project to camera coordinates
                x_cam = (xs.float() - cam.Cx) / cam.Fx * d
                y_cam = (ys.float() - cam.Cy) / cam.Fy * d
                z_cam = d

                ones = torch.ones_like(z_cam)
                p_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)  # [M, 4]
                Twc = cam.Twc.to(device)  # [4, 4]
                p_world_h = p_cam @ Twc.T  # [M, 4]
                p_world = p_world_h[..., :3] / p_world_h[..., 3:]  # [M, 3]

                # kNN in Gaussian means — cap M so cdist stays in memory
                # dists = torch.cdist(p_world, positions)  # [M, N]
                # knn_dists, knn_idx = torch.topk(dists, k=min(self.k_nn, N), dim=-1, largest=False)
                # del dists  # free immediately
                knn_dists, knn_idx = self.search_with_index(p_world, faiss_res, faiss_index, k=min(self.k_nn, N))  # [M,k]

                # Local scale-aware distance
                local_scales = scales[knn_idx]  # [M, k, 3]
                denom = local_scales.norm(dim=-1) + 1e-6  # [M, k]
                d_local = knn_dists / denom

                valid = d_local < self.local_radius_thresh  # [M, k]
                if valid.any():
                    # Depth consistency: project only the k neighbour means (not all N)
                    # into camera space — (M, k, 4) instead of (M, N, 4).
                    Tcw = torch.inverse(Twc)  # [4, 4]
                    knn_means = positions[knn_idx]  # [M, k, 3]
                    M, k = knn_means.shape[:2]
                    knn_means_h = torch.cat(
                        [knn_means, torch.ones(M, k, 1, device=device)], dim=-1
                    )  # [M, k, 4]
                    knn_cam = knn_means_h @ Tcw.T  # [M, k, 4]
                    z_knn = knn_cam[..., 2]  # [M, k]

                    depth_pix = d.unsqueeze(-1)  # [M, 1]
                    depth_ok = (z_knn - depth_pix).abs() < (
                        self.depth_tol_abs + self.depth_tol_rel * depth_pix
                    )

                    valid_final = valid & depth_ok  # [M, k]
                    if valid_final.any():
                        d_local_valid = d_local.masked_fill(~valid_final, 1e9)
                        weights = torch.exp(-0.5 * d_local_valid**2)
                        weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-8
                        weights = weights / weights_sum  # [M, k]

                        mask_vals = mask[ys, xs].unsqueeze(-1).float()  # [M, 1]
                        contrib = mask_vals * weights  # [M, k]

                        flat_idx = knn_idx.view(-1)
                        flat_contrib = contrib.view(-1)
                        flat_valid = valid_final.view(-1)

                        flat_idx = flat_idx[flat_valid]
                        flat_contrib = flat_contrib[flat_valid]

                        seed_score.index_add_(0, flat_idx, flat_contrib)
                        seed_votes.index_add_(0, flat_idx, flat_contrib)

                        affected = torch.unique(flat_idx)
                        positive_views[affected] += 1
                        seed_views[affected] += 1
                        visible_views[affected] += 1

            # --- Weak negatives from un-masked pixels (sub-sampled) ---
            neg_pixels = (~pos_pixels) & torch.isfinite(depth) & (depth > 0)
            if neg_pixels.any():
                ys_n, xs_n = torch.nonzero(neg_pixels, as_tuple=True)
                max_neg = min(ys_n.numel(), 1024)
                perm = torch.randperm(ys_n.numel(), device=device)[:max_neg]
                ys_n = ys_n[perm]
                xs_n = xs_n[perm]
                d_n = depth[ys_n, xs_n]

                x_cam_n = (xs_n.float() - cam.Cx) / cam.Fx * d_n
                y_cam_n = (ys_n.float() - cam.Cy) / cam.Fy * d_n
                z_cam_n = d_n

                ones_n = torch.ones_like(z_cam_n)
                p_cam_n = torch.stack([x_cam_n, y_cam_n, z_cam_n, ones_n], dim=-1)
                Twc = cam.Twc.to(device)
                p_world_h_n = p_cam_n @ Twc.T
                p_world_n = p_world_h_n[..., :3] / p_world_h_n[..., 3:]

                # dists_n = torch.cdist(p_world_n, positions)
                # knn_dists_n, knn_idx_n = torch.topk(
                #     dists_n, k=min(self.k_nn, N), dim=-1, largest=False
                # )
                knn_dists_n, knn_idx_n = self.search_with_index(p_world_n, faiss_res, faiss_index, k=min(self.k_nn, N))

                local_scales_n = scales[knn_idx_n]
                denom_n = local_scales_n.norm(dim=-1) + 1e-6
                d_local_n = knn_dists_n / denom_n

                valid_n = d_local_n < self.local_radius_thresh
                if valid_n.any():
                    d_local_valid_n = d_local_n.masked_fill(~valid_n, 1e9)
                    weights_n = torch.exp(-0.5 * d_local_valid_n**2)
                    weights_sum_n = weights_n.sum(dim=-1, keepdim=True) + 1e-8
                    weights_n = weights_n / weights_sum_n

                    mask_vals_n = mask[ys_n, xs_n].unsqueeze(-1).float()
                    contrib_n = (1.0 - mask_vals_n) * weights_n

                    flat_idx_n = knn_idx_n.view(-1)
                    flat_contrib_n = contrib_n.view(-1)
                    flat_valid_n = valid_n.view(-1)

                    flat_idx_n = flat_idx_n[flat_valid_n]
                    flat_contrib_n = flat_contrib_n[flat_valid_n]

                    neg_score.index_add_(0, flat_idx_n, flat_contrib_n)
                    neg_votes.index_add_(0, flat_idx_n, flat_contrib_n)

                    affected_n = torch.unique(flat_idx_n)
                    visible_views[affected_n] += 1

        # Combine evidence
        pos = self.lambda_seed * seed_score
        neg = self.lambda_neg * neg_score

        score = pos / (pos + neg + 1e-8)

        # Multi-view consistency filtering
        keep = (
            (visible_views >= self.min_visible_views)
            & (positive_views >= self.min_positive_views)
            & (seed_views >= self.min_seed_views)
            & (positive_views.float() / (visible_views.float() + 1e-8) >= self.min_positive_ratio)
        )
        score = torch.where(keep, score, torch.zeros_like(score))

        changed_gaussians = score > self.final_thresh

        # If too few changed, return all as changed for incremental update
        if changed_gaussians.sum() < 100:
            changed_gaussians = torch.ones(N, dtype=torch.bool, device=device)

        return LiftResult(
            positive_mask=changed_gaussians,
            seed_score=score,
            neg_score=neg_score,
        )


def create_depth_anything_lifter(**kwargs):
    """Create a DepthAnythingLifter instance."""
    return DepthAnythingLifter(**kwargs)
