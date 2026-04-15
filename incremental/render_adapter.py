"""Render Adapter.

Adapts the current project's rendering pipeline (diff_gaussian_rasterization)
to the interface expected by cl-splats.
"""

import math
from PIL import Image as PILImage
import torch
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as ggs_render
from arguments import PipelineParams


@dataclass
class RenderOutput:
    """Render output in a format similar to gsplat."""

    image: torch.Tensor  # (C, H, W)
    alpha: torch.Tensor  # (1, H, W)
    depth: torch.Tensor  # (1, H, W)
    normal: Optional[torch.Tensor] = None  # (3, H, W)
    radii: Optional[torch.Tensor] = None  # (N,)
    viewspace_points: Optional[torch.Tensor] = None  # (N, 3)
    visibility_filter: Optional[torch.Tensor] = None  # (N,)
    topk_ids: Optional[torch.Tensor] = None  # (K, H, W)
    topk_weights: Optional[torch.Tensor] = None  # (K, H, W)
    topk_valid_count: Optional[torch.Tensor] = None  # (H, W)


class GGSRenderAdapter:
    """Adapter for rendering with diff_gaussian_rasterization.

    This adapter wraps the GaussianModel rendering and provides output
    compatible with what cl-splats expects.
    """

    def __init__(
        self,
        gaussian_model: GaussianModel,
        pipe_params: Optional[Any] = None,
        bg_color: Optional[torch.Tensor] = None,
    ):
        """Initialize the render adapter.

        Args:
            gaussian_model: The GaussianModel to render
            pipe_params: Pipeline parameters for rendering
            bg_color: Background color tensor (C,)
        """
        self.model = gaussian_model

        # Default pipeline params if not provided
        if pipe_params is None:
            # Create a simple object with required attributes
            class SimplePipeParams:
                convert_SHs_python = False
                compute_cov3D_python = False
                debug = False
            self.pipe = SimplePipeParams()
        else:
            self.pipe = pipe_params

        # Default background color (black)
        if bg_color is None:
            self.bg_color = torch.zeros(3, device="cuda")
        else:
            self.bg_color = bg_color

        self.kernel_size = 1  # Default kernel size

    def render(
        self,
        camera,
        scaling_modifier: float = 1.0,
        require_depth: bool = True,
        return_topk: bool = False,
        topk_k: int = 3,
    ) -> RenderOutput:
        """Render the scene from a camera viewpoint.

        Args:
            camera: Camera object with required attributes
            scaling_modifier: Scale modifier for 3D filters
            require_depth: Whether to compute depth
            return_topk: Whether to return per-pixel Top-K contributor buffers
            topk_k: Number of Top-K contributor slots to request

        Returns:
            RenderOutput object with rendered image, alpha, depth, etc.
        """
        result = ggs_render(
            viewpoint_camera=camera,
            pc=self.model,
            pipe=self.pipe,
            bg_color=self.bg_color,
            kernel_size=self.kernel_size,
            scaling_modifier=scaling_modifier,
            require_depth=require_depth,
            return_topk=return_topk,
            topk_k=topk_k,
        )

        rendered_image = result["render"]
        rendered_alpha = result["mask"]
        rendered_depth = result["median_depth"]
        radii = result["radii"]

        # Handle optional normal output
        rendered_normal = None
        if "normal" in result and result["normal"] is not None:
            rendered_normal = result["normal"]

        return RenderOutput(
            image=rendered_image,
            alpha=rendered_alpha,
            depth=rendered_depth,
            normal=rendered_normal,
            radii=radii,
            viewspace_points=result.get("viewspace_points"),
            visibility_filter=result.get("visibility_filter"),
            topk_ids=result.get("topk_ids"),
            topk_weights=result.get("topk_weights"),
            topk_valid_count=result.get("topk_valid_count"),
        )

    def render_batch(
        self,
        cameras: list,
        scaling_modifier: float = 1.0,
        require_depth: bool = True,
        return_topk: bool = False,
        topk_k: int = 3,
    ) -> list[RenderOutput]:
        """Render a batch of cameras.

        Args:
            cameras: List of camera objects
            scaling_modifier: Scale modifier for 3D filters
            require_depth: Whether to compute depth
            return_topk: Whether to return per-pixel Top-K contributor buffers
            topk_k: Number of Top-K contributor slots to request

        Returns:
            List of RenderOutput objects
        """
        outputs = []
        for camera in cameras:
            output = self.render(
                camera,
                scaling_modifier=scaling_modifier,
                require_depth=require_depth,
                return_topk=return_topk,
                topk_k=topk_k,
            )
            outputs.append(output)
        return outputs


class GSGaussiansWrapper:
    """Wrapper for GaussianModel to make it compatible with gsplat-style rendering.

    This provides a more complete interface for integration with cl-splats.
    """

    def __init__(self, gaussian_model: GaussianModel):
        """Initialize wrapper.

        Args:
            gaussian_model: The GaussianModel to wrap
        """
        self.model = gaussian_model

    @property
    def means(self) -> torch.Tensor:
        """Get means (positions)."""
        return self.model.get_xyz

    @property
    def scales(self) -> torch.Tensor:
        """Get scales (in linear space)."""
        return self.model.get_scaling

    @property
    def quats(self) -> torch.Tensor:
        """Get quaternions."""
        return self.model.get_rotation

    @property
    def opacities(self) -> torch.Tensor:
        """Get opacities (in linear space)."""
        return self.model.get_opacity

    @property
    def sh0(self) -> torch.Tensor:
        """Get DC SH coefficients (N, 1, 3)."""
        return self.model._features_dc.transpose(1, 2)

    @property
    def shN(self) -> torch.Tensor:
        """Get non-DC SH coefficients (N, K-1, 3)."""
        return self.model._features_rest.transpose(1, 2)

    @property
    def num_gaussians(self) -> int:
        """Get number of Gaussians."""
        return self.model._xyz.shape[0]

    def get_features(self) -> torch.Tensor:
        """Get all SH features (N, 3, K)."""
        return self.model.get_features

    def get_sg_features(self) -> Dict[str, torch.Tensor]:
        """Get SG features."""
        return {
            "axis": self.model.get_sg_axis,
            "sharpness": self.model.get_sg_sharpness,
            "color": self.model.get_sg_color,
        }


def create_camera_from_colmap(
    R: torch.Tensor,
    T: torch.Tensor,
    fx: float,
    fy: float,
    width: int,
    height: int,
    image_path: Optional[str] = None,
    camera_id: int = 0,
) -> Any:
    """Create a camera object from COLMAP parameters.

    This is a helper to create camera objects compatible with the renderer.

    Args:
        R: (3, 3) rotation matrix
        T: (3,) translation vector
        fx: Focal length x
        fy: Focal length y
        width: Image width
        height: Image height
        image_path: Optional path to image
        camera_id: Camera ID

    Returns:
        Camera object
    """
    from scene.camera import Camera
    from utils.camera_utils import camera_to_JSON

    # Create camera from parameters
    camera = Camera(
        colmap_id=camera_id,
        R=R,
        T=T,
        FoVx=2 * math.atan(width / (2 * fx)),
        FoVy=2 * math.atan(height / (2 * fy)),
        image=PILImage.open(image_path) if image_path else None,
        image_name=str(camera_id),
        uid=camera_id,
        camera_name=camera_id,
        resolution=(width, height),
        preloaded_image=None,
        image_device=torch.device("cuda"),
    )

    return camera
