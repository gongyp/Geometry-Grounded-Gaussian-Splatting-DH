import torch
import math
import numpy as np

def batch_project_gaussians_to_pixels(
    means,          # [N, 3]  高斯世界坐标
    covs3d,         # [N, 3, 3] 高斯世界协方差
    K,              # [3, 3]  相机内参
    R,              # [3, 3]  world2cam
    T,              # [3, 1]  world2cam
    H, W,           # 图像高宽
    batch_size=8192,# 分块大小
    sigma_scale=3.0 # 3σ 覆盖 99.7%
):
    """
    CUDA 批量前向投影
    返回：
        pixel_to_gaussian_indices: [H, W, max_gaussians_per_pixel] 每个像素对应高斯 index (padding with -1)
        actual_counts: [H, W] 每个像素实际有多少个高斯
    """
    device = means.device

    # 使用 numpy 列表收集结果 (更快)
    pixel_to_gaussian = [[[] for _ in range(W)] for _ in range(H)]

    fx, fy = K[0,0].item(), K[1,1].item()
    cx, cy = K[0,2].item(), K[1,2].item()

    N = means.shape[0]

    # 分批次处理
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        means_b = means[i:end]
        covs_b = covs3d[i:end]

        with torch.no_grad():
            # 1. 世界 → 相机坐标
            x_cam = means_b @ R.T + T.T
            z = x_cam[:, 2]
            valid = z > 0.01

            if valid.sum() == 0:
                continue

            x_cam = x_cam[valid]
            covs_b = covs_b[valid]
            z = z[valid]
            B_valid = x_cam.shape[0]
            if B_valid == 0:
                continue

            # 2. 相机协方差
            cov_cam = R @ covs_b @ R.T.transpose(-1,-2)

            # 3. 雅可比矩阵
            x, y, z_vals = x_cam[:, 0], x_cam[:, 1], x_cam[:, 2]
            z2 = z_vals ** 2
            J = torch.zeros(B_valid, 2, 3, device=device)
            J[:, 0, 0] = fx / z_vals
            J[:, 0, 2] = -fx * x / z2
            J[:, 1, 1] = fy / z_vals
            J[:, 1, 2] = -fy * y / z2

            # 4. 3D Cov → 2D Cov
            cov2d = J @ cov_cam @ J.transpose(-1,-2)

            # 5. 投影 2D 中心
            u = fx * x / z_vals + cx
            v = fy * y / z_vals + cy

            # 6. 椭圆半径
            eigvals, _ = torch.linalg.eigh(cov2d)
            rx = sigma_scale * torch.sqrt(eigvals[:, 1].clamp(min=1e-8))
            ry = sigma_scale * torch.sqrt(eigvals[:, 0].clamp(min=1e-8))

            # 7. 包围盒
            u_min = (u - rx).clamp(min=0).long()
            u_max = (u + rx).clamp(max=W-1).long()
            v_min = (v - ry).clamp(min=0).long()
            v_max = (v + ry).clamp(max=H-1).long()

            # 8. 逆协方差
            inv_cov2d = torch.linalg.inv(cov2d)

            # 9. 获取有效索引
            valid_indices = torch.where(valid)[0] + i

            # 10. 遍历高斯
            for j in range(B_valid):
                v0, v1 = v_min[j].item(), v_max[j].item()
                u0, u1 = u_min[j].item(), u_max[j].item()
                if v0 >= v1 or u0 >= u1:
                    continue

                # 生成网格像素
                v_pix = torch.arange(v0, v1+1, device=device)
                u_pix = torch.arange(u0, u1+1, device=device)
                vv, uu = torch.meshgrid(v_pix, u_pix, indexing='ij')
                vv_flat = vv.flatten()
                uu_flat = uu.flatten()

                # 偏移
                du = uu_flat - u[j]
                dv = vv_flat - v[j]
                d = torch.stack([du, dv], dim=-1)

                # 马氏距离
                mahalanobis = (d @ inv_cov2d[j] * d).sum(dim=-1)
                inside = mahalanobis <= 1.0

                # 全局高斯 index
                g_idx = valid_indices[j].item()

                # 填充
                for vp, up in zip(vv_flat[inside].cpu().tolist(), uu_flat[inside].cpu().tolist()):
                    pixel_to_gaussian[vp][up].append(g_idx)

    # 转换为 tensor 格式
    max_per_pixel = max(len(pixel_to_gaussian[v][u]) for v in range(H) for u in range(W))
    max_per_pixel = min(max(max_per_pixel, 50), 200)  # 限制范围

    pixel_to_gaussian_tensor = torch.full((H, W, max_per_pixel), -1, dtype=torch.long)
    actual_counts = torch.zeros((H, W), dtype=torch.long)

    for v in range(H):
        for u in range(W):
            gaussians = pixel_to_gaussian[v][u]
            cnt = len(gaussians)
            if cnt > 0:
                actual_counts[v, u] = cnt
                pixel_to_gaussian_tensor[v, u, :cnt] = torch.tensor(gaussians[:max_per_pixel], dtype=torch.long)

    return pixel_to_gaussian_tensor, actual_counts
