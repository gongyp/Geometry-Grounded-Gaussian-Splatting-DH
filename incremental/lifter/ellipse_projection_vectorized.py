import torch
import math

def batch_project_gaussians_to_pixels_vectorized(
    means,          # [N, 3]  高斯世界坐标
    covs3d,         # [N, 3, 3] 高斯世界协方差
    K,              # [3, 3]  相机内参
    R,              # [3, 3]  world2cam
    T,              # [3, 1]  world2cam
    H, W,           # 图像高宽
    batch_size=4096,# 分块大小
    sigma_scale=3.0 # 3σ 覆盖 99.7%
):
    """优化的 CUDA 批量前向投影 - 向量化版本

    返回：
        pixel_to_gaussian_indices: [H, W, max_gaussians_per_pixel] 每个像素对应高斯 index (padding with -1)
        actual_counts: [H, W] 每个像素实际有多少个高斯
    """
    device = means.device
    N = means.shape[0]

    fx, fy = K[0,0].item(), K[1,1].item()
    cx, cy = K[0,2].item(), K[1,2].item()

    # 预分配输出 - 假设每个像素平均 10 个高斯，最多 50 个
    max_per_pixel = 50
    pixel_to_gaussian = torch.full((H, W, max_per_pixel), -1, dtype=torch.long, device=device)
    actual_counts = torch.zeros((H, W), dtype=torch.long, device=device)

    # 分批次处理
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        means_b = means[i:end]
        covs_b = covs3d[i:end]
        B = means_b.shape[0]

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

            # 4. 2D Cov
            cov2d = J @ cov_cam @ J.transpose(-1,-2)

            # 5. 2D 中心
            u = fx * x / z_vals + cx
            v = fy * y / z_vals + cy

            # 6. 逆协方差
            inv_cov2d = torch.linalg.inv(cov2d + torch.eye(2, device=device).unsqueeze(0) * 1e-6)

            # 获取有效高斯的全局索引
            valid_indices = torch.where(valid)[0] + i  # 全局高斯索引

            # 7. 向量化处理每个高斯
            for j in range(B_valid):
                g_idx = valid_indices[j].item()

                # 包围盒
                u_min = max(0, int((u[j] - sigma_scale * torch.sqrt(cov2d[j, 0, 0]).item()).clamp(min=0).item()))
                u_max = min(W-1, int((u[j] + sigma_scale * torch.sqrt(cov2d[j, 0, 0]).item()).clamp(max=W-1).item()))
                v_min = max(0, int((v[j] - sigma_scale * torch.sqrt(cov2d[j, 1, 1]).item()).clamp(min=0).item()))
                v_max = min(H-1, int((v[j] + sigma_scale * torch.sqrt(cov2d[j, 1, 1]).item()).clamp(max=H-1).item()))

                if u_min >= u_max or v_min >= v_max:
                    continue

                # 生成网格像素 - 向量化
                u_pix = torch.arange(u_min, u_max + 1, device=device, dtype=torch.float32)
                v_pix = torch.arange(v_min, v_max + 1, device=device, dtype=torch.float32)
                vv, uu = torch.meshgrid(v_pix, u_pix, indexing='ij')
                uu_flat = uu.flatten()
                vv_flat = vv.flatten()

                # 偏移
                du = uu_flat - u[j]
                dv = vv_flat - v[j]
                d = torch.stack([du, dv], dim=-1)

                # 马氏距离 - 向量化
                mahalanobis = (d @ inv_cov2d[j] * d).sum(dim=-1)
                inside = mahalanobis <= 1.0

                # 获取椭圆内的像素坐标
                u_inside = uu_flat[inside].long()
                v_inside = vv_flat[inside].long()

                if u_inside.numel() == 0:
                    continue

                # 原子操作：添加到对应像素
                for idx_in_range in range(u_inside.numel()):
                    up = u_inside[idx_in_range].item()
                    vp = v_inside[idx_in_range].item()

                    cnt = actual_counts[vp, up].item()
                    if cnt < max_per_pixel:
                        pixel_to_gaussian[vp, up, cnt] = g_idx
                        actual_counts[vp, up] = cnt + 1

    return pixel_to_gaussian, actual_counts
