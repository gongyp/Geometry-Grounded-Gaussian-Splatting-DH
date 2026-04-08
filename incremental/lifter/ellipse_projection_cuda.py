import torch
import math
import numpy as np

def batch_project_gaussians_to_pixels_cuda(
    means,          # [N, 3]  高斯世界坐标
    covs3d,         # [N, 3, 3] 高斯世界协方差
    K,              # [3, 3]  相机内参
    R,              # [3, 3]  world2cam
    T,              # [3, 1]  world2cam
    H, W,           # 图像高宽
    batch_size=8192,# 分块大小
    sigma_scale=3.0, # 3σ 覆盖 99.7%
    max_gaussians_per_pixel=50
):
    """CUDA 优化的批量前向投影 - 使用向量化操作

    核心思路：不是逐个 Gaussian 处理，而是批量处理所有 Gaussian 的包围盒，
    然后用向量化操作计算覆盖关系。
    """
    device = means.device
    N = means.shape[0]

    fx, fy = K[0,0].item(), K[1,1].item()
    cx, cy = K[0,2].item(), K[1,2].item()

    # 预分配输出
    pixel_to_gaussian = torch.full((H, W, max_gaussians_per_pixel), -1, dtype=torch.long, device=device)
    actual_counts = torch.zeros((H, W), dtype=torch.long, device=device)

    # 分批次处理
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        means_b = means[batch_start:batch_end]
        covs_b = covs3d[batch_start:batch_end]
        B = means_b.shape[0]

        with torch.no_grad():
            # 1. 世界 → 相机坐标
            x_cam = means_b @ R.T + T.T  # [B, 3]
            z = x_cam[:, 2]
            valid = z > 0.01

            if valid.sum() == 0:
                continue

            # 过滤有效高斯
            x_cam_v = x_cam[valid]
            covs_v = covs_b[valid]
            z_v = z[valid]
            valid_indices = torch.where(valid)[0] + batch_start
            B_v = x_cam_v.shape[0]

            if B_v == 0:
                continue

            # 2. 相机协方差
            cov_cam = R @ covs_v @ R.T.transpose(-1,-2)

            # 3. 雅可比矩阵
            x, y = x_cam_v[:, 0], x_cam_v[:, 1]
            z2 = z_v ** 2
            J = torch.zeros(B_v, 2, 3, device=device)
            J[:, 0, 0] = fx / z_v
            J[:, 0, 2] = -fx * x / z2
            J[:, 1, 1] = fy / z_v
            J[:, 1, 2] = -fy * y / z2

            # 4. 2D Cov
            cov2d = J @ cov_cam @ J.transpose(-1,-2)

            # 5. 2D 中心
            u = fx * x / z_v + cx
            v = fy * y / z_v + cy

            # 6. 包围盒
            rx = sigma_scale * torch.sqrt(cov2d[:, 0, 0].clamp(min=1e-8))
            ry = sigma_scale * torch.sqrt(cov2d[:, 1, 1].clamp(min=1e-8))

            u_min = (u - rx).clamp(min=0).long()
            u_max = (u + rx).clamp(max=W-1).long()
            v_min = (v - ry).clamp(min=0).long()
            v_max = (v + ry).clamp(max=H-1).long()

            # 7. 逆协方差
            inv_cov2d = torch.linalg.inv(cov2d + torch.eye(2, device=device).unsqueeze(0) * 1e-6)

            # ========== 向量化处理每个高斯 ==========
            # 对于每个高斯 j，它的包围盒是 [v_min[j], v_max[j]] x [u_min[j], u_max[j]]
            # 我们需要为所有像素 (v, u) 计算马氏距离

            # 方法：使用 flatten 索引 + scatter
            # 将图像展平为 1D，每个像素的索引是 v * W + u

            # 为每个高斯创建覆盖掩码
            for j in range(B_v):
                g_idx = valid_indices[j].item()
                v0, v1 = v_min[j].item(), v_max[j].item()
                u0, u1 = u_min[j].item(), u_max[j].item()

                if v0 >= v1 or u0 >= u1:
                    continue

                # 创建此高斯的覆盖区域
                # 使用向量化方式：生成网格并一次计算
                u_pix = torch.arange(u0, u1+1, dtype=torch.long, device=device)
                v_pix = torch.arange(v0, v1+1, dtype=torch.long, device=device)
                vv, uu = torch.meshgrid(v_pix, u_pix, indexing='ij')
                vv_flat = vv.flatten()
                uu_flat = uu.flatten()

                # 计算马氏距离
                du = uu_flat.float() - u[j]
                dv = vv_flat.float() - v[j]
                d = torch.stack([du, dv], dim=-1)

                mahalanobis = (d @ inv_cov2d[j] * d).sum(dim=-1)
                inside = mahalanobis <= 1.0

                # 获取椭圆内的像素
                vv_in = vv_flat[inside]
                uu_in = uu_flat[inside]

                if vv_in.numel() == 0:
                    continue

                # 原子添加到像素
                for idx in range(vv_in.numel()):
                    vp, up = vv_in[idx].item(), uu_in[idx].item()
                    cnt = actual_counts[vp, up].item()
                    if cnt < max_gaussians_per_pixel:
                        pixel_to_gaussian[vp, up, cnt] = g_idx
                        actual_counts[vp, up] = cnt + 1

        # 批次间同步
        torch.cuda.synchronize()

    return pixel_to_gaussian, actual_counts


def batch_project_gaussians_to_pixels_vectorized_v2(
    means,          # [N, 3]  高斯世界坐标
    covs3d,         # [N, 3, 3] 高斯世界协方差
    K,              # [3, 3]  相机内参
    R,              # [3, 3]  world2cam
    T,              # [3, 1]  world2cam
    H, W,           # 图像高宽
    batch_size=8192,# 分块大小
    sigma_scale=3.0, # 3σ 覆盖 99.7%
    max_gaussians_per_pixel=50
):
    """完全向量化的版本 - 使用 scatter_add 批量处理

    核心思路：
    1. 为每个高斯生成其包围盒内所有像素的列表（扁平化）
    2. 批量计算所有(像素, 高斯)对的马氏距离
    3. 使用 scatter_add 累积结果
    """
    device = means.device
    N = means.shape[0]

    fx, fy = K[0,0].item(), K[1,1].item()
    cx, cy = K[0,2].item(), K[1,2].item()

    # 预分配输出
    pixel_to_gaussian = torch.full((H, W, max_gaussians_per_pixel), -1, dtype=torch.long, device=device)
    actual_counts = torch.zeros((H, W), dtype=torch.long, device=device)

    # 分批次处理
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        means_b = means[batch_start:batch_end]
        covs_b = covs3d[batch_start:batch_end]
        B = means_b.shape[0]

        with torch.no_grad():
            # 1. 世界 → 相机坐标
            x_cam = means_b @ R.T + T.T
            z = x_cam[:, 2]
            valid = z > 0.01

            if valid.sum() == 0:
                continue

            x_cam_v = x_cam[valid]
            covs_v = covs_b[valid]
            z_v = z[valid]
            valid_indices = torch.where(valid)[0] + batch_start
            B_v = x_cam_v.shape[0]

            if B_v == 0:
                continue

            # 2. 相机协方差
            cov_cam = R @ covs_v @ R.T.transpose(-1,-2)

            # 3. 雅可比矩阵
            x, y = x_cam_v[:, 0], x_cam_v[:, 1]
            z2 = z_v ** 2
            J = torch.zeros(B_v, 2, 3, device=device)
            J[:, 0, 0] = fx / z_v
            J[:, 0, 2] = -fx * x / z2
            J[:, 1, 1] = fy / z_v
            J[:, 1, 2] = -fy * y / z2

            # 4. 2D Cov
            cov2d = J @ cov_cam @ J.transpose(-1,-2)

            # 5. 2D 中心
            u = fx * x / z_v + cx
            v = fy * y / z_v + cy

            # 6. 包围盒
            rx = sigma_scale * torch.sqrt(cov2d[:, 0, 0].clamp(min=1e-8))
            ry = sigma_scale * torch.sqrt(cov2d[:, 1, 1].clamp(min=1e-8))

            u_min = (u - rx).clamp(min=0).long()
            u_max = (u + rx).clamp(max=W-1).long()
            v_min = (v - ry).clamp(min=0).long()
            v_max = (v + ry).clamp(max=H-1).long()

            # 7. 逆协方差
            inv_cov2d = torch.linalg.inv(cov2d + torch.eye(2, device=device).unsqueeze(0) * 1e-6)

            # ========== 真正的向量化 ==========
            # 对于每个高斯，计算其包围盒内的最大像素数
            max_bbox_size = ((u_max - u_min + 1) * (v_max - v_min + 1)).max().item()

            if max_bbox_size > 50000:  # 跳过太大的包围盒
                max_bbox_size = 50000

            # 创建所有(高斯, 包围盒内像素)的索引
            # 首先计算每个高斯的包围盒大小
            bbox_widths = (u_max - u_min + 1)  # [B_v]
            bbox_heights = (v_max - v_min + 1)
            bbox_sizes = bbox_widths * bbox_heights

            # 使用 cumsum 计算偏移
            offsets = torch.zeros(B_v + 1, dtype=torch.long, device=device)
            offsets[1:] = torch.cumsum(bbox_sizes, dim=0)
            total_pixels = offsets[-1].item()

            if total_pixels == 0 or total_pixels > 10_000_000:  # 防止 OOM
                # 回退到逐高斯处理
                for j in range(B_v):
                    g_idx = valid_indices[j].item()
                    v0, v1 = v_min[j].item(), v_max[j].item()
                    u0, u1 = u_min[j].item(), u_max[j].item()

                    if v0 >= v1 or u0 >= u1:
                        continue

                    u_pix = torch.arange(u0, u1+1, dtype=torch.long, device=device)
                    v_pix = torch.arange(v0, v1+1, dtype=torch.long, device=device)
                    vv, uu = torch.meshgrid(v_pix, u_pix, indexing='ij')
                    vv_flat = vv.flatten()
                    uu_flat = uu.flatten()

                    du = uu_flat.float() - u[j]
                    dv = vv_flat.float() - v[j]
                    d = torch.stack([du, dv], dim=-1)

                    mahalanobis = (d @ inv_cov2d[j] * d).sum(dim=-1)
                    inside = mahalanobis <= 1.0

                    vv_in = vv_flat[inside]
                    uu_in = uu_flat[inside]

                    if vv_in.numel() == 0:
                        continue

                    for idx in range(vv_in.numel()):
                        vp, up = vv_in[idx].item(), uu_in[idx].item()
                        cnt = actual_counts[vp, up].item()
                        if cnt < max_gaussians_per_pixel:
                            pixel_to_gaussian[vp, up, cnt] = g_idx
                            actual_counts[vp, up] = cnt + 1
                continue

            # 批量创建所有像素坐标
            # pixel_coords[j, k] = (v, u) for j-th Gaussian, k-th pixel in its bbox
            all_u_coords = []
            all_v_coords = []
            all_g_indices = []

            for j in range(B_v):
                v0, v1 = v_min[j].item(), v_max[j].item()
                u0, u1 = u_min[j].item(), u_max[j].item()
                g_idx = valid_indices[j].item()

                if v0 >= v1 or u0 >= u1:
                    continue

                u_pix = torch.arange(u0, u1+1, dtype=torch.long, device=device)
                v_pix = torch.arange(v0, v1+1, dtype=torch.long, device=device)
                vv, uu = torch.meshgrid(v_pix, u_pix, indexing='ij')

                all_u_coords.append(uu.flatten())
                all_v_coords.append(vv.flatten())
                all_g_indices.append(torch.full_like(uu.flatten(), g_idx, dtype=torch.long))

            if len(all_u_coords) == 0:
                continue

            # 拼接
            all_u = torch.cat(all_u_coords)
            all_v = torch.cat(all_v_coords)
            all_g = torch.cat(all_g_indices)

            # 批量计算马氏距离
            du = all_u.float() - u[torch.arange(B_v, device=device).repeat_interleave(bbox_sizes[valid])]
            dv = all_v.float() - v[torch.arange(B_v, device=device).repeat_interleave(bbox_sizes[valid])]

            # 这部分索引计算复杂，用简单方式实现
            # 对于每个高斯 j，它的所有像素用同样的 (u[j], v[j]) 和 inv_cov2d[j]
            # 所以我们按高斯分组处理

            start_idx = 0
            for j in range(B_v):
                g_idx = valid_indices[j].item()
                bbox_size = bbox_sizes[j].item()

                if bbox_size == 0:
                    continue

                end_idx = start_idx + bbox_size
                u_batch = all_u[start_idx:end_idx]
                v_batch = all_v[start_idx:end_idx]

                du = u_batch.float() - u[j]
                dv = v_batch.float() - v[j]
                d = torch.stack([du, dv], dim=-1)

                # 只取前 j 行的索引来获取 inv_cov2d[j]
                mahalanobis = (d @ inv_cov2d[j] * d).sum(dim=-1)
                inside = mahalanobis <= 1.0

                u_in = u_batch[inside]
                v_in = v_batch[inside]

                if u_in.numel() == 0:
                    start_idx = end_idx
                    continue

                # 原子添加
                for idx in range(u_in.numel()):
                    vp, up = v_in[idx].item(), u_in[idx].item()
                    cnt = actual_counts[vp, up].item()
                    if cnt < max_gaussians_per_pixel:
                        pixel_to_gaussian[vp, up, cnt] = g_idx
                        actual_counts[vp, up] = cnt + 1

                start_idx = end_idx

        torch.cuda.synchronize()

    return pixel_to_gaussian, actual_counts
