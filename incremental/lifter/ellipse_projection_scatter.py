import torch
import math

def batch_project_gaussians_to_pixels_scatter(
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
    """使用 scatter 的向量化版本

    核心思路：
    1. 为所有高斯批量计算投影参数
    2. 使用 torch.repeat 生成所有(高斯, 像素)对
    3. 批量计算马氏距离
    4. 使用 scatter_add 累积
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

            # ========== 真正的向量化实现 ==========
            # 对于每个高斯 j，生成其包围盒内所有像素的 flat index

            # 计算每个高斯的包围盒大小
            bbox_widths = (u_max - u_min + 1).long()  # [B_v]
            bbox_heights = (v_max - v_min + 1).long()
            bbox_sizes = bbox_widths * bbox_heights  # [B_v]

            # 跳过太大的包围盒（防止 OOM）
            max_bbox = bbox_sizes.max().item()
            if max_bbox > 100000:
                # 使用逐高斯方式
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

            # 使用 cumsum 计算每个高斯的起始偏移
            offsets = torch.zeros(B_v + 1, dtype=torch.long, device=device)
            offsets[1:] = torch.cumsum(bbox_sizes, dim=0)
            total_pixels = offsets[-1].item()

            if total_pixels > 50_000_000:  # 限制总像素数
                # 分块处理
                chunk_size = 1000000
                for chunk_start in range(0, total_pixels, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_pixels)

                    # 找到对应的高斯范围
                    g_indices = torch.searchsorted(offsets, torch.arange(chunk_start, chunk_end, device=device)) - 1
                    g_indices = g_indices.clamp(min=0, max=B_v-1)

                    # 计算每个 chunk 像素的高斯内偏移
                    local_indices = torch.arange(chunk_start, chunk_end, device=device) - offsets[g_indices]

                    # 计算像素坐标
                    u_coords = u_min[g_indices] + (local_indices % bbox_widths[g_indices])
                    v_coords = v_min[g_indices] + (local_indices // bbox_widths[g_indices])

                    # 计算马氏距离
                    du = u_coords.float() - u[g_indices]
                    dv = v_coords.float() - v[g_indices]
                    d = torch.stack([du, dv], dim=-1)

                    # 需要正确索引 inv_cov2d - 使用高级索引
                    mahal = (d.unsqueeze(1) @ inv_cov2d[g_indices].unsqueeze(2)).squeeze(-1).squeeze(-1)
                    mahal = (d * (d @ inv_cov2d[g_indices].transpose(-2, -1))).sum(dim=-1)

                    inside = mahal <= 1.0

                    # 处理覆盖的像素
                    u_covered = u_coords[inside]
                    v_covered = v_coords[inside]
                    g_covered = valid_indices[g_indices][inside]

                    for idx in range(u_covered.numel()):
                        vp, up = v_covered[idx].item(), u_covered[idx].item()
                        g_idx = g_covered[idx].item()
                        cnt = actual_counts[vp, up].item()
                        if cnt < max_gaussians_per_pixel:
                            pixel_to_gaussian[vp, up, cnt] = g_idx
                            actual_counts[vp, up] = cnt + 1
            else:
                # 生成所有像素坐标
                # u_coords[j, k] = u_min[j] + k % bbox_widths[j]
                # v_coords[j, k] = v_min[j] + k // bbox_widths[j]

                # 使用 torch.arange 和广播
                k_vals = torch.arange(total_pixels, device=device)

                # 找到每个 k 对应的高斯索引 j
                # offsets[j] <= k < offsets[j+1] 意味着 j 是 k 的高斯索引
                g_indices = torch.searchsorted(offsets[1:], k_vals)  # [total_pixels]

                # 高斯内偏移
                local_k = k_vals - offsets[g_indices]  # [total_pixels]

                # 像素坐标
                u_coords = u_min[g_indices] + (local_k % bbox_widths[g_indices])
                v_coords = v_min[g_indices] + (local_k // bbox_widths[g_indices])

                # 批量计算马氏距离
                du = u_coords.float() - u[g_indices]
                dv = v_coords.float() - v[g_indices]
                d = torch.stack([du, dv], dim=-1)

                # 马氏距离 d^T @ inv_cov @ d
                # 使用爱因斯坦求和
                mahal = torch.einsum('ni,nij,nj->n', d, inv_cov2d[g_indices], d)

                inside = mahal <= 1.0

                # 处理覆盖的像素
                u_covered = u_coords[inside]
                v_covered = v_coords[inside]
                g_covered = valid_indices[g_indices][inside]

                for idx in range(u_covered.numel()):
                    vp, up = v_covered[idx].item(), u_covered[idx].item()
                    g_idx = g_covered[idx].item()
                    cnt = actual_counts[vp, up].item()
                    if cnt < max_gaussians_per_pixel:
                        pixel_to_gaussian[vp, up, cnt] = g_idx
                        actual_counts[vp, up] = cnt + 1

        torch.cuda.synchronize()

    return pixel_to_gaussian, actual_counts
