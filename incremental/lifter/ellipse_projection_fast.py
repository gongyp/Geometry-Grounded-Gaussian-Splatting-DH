import torch
import math
import numpy as np

def batch_project_gaussians_to_pixels_fast(
    means,          # [N, 3]  高斯世界坐标
    covs3d,         # [N, 3, 3] 高斯世界协方差
    K,              # [3, 3]  相机内参
    R,              # [3, 3]  world2cam
    T,              # [3, 1]  world2cam
    H, W,           # 图像高宽
    batch_size=8192,# 分块大小
    sigma_scale=3.0 # 3σ 覆盖 99.7%
):
    """优化的批量前向投影 - 使用 Numba JIT 加速

    返回：
        pixel_to_gaussian_indices: [H, W, max_gaussians_per_pixel] 每个像素对应高斯 index (padding with -1)
        actual_counts: [H, W] 每个像素实际有多少个高斯
    """
    device = means.device
    N = means.shape[0]

    fx, fy = K[0,0].item(), K[1,1].item()
    cx, cy = K[0,2].item(), K[1,2].item()

    # 预分配输出
    max_per_pixel = 50
    pixel_to_gaussian = [[[] for _ in range(W)] for _ in range(H)]

    # 分批次处理
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        means_b = means[batch_start:batch_end]
        covs_b = covs3d[batch_start:batch_end]

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
            valid_indices = torch.where(valid)[0] + batch_start

            # 10. 使用 Numba 加速的函数
            # 转换数据为 CPU numpy 进行快速处理
            u_cpu = u.cpu()
            v_cpu = v.cpu()
            u_min_cpu = u_min.cpu()
            u_max_cpu = u_max.cpu()
            v_min_cpu = v_min.cpu()
            v_max_cpu = v_max.cpu()
            inv_cov2d_cpu = inv_cov2d.cpu()
            valid_indices_cpu = valid_indices.cpu()
            H_cpu = H
            W_cpu = W

            # 使用 Numba JIT 加速的内部函数
            from numba import njit, prange

            @njit(parallel=True)
            def process_gaussians_numba(
                u_arr, v_arr, u_min_arr, u_max_arr, v_min_arr, v_max_arr,
                inv_cov2d_arr, valid_indices_arr, pixel_to_gaussian, H, W
            ):
                """Numba 加速的高斯处理"""
                B = len(u_arr)

                for j in prange(B):
                    u_j = u_arr[j]
                    v_j = v_arr[j]
                    u0 = u_min_arr[j]
                    u1 = u_max_arr[j]
                    v0 = v_min_arr[j]
                    v1 = v_max_arr[j]

                    if u0 >= u1 or v0 >= v1:
                        continue

                    inv00 = inv_cov2d_arr[j, 0, 0]
                    inv01 = inv_cov2d_arr[j, 0, 1]
                    inv11 = inv_cov2d_arr[j, 1, 1]

                    g_idx = valid_indices_arr[j]

                    # 遍历像素
                    for vp in range(v0, v1 + 1):
                        for up in range(u0, u1 + 1):
                            du = up - u_j
                            dv = vp - v_j
                            mahal = du * (inv00 * du + inv01 * dv) + dv * (inv01 * du + inv11 * dv)

                            if mahal <= 1.0:
                                pixel_to_gaussian[vp][up].append(g_idx)

                return pixel_to_gaussian

            # 准备 numpy 数组
            inv_cov2d_np = inv_cov2d_cpu.numpy()
            valid_indices_np = valid_indices_cpu.numpy()
            u_min_np = u_min_cpu.numpy()
            u_max_np = u_max_cpu.numpy()
            v_min_np = v_min_cpu.numpy()
            v_max_np = v_max_cpu.numpy()
            u_np = u_cpu.numpy()
            v_np = v_cpu.numpy()

            # 处理
            pixel_to_gaussian = process_gaussians_numba(
                u_np, v_np, u_min_np, u_max_np, v_min_np, v_max_np,
                inv_cov2d_np, valid_indices_np, pixel_to_gaussian, H_cpu, W_cpu
            )

    # 转换为 tensor 格式
    max_count = max(len(pixel_to_gaussian[v][u]) for v in range(H) for u in range(W))
    max_count = min(max(max_count, 50), 200)

    pixel_to_gaussian_tensor = torch.full((H, W, max_count), -1, dtype=torch.long)
    actual_counts = torch.zeros((H, W), dtype=torch.long)

    for v in range(H):
        for u in range(W):
            gaussians = pixel_to_gaussian[v][u]
            cnt = len(gaussians)
            if cnt > 0:
                actual_counts[v, u] = cnt
                pixel_to_gaussian_tensor[v, u, :cnt] = torch.tensor(gaussians[:max_count], dtype=torch.long)

    return pixel_to_gaussian_tensor, actual_counts
