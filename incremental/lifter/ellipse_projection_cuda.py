import torch
import torch_scatter

def fast_pixel2gaussians(
    means,          # [N, 3] GPU
    covs3d,         # [N, 3, 3] GPU
    K,              # [3, 3]
    R,              # [3, 3] world2cam
    T,              # [3, 1]
    H, W,
    batch_size=4096,
    sigma_scale=3.0
):
    device = means.device
    N = means.shape[0]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # 最终输出：每个像素存储高斯索引列表
    pixel_gaus = [[] for _ in range(H * W)]

    for b_start in range(0, N, batch_size):
        b_end = min(b_start + batch_size, N)
        B = b_end - b_start

        m = means[b_start:b_end]
        c = covs3d[b_start:b_end]

        with torch.no_grad():
            # --------------------------
            # 1. 相机坐标系
            # --------------------------
            x_cam = m @ R.T + T.T  # [B,3]
            x, y, z = x_cam.unbind(-1)
            valid = z > 0.01
            if not valid.any(): continue

            x_cam = x_cam[valid]
            c = c[valid]
            x, y, z = x_cam.unbind(-1)
            B = len(x)

            # --------------------------
            # 2. 投影到 2D (u, v)
            # --------------------------
            u = fx * x / z + cx
            v = fy * y / z + cy

            # --------------------------
            # 3. 2D 协方差
            # --------------------------
            z2 = z ** 2
            J = torch.zeros(B, 2, 3, device=device)
            J[:,0,0] = fx / z
            J[:,0,2] = -fx * x / z2
            J[:,1,1] = fy / z
            J[:,1,2] = -fy * y / z2

            cov_cam = R @ c @ R.T
            cov2d = J @ cov_cam @ J.transpose(1,2)

            # --------------------------
            # 4. 椭圆范围
            # --------------------------
            vals, vecs = torch.linalg.eigh(cov2d)
            rx = sigma_scale * torch.sqrt(vals[:,1].clamp(1e-8))
            ry = sigma_scale * torch.sqrt(vals[:,0].clamp(1e-8))

            u0 = (u - rx).clamp(0, W-1).long()
            u1 = (u + rx).clamp(0, W-1).long()
            v0 = (v - ry).clamp(0, H-1).long()
            v1 = (v + ry).clamp(0, H-1).long()

            # --------------------------
            # 5. 预计算所有逆矩阵（关键优化）
            # --------------------------
            eps = 1e-6
            cov2d_reg = cov2d + eps * torch.eye(2, device=device).unsqueeze(0)
            inv_covs = torch.linalg.inv(cov2d_reg)

            valid_indices = torch.where(valid)[0]

            # --------------------------
            # 6. 批量生成像素网格
            # --------------------------
            u_list, v_list, g_list = [], [], []

            for i in range(B):
                vv = torch.arange(v0[i], v1[i]+1, device=device)
                uu = torch.arange(u0[i], u1[i]+1, device=device)
                uu, vv = torch.meshgrid(uu, vv, indexing='ij')
                uu = uu.flatten()
                vv = vv.flatten()

                # 马氏距离（使用预计算的逆矩阵）
                du = uu - u[i]
                dv = vv - v[i]
                d = torch.stack([du, dv], dim=-1)
                maha = (d @ inv_covs[i] * d).sum(-1)
                mask = maha <= 1.0

                uu = uu[mask]
                vv = vv[mask]

                u_list.append(uu)
                v_list.append(vv)
                g_list.append(torch.full_like(uu, b_start + i))

            uu = torch.cat(u_list)
            vv = torch.cat(v_list)
            gg = torch.cat(g_list)
            pix_idx = vv * W + uu

            # --------------------------
            # 6. GPU 原子写入（超快）
            # --------------------------
            for pix, g in zip(pix_idx.cpu().tolist(), gg.cpu().tolist()):
                pixel_gaus[pix].append(g)

    #  reshape [H, W]
    return [pixel_gaus[i*W : (i+1)*W] for i in range(H)]