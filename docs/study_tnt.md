# Geometry-Grounded Gaussian Splatting - TNT 数据集训练与评估指南

本文档总结了在 TNT (Tanks and Temples) 数据集上进行训练、网格提取和评估的完整流程，以及常见问题解答。

---

## 目录

1. [训练阶段](#一训练阶段)
2. [网格提取阶段](#二网格提取阶段)
3. [评估阶段](#三评估阶段)
4. [输出文件说明](#四输出文件说明)
5. [自定义数据预处理](#五自定义数据预处理)
6. [常见问题解答](#六常见问题解答)

---

## 一、训练阶段

### 1.1 训练命令

```bash
python train.py -s eval_tnt/GT_TNT_dataset/Caterpillar \
                -m output/Caterpillar \
                -r 2 \
                --use_decoupled_appearance 3
```

### 1.2 参数说明

| 参数 | 说明 |
|------|------|
| `-s` | 数据集路径 |
| `-m` | 模型输出路径 |
| `-r` | 分辨率缩放因子 (2 = 半分辨率) |
| `--use_decoupled_appearance 3` | 使用解耦外观建模 |

### 1.3 训练输出文件

训练完成后，`output/Caterpillar/` 目录下生成以下文件：

| 文件/文件夹 | 说明 |
|------------|------|
| `cameras.json` | 相机参数 |
| `cfg_args` | 训练配置参数 |
| `chkpnt15000.pth` | 中间检查点 |
| `point_cloud/iteration_30000/point_cloud.ply` | 最终 Gaussian 点云 |
| `debug/` | 训练过程中的渲染验证图像 |
| `events.out.tfevents.*` | TensorBoard 日志 |
| `multi_view.json` | 多视角邻接关系 |

### 1.4 Debug 图像说明

`debug/` 文件夹下的每个 JPG 图片是 2×3 的组合图：

```
┌─────────────────┬─────────────────┬─────────────────┐
│   GT 图像       │   渲染图像      │  最近邻视角图像  │
│  (gt_img_show)  │   (img_show)   │ (depth_normal) │
├─────────────────┼─────────────────┼─────────────────┤
│  深度掩码(彩色) │   深度图(彩色)  │    法线图       │
│ (d_mask_color) │  (depth_color) │  (normal_show) │
└─────────────────┴─────────────────┴─────────────────┘
```

各子图含义：
- **GT 图像**：输入的真实图像
- **渲染图像**：当前模型渲染的结果
- **最近邻视角图像**：用于多视角几何约束的最近邻相机图像
- **深度掩码**：多视角几何一致性有效像素区域（Jet 色彩映射）
- **深度图**：渲染的深度值（Jet 色彩映射）
- **法线图**：渲染的场景法线

---

## 二、网格提取阶段

### 2.1 提取命令

```bash
python mesh_extract_tetrahedra.py -m output/Caterpillar --export_color
```

### 2.2 输出文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| `recon_init.ply` | ~709 MB | 初始网格（无颜色） |
| `recon.ply` | ~707 MB | 优化后网格（无颜色） |
| `recon_post.ply` | ~972 MB | 后处理网格（带颜色） |
| `cells.pt` | ~2.2 GB | 四面体网格数据（中间文件） |

### 2.3 recon.ply vs recon_post.ply

| 文件 | 是否带颜色 | 处理程度 |
|------|-----------|---------|
| recon_init.ply | ❌ 无 | 初始提取 |
| recon.ply | ❌ 无 | 二值搜索优化 |
| **recon_post.ply** | ✅ 有 | 后处理 + 着色 |

只有 `recon_post.ply` 是最终可用的带颜色的网格模型。

### 2.4 cells.pt 用途

`cells.pt` 是四面体网格的单元格结构数据，用于 Marching Tetrahedra 算法：
- 从 Gaussian 模型生成四面体网格顶点
- 构建四面体单元格连接关系
- 加速重复提取（可缓存复用）

---

## 三、评估阶段

### 3.1 评估命令

```bash
python eval_tnt/run.py --dataset-dir eval_tnt/GT_TNT_dataset/Caterpillar \
                       --traj-path eval_tnt/GT_TNT_dataset/Caterpillar/Caterpillar_COLMAP_SfM.log \
                       --ply-path output/Caterpillar/recon_post.ply \
                       --out-dir output/Caterpillar/mesh
```

### 3.2 评估指标

| 指标 | 说明 |
|------|------|
| **Precision** | 重建网格中有多少比例是准确的 |
| **Recall** | GT 场景中有多少比例被重建出来 |
| **F-Score** | Precision 和 Recall 的调和平均 |

### 3.3 评估输出文件

`output/Caterpillar/mesh/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `metrics.json` | 评估指标 JSON |
| `Caterpillar.precision.ply` | 精确率可视化（重建点云 + 误差颜色） |
| `Caterpillar.recall.ply` | 召回率可视化（GT 点云 + 缺失颜色） |
| `Caterpillar.precision.txt` | 精确率直方图数据 |
| `Caterpillar.recall.txt` | 召回率直方图数据 |
| `PR_*.png/pdf` | P-R 曲线图 |

### 3.4 评估流程详解

```
1. 加载数据
   ├── 读取重建网格 (recon_post.ply)
   ├── 读取 GT 点云 (Caterpillar.ply)
   └── 读取相机轨迹和变换矩阵

2. 点云采样
   ├── 从网格面片中心采样顶点
   └── 在面片内添加额外采样点

3. 轨迹对齐 (Registration)
   ├── 粗配准: registration_vol_ds (voxel_size = dTau*80)
   ├── 精配准: registration_vol_ds (voxel_size = dTau/2)
   └── 超精细配准: registration_unif (voxel_size = 2*dTau)

4. 裁剪
   ├── 使用 JSON 定义的有效区域裁剪
   └── 去除重建区域外的噪声

5. 下采样
   └── 对重建和 GT 点云进行体素下采样

6. 距离计算
   ├── distance1: 重建点 → GT 的最近距离
   └── distance2: GT 点 → 重建的最近距离

7. 计算指标
   ├── Precision = count(distance1 < τ) / 总重建点数
   ├── Recall = count(distance2 < τ) / 总 GT 点数
   └── F-Score = 2 × P × R / (P + R)
```

---

## 四、输出文件说明

### 4.1 cfg_args 文件

存储训练时的所有参数配置：
```
Namespace(
    sh_degree=3,
    source_path='.../Caterpillar',
    model_path='output/Caterpillar',
    resolution=2,
    use_decoupled_appearance=3,
    multi_view_num=8,
    multi_view_max_angle=30,
    ...
)
```

### 4.2 multi_view.json 文件

记录每个视角对应的 8 个最近邻视角（用于多视角几何一致性约束）：
```json
{"ref_name":"000001","nearest_name":["000002","000003","000004","000005","000138","000363","000137","000364"]}
```

---

## 五、自定义数据预处理

### 5.1 需要生成的文件

| 文件 | 说明 |
|------|------|
| `*_COLMAP_SfM.log` | COLMAP 相机轨迹 |
| `*.ply` | GT 点云或重建点云 |
| `*_trans.txt` | 坐标变换矩阵 |
| `*.json` | 评估区域裁剪定义 |

### 5.2 生成方法

#### 5.2.1 COLMAP_SfM.log（相机轨迹）

```bash
# 1. COLMAP 特征提取
colmap feature_extractor \
    --database_path database.db \
    --image_path images/ \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.camera_params "fx,fy,cx,cy"

# 2. 特征匹配
colmap exhaustive_matcher --database_path database.db

# 3. SfM 重建
colmap mapper \
    --database_path database.db \
    --image_path images/ \
    --output_path sparse/

# 4. 导出为 SfM log 格式
colmap model_converter \
    --input_path sparse/0 \
    --output_path scene_NAME_COLMAP_SfM.log \
    --output_type COLMAP
```

#### 5.2.2 GT 点云

- **方式 A**：使用 COLMAP MVS
```bash
colmap point_cloud_triangulator \
    --database_path database.db \
    --image_path images/ \
    --input_path sparse/0 \
    --output_path sparse/points.ply
```

- **方式 B**：下载 TNT 官方 GT 点云（推荐）

#### 5.2.3 trans.txt（坐标变换矩阵）

```python
import numpy as np
import open3d as o3d

# 1. 加载重建点云和 GT 点云
pcd_rec = o3d.io.read_point_cloud("reconstruction.ply")
pcd_gt = o3d.io.read_point_cloud("gt_pointcloud.ply")

# 2. 使用 ICP 精对齐
# (手动或使用 Open3D)

# 3. 保存变换矩阵
transformation = ...  # 4x4 矩阵
np.savetxt("scene_NAME_trans.txt", transformation)
```

#### 5.2.4 json（评估区域裁剪）

```python
import json

crop_data = {
    "class_name": "SelectionPolygonVolume",
    "version_major": 1,
    "version_minor": 0,
    "axis_min": z_min,
    "axis_max": z_max,
    "bounding_polygon": [[x1, y1, 0], [x2, y2, 0], ...],
    "orthogonal_axis": "Z"
}

with open("scene_NAME.json", "w") as f:
    json.dump(crop_data, f)
```

### 5.3 TNT 官方资源

从以下地址下载 TNT 官方评估文件：
- https://tanksandtemples.org/download/

包含：
- 重建网格 (Reconstruction)
- 相机位姿 (Camera Poses)
- 对齐矩阵 (Alignment)
- 裁剪文件 (Crop files)

---

## 六、常见问题解答

### Q1: recon.ply 和 recon_post.ply 有什么区别？

**答**：只有 `recon_post.ply` 带颜色。`recon.ply` 是中间结果，不包含颜色信息。颜色是在后处理阶段（`recon_post.ply`）添加的。

### Q2: cells.pt 有什么用？

**答**：`cells.pt` 是四面体网格的单元格结构，用于 Marching Tetrahedra 算法计算 SDF 零点位置。这是中间数据，对最终使用只需要 `.ply` 网格文件即可。

### Q3: debug 文件夹下的 6 个子图分别是什么？

**答**：
1. GT 图像 - 输入的真实图像
2. 渲染图像 - 模型渲染结果
3. 最近邻视角图像 - 多视角约束参考图像
4. 深度掩码 - 有效像素区域
5. 深度图 - 渲染深度
6. 法线图 - 场景法线

### Q4: 如何在自己的数据上进行训练？

**答**：
1. 准备图像数据
2. 运行 COLMAP SfM 重建
3. 使用项目提供的预处理脚本生成训练格式
4. 运行训练命令

### Q5: Precision 和 Recall 是如何计算的？

**答**：
- **Precision** = 距离 GT 表面 < τ 的重建点数 / 重建点总数
- **Recall** = 距离重建表面 < τ 的 GT 点数 / GT 点总数
- **F-Score** = 2 × Precision × Recall / (Precision + Recall)

### Q6: Caterpillar.json, Caterpillar_trans.txt, Caterpillar_COLMAP_SfM.log 分别是什么？

- **Caterpillar.json**: 评估区域裁剪定义（TNT 官方提供）
- **Caterpillar_trans.txt**: 坐标变换矩阵（对齐重建与 GT）
- **Caterpillar_COLMAP_SfM.log**: COLMAP 相机位姿轨迹

---

## 训练结果摘要

| 项目 | 数值 |
|------|------|
| 训练迭代次数 | 30,000 |
| 训练时间 | ~44 分钟 |
| 最终 PSNR | 26.59 dB |
| 最终 L1 Loss | 0.0383 |
| Precision | 48.09% |
| Recall | 67.02% |
| **F-Score** | **56.00%** |

---

*文档生成日期: 2024-03-14*
