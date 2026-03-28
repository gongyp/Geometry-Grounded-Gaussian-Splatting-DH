# train.py 训练流程分析

## 一、所需输入文件

训练支持两种数据集格式：

### 1. Colmap 格式（常用）
```
source_path/
├── sparse/0/
│   ├── images.bin 或 images.txt    # 相机外参
│   ├── cameras.bin 或 cameras.txt  # 相机内参
│   └── points3D.ply 或 points3D.bin  # 3D点云
└── images/                         # 图像目录
    ├── image001.png
    ├── image002.png
    └── ...
```

### 2. Blender/NeRF 格式
```
source_path/
├── transforms_train.json   # 训练集相机变换矩阵
├── transforms_test.json    # 测试集相机变换矩阵
├── points3d.ply           # 3D点云
└── images/                # 图像文件
```

---

## 二、命令行参数

```bash
python train.py -s <source_path> -m <model_path> [其他选项]
```

主要参数（来自 `arguments/__init__.py`）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-s, --source_path` | 数据集路径 | 必须指定 |
| `-m, --model_path` | 输出模型路径 | 必须指定 |
| `--images` | 图像文件夹名 | "images" |
| `--sh_degree` | 球谐函数阶数 | 3 |
| `--white_background` | 白色背景 | False |
| `--iterations` | 训练迭代数 | 30000 |
| `--test_iterations` | 测试输出迭代 | [7000, 30000] |
| `--save_iterations` | 保存模型迭代 | [7000, 30000] |
| `--checkpoint_iterations` | 保存checkpoint迭代 | [15000] |

---

## 三、训练流程

### 1. 初始化阶段 (`training()` 函数)
```
Scene.__init__()
├── 读取数据集 (Colmap 或 Blender)
│   ├── 解析相机内外参
│   ├── 加载点云 (points3D.ply)
│   └── 加载图像
├── 创建 GaussianModel (高斯点模型)
├── 创建 app_model (appearance model)
└── 计算相机间最近邻关系 (multi_view.json)
```

### 2. 训练循环 (`for iteration in range(...)`)
```
for iteration in range(first_iter, opt.iterations + 1):
    │
    ├── 1. 更新学习率 (update_learning_rate)
    │
    ├── 2. 增加 SH 阶数 (每1000迭代)
    │
    ├── 3. 随机选择一个相机视角
    │
    ├── 4. 渲染 (render)
    │       └── 返回: render, depth, normal, visibility, radii
    │
    ├── 5. 计算损失
    │       ├── Ll1_render: L1 + Appearance loss
    │       ├── depth_normal_loss: 深度法线一致性 (iteration >= 7000)
    │       ├── ncc_loss: 多视角NCC loss (iteration >= 7000)
    │       ├── geo_loss: 多视角几何损失 (iteration >= 7000)
    │       └── total_loss = rgb_loss + λ_normal*depth_normal + λ_ncc*ncc + λ_geo*geo
    │
    ├── 6. 反向传播 (loss.backward())
    │
    ├── 7. densify_and_prune (迭代 500-15000)
    │       ├── 密集化: 添加新高斯点
    │       └── 剪枝: 移除多余高斯点
    │
    ├── 8. 优化器更新 (optimizer.step())
    │
    └── 9. 定期保存
            ├── scene.save(): 保存点云
            └── torch.save(): 保存checkpoint
```

### 3. 关键损失函数
- **L1_loss_appearance**: 渲染图与真值图的 L1 损失 + appearance embedding 损失
- **DSSIM**: 结构相似性损失
- **depth_normal**: 从深度图计算法线，与渲染法线的一致性
- **ncc_loss**: 多视角 patch 归一化互相关
- **geo_loss**: 多视角几何一致性

### 4. 输出文件
```
model_path/
├── cfg_args           # 训练配置参数
├── input.ply          # 复制输入点云
├── cameras.json       # 相机参数
├── multi_view.json    # 相机最近邻关系
├── point_cloud/iteration_XXX/
│   └── point_cloud.ply  # 训练后的点云
├── chkpntXXX.pth      # checkpoint
└── events.*           # TensorBoard 日志
```

---

## 四、典型运行命令

```bash
# Colmap 数据集
python train.py -s /path/to/colmap/scene -m ./output/my_model --iterations 30000

# Blender/NeRF 数据集
python train.py -s /path/to/blender/scene -m ./output/my_model --white_background

# 从 checkpoint 继续训练
python train.py --start_checkpoint ./output/my_model/chkpnt15000.pth
```
