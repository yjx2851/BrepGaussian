### B-Rep拟合部分

本目录包含用于从带语义标签的点云自动恢复规则平面/柱面边界线与面片的完整流水线实现，包括三部分：
- 面片几何基元拟合与交线提取
- 线框重建
- 表面重建

整体流程：对带 `label` 与 `edge` 字段的 PCD 点云，先按标签对每个区域进行平面/柱面 RANSAC 拟合并去重，然后计算面面交线，并利用边缘点约束线段范围；随后将线段端点吸附为角点集合，并输出为 JSON/OBJ 线框结果；最后基于线、圆与角点在每个面上构造 2D 边界多边形，通过三角剖分/布尔运算生成三角网格面片，形成完整B-Rep结构。

### 环境依赖

- **Python**: 3.11 or later
- **必须的第三方库**：  

```bash
pip install numpy scipy scikit-learn open3d trimesh shapely
```

### 运行

#### 1. 对单个 PCD 运行完整流水线

```bash
python surface_fitting_pipeline.py --pcd path/to/input.pcd --output output_dir
```

参数说明：
- `--pcd`: 输入 PCD 文件路径。  
- `--output`: 输出目录，内含 `pipeline_results.json` 与多个 OBJ。  
- `--plane_threshold` / `--cylinder_threshold`: RANSAC 内点距离阈值，单位与点云坐标一致。  
- `--min_points`: 每个标签用于拟合的最小点数。 
- `--no_visualize`: 若指定，则不进行在线可视化，仅保存文件。

#### 2. 重建线框（多 PCD）

```bash
python batch_process_pipeline.py ^
  --inputs path/to/pcd_dir ^
  --output line_batch_output ^
  --plane_threshold 0.01 --cylinder_threshold 0.01 --min_points 50
```

- `--inputs`：包含若干 `.pcd` 。  
- `--output`：批量输出根目录，每个输入文件对应 `output/<basename>/` 子目录，其中包含该样本的 `pipeline_results.json` 与 OBJ。

#### 3. 生成面片网格

在前一步得到 `line_batch_output` 后：

```bash
python batch_process_surfaces.py ^
  --inputs line_batch_output ^
  --output surface_batch_output
```

- `--inputs`：包含若干子目录、每个子目录内有 `pipeline_results.json` 。  
- `--output`：每个输入对应一个输出子目录，内含每个面的 `surface_XXX.obj` 与整体 `all_surfaces.obj`。  
- 可选 `--force`：若输出目录已存在，则覆盖生成。

