# GPBSF PointNet++ Kaggle 云端双卡极速训练指南

本文档提供将 `GPBSF` 中的 PointNet++ (SSG) 训练任务部署至 Kaggle 双 GPU（NVIDIA T4 x 2）平台的完整操作指南。

> 💡 **说明**：Mamba3D 模型依赖特定的 CUDA 11.7 算子与预编译扩展，推荐直接在本地使用专用 Docker 容器（`.docker` 下 `docker compose run mamba3d`）运行；PointNet++ 采用纯 PyTorch + TorchScript JIT 向量化算子，与 Kaggle 原生环境完全兼容，可直接利用双卡并发训练。

---

## 一、平台规格与配置要求

在 Kaggle 创建 Notebook 时，请在右侧设置面板（**Notebook options**）完成以下配置：

| 配置项 | 推荐设定 | 说明 |
| :--- | :--- | :--- |
| **Accelerator** | `GPU T4 x 2` | 必须选择双卡 T4，用于并行并发训练两个随机种子 |
| **Language** | `Python` | 基础宿主语言 |
| **Environment** | `Always use latest environment` | Kaggle 默认环境自带兼容的 PyTorch，无需额外安装沙盒 |
| **Internet** | `Internet On` | 建议开启 |
| **Persistence** | `Variables and Files` | 建议开启 |

---

## 二、本地资源打包与上传至 Kaggle

### 1. 运行一键打包脚本

在本地机器打开 Windows PowerShell，定位至 `scripts/kaggle` 目录：

```powershell
Set-Location "d:\0-research\00-papers\thesis\graduate-thesis\code\GPBSF\scripts\kaggle"

# 语法：.\prepare_kaggle_assets.ps1 -DatasetPath <本地bgspcd_v4_robust的绝对路径>
.\prepare_kaggle_assets.ps1 -DatasetPath "D:\你的路径\data\bgspcd_v4_robust"
```

该脚本会在 `kaggle_assets/` 下自动生成：
- `gpbsf_code.zip`：过滤了 `.git`、`runs/`、历史缓存与本地大数据的纯净代码包。
- `gpbsf_dataset_v4.zip`：包含 30,400 个样本及 `manifest.jsonl` 的数据集压缩包。
- `dataset-metadata-code.json` 与 `dataset-metadata-dataset.json`：Kaggle CLI 上传元数据模板。

---

### 2. 上传为 Kaggle Dataset（提供两种方式）

#### 方式 A：通过 Kaggle 网页端直接上传（推荐）

1. 登录 [Kaggle](https://www.kaggle.com/)，点击左侧菜单栏 **Datasets** -> **New Dataset**。
2. 上传代码包：
   - 拖拽 `gpbsf_code.zip`。
   - 数据集标题输入：`gpbsf-code`。
   - 点击 **Create**。
3. 上传数据包：
   - 再次点击 **New Dataset**。
   - 拖拽 `gpbsf_dataset_v4.zip`。
   - 数据集标题输入：`gpbsf-dataset-v4`。
   - 点击 **Create**。

#### 方式 B：通过 Kaggle 官方 CLI 命令行上传

若本地已配置 Kaggle API Token（`~/.kaggle/kaggle.json`）：

```powershell
kaggle datasets create -p "d:\0-research\00-papers\thesis\graduate-thesis\code\GPBSF\kaggle_assets" -r zip
```

---

## 三、Kaggle Notebook 运行步骤

### 1. 新建 Notebook 并挂载 Datasets

1. 在 Kaggle 顶部点击 **+ Create** -> **New Notebook**。
2. 在右侧面板中，确保 **Accelerator** 设为 `GPU T4 x 2`。
3. 点击右侧面板中的 **Add Input**（或 **Data** -> **Add Data**）：
   - 搜索并添加已创建的两个数据集：`gpbsf-code` 与 `gpbsf-dataset-v4`。
   - 挂载成功后，输入路径分别为：
     - `/kaggle/input/gpbsf-code/`
     - `/kaggle/input/gpbsf-dataset-v4/`

### 2. 导入 Notebook 执行

直接将仓库中的 [`scripts/kaggle/pointnet2_kaggle_train.ipynb`](file:///d:/0-research/00-papers/thesis/graduate-thesis/code/GPBSF/scripts/kaggle/pointnet2_kaggle_train.ipynb) 上传至 Kaggle（点击 **File** -> **Import Notebook**）：

- **[Cell 1] 工作区初始化与数据集链接**：
  自动扫描 `/kaggle/input` 下的代码和数据集并建立软链接，免去手动解压等待。
- **[Cell 2] 环境检测与 PointNet++ JIT 模型自检**：
  验证双 GPU 可用性，并执行一次单批次前向传播自检。
- **[Cell 3] 双 GPU 并发实时流式训练、评估与成果打包**：
  - GPU 0 运行种子 `3407`，GPU 1 同时运行种子 `3408`；
  - 纯 Python 行缓冲实时将两张卡的训练 Epoch 日志流式回传显示；
  - 训练完成后自动执行两卡的测试集评估并汇总指标；
  - 自动生成打包归档文件 `/kaggle/working/pointnet2_results_v4_robust.tar.gz`。

---

## 四、训练成果下载

训练完成后：
1. 在 Kaggle Notebook 右侧面板展开 **Output** 区域；
2. 刷新文件列表，找到 `/kaggle/working/pointnet2_results_v4_robust.tar.gz`；
3. 点击右侧三点菜单（**...**）并选择 **Download**，即可将包含两组随机种子的检查点 `best.pt`、`last.pt`、`training.log`、`metrics.jsonl` 及汇总结果 `aggregate.json` 一键下载至本地。
