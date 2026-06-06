# YAML 配置管理指南 — 优势、格式与实践

在机器学习与深度学习项目（特别是 Transformer 模型的开发）中，合理地管理超参数是实验成功的关键。本项目采用 **YAML (YAML Ain't Markup Language)** 格式来管理模型和训练配置。

本篇指南将详细阐述 YAML 在深度学习中的优势、其基础语法格式，以及具体使用方法。

---
## 1. 为什么使用 YAML？（核心优势）

在编写深度学习代码时，超参数通常有三种管理方式：**硬编码在代码中**、**完全依赖命令行参数 (argparse)**、或**使用配置文件**。相比之下，使用 YAML 配置文件具有压倒性的优势：
### 1.1 结构清晰与可读性高

YAML 采用缩进表示层级关系，非常适合表达深度学习中“模型结构”、“训练流程”、“数据路径”等分类清晰的超参数组。它的排版干净，没有 XML 的繁琐标签或 JSON 的大量括号与逗号。

### 1.2 实验的可复现性（Reproducibility）

* **痛点**：做实验时，如果我们只用命令行调参（例如：`python train.py --d_model 256 --lr 0.0003`），几天后很容易忘记当时用了什么参数，或者由于拼写错误导致实验对不上。

* **解决**：在本项目中，训练脚本 `src/train.py` 开始运行后，会**自动将本次训练所用的完整 YAML 配置复制一份并保存到实验输出目录下**（如 `experiments/baseline/config.yaml`）。这意味着任何实验的结果都与其参数配置文件永久绑定，任何人都可以随时复现该实验。

### 1.3 版本控制与可追溯性

每一组实验配置（例如 `configs/experiment_dmodel.yaml`）都是一个纯文本文件，可以轻松地提交到 Git 仓库。你可以通过 `git diff` 清楚地对比“今天跑的模型”和“上周最好的模型”在参数上有何不同。

### 1.4 代码与配置的完美解耦

开发者无需修改任何一行 Python 源代码，即可调整模型的层数、注意力头数、学习率等关键参数。这使得核心代码（`src/transformer.py`）非常纯粹地专注于计算逻辑。

---
## 2. YAML 语法与格式规范

YAML 的设计目标是“易于人阅读”，但它有一些严格的格式规范：

### 2.1 基础键值对

使用冒号加空格 `key: value` 表示。**注意：冒号后面必须有至少一个空格！**

```yaml

# 正确写法

learning_rate: 0.001

  

# 错误写法（缺少空格，会被解析为普通字符串）

learning_rate:0.001

```

### 2.2 层级与缩进（嵌套结构）

* YAML 使用缩进表示层级关系。

* **规则**：**必须使用空格缩进，绝对不能使用 Tab 键！**（通常使用 2 个或 4 个空格）。

```yaml

model:

  d_model: 128

  num_heads: 4

```

### 2.3 注释

使用井号 `#` 进行单行注释。这在调参时极其有用，可以直接在参数后面写明修改原因或建议取值：
```yaml

dropout: 0.1        # 随机失活率（防止过拟合，如果发现过拟合可以调大至 0.2）

```

### 2.4 数据类型支持

YAML 会自动识别常见的数据类型，无需加引号（除非字符串中含有特殊字符）：

* **整数/浮点数**：`d_model: 128`, `learning_rate: 0.001`

* **布尔值**：`use_pe: true` 或 `use_pe: false`（全部小写）

* **字符串**：`name: "baseline"`（可用双引号、单引号或不用引号）

* **列表/序列**：

  ```yaml

  # 方式 A（单行，流式）

  devices: [0, 1, 2]

  # 方式 B（多行，以短横线开头并加空格）

  devices:

    - 0

    - 1

    - 2

  ```
---

## 3. 实战：如何使用 YAML 进行调参实验？

假设想开展一项**“验证将注意力层数增加到 4 层是否能降低验证 Loss”**的实验。以下是规范的实验步骤：

### 第一步：新建配置文件

在 `configs/` 目录下创建一个名为 `experiment_4layers.yaml` 的文件，内容只需填写需要修改的参数（得益于 `_deep_update` 机制）：

```yaml

# configs/experiment_4layers.yaml

model:

  num_layers: 4       # 从默认的 2 层修改为 4 层

  

experiment:

  name: "4layers"

  output_dir: "experiments/4layers"

```

### 第二步：运行训练

在命令行中，通过 `--config` 指向该文件：

```bash

python -m src.train --config configs/experiment_4layers.yaml --device cuda

```

### 第三步：对比实验结果

训练完成后，可以直接在 `experiments/baseline/` 和 `experiments/4layers/` 下找到它们各自的 `config.yaml` 和 `train_log.json`，并利用画图脚本或工具对比它们的验证 Loss 曲线。