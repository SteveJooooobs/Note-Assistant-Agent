# Python 导包机制详解

## 前置：读这篇文章之前

如果你正在疯狂尝试 `from ..xxx import yyy`、`sys.path.append("../../")`、`PYTHONPATH=. python xxx.py`，但导入还是一直炸——停下来，花 5 分钟读完这篇文章。Python 的导入机制只有两个核心概念：`sys.path` 和 `__package__`，理解了它们，所有导入问题都会消失。

---

## 1. Python 如何找到一个模块

当你写 `from X.Y import Z`，Python 做的事情非常简单：

> 在 `sys.path` 的**每一个目录**下，查找 `X/Y.py` 或 `X/Y/__init__.py`。

`sys.path` 是一个字符串列表，启动时自动填充：

```
sys.path = [
    '<脚本文件所在的目录>',      # 直接运行 python foo/bar.py 时有
    '',                         # 当前工作目录 CWD
    '<site-packages>/',         # 第三方库
    ...                         # PYTHONPATH 环境变量
]
```

这就是全部。没有任何魔法。你之所以调不通，是因为你的**项目根目录**不在这几个路径里。

---

## 2. 三种运行方式，三种 sys.path

假设项目结构如下：

```
/home/you/myproject/             ← 项目根目录（包含顶层包 mypkg/）
├── mypkg/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── engine.py            ← 你要运行的文件
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           ← 你要导入的目标
└── scripts/
    └── run.py
```

工作目录是 `/home/you/myproject/`。

### 方式 A：直接运行脚本 `python mypkg/core/engine.py`

```
sys.path[0] = '/home/you/myproject/mypkg/core/'   ← 脚本所在目录
sys.path[1] = ''                                  ← CWD = /home/you/myproject/
__package__  = None                               ← 脚本模式，不属于任何包
```

- `from mypkg.utils.helpers import xxx` → **能成功**（CWD 是项目根，里面有 `mypkg/`）
- `from ..utils.helpers import xxx` → **必炸**（`__package__` = None，Python 不知道"上一级"是什么）
- `from utils.helpers import xxx` → **必炸**（`mypkg/core/` 和 CWD 下都没有 `utils/` 目录）

### 方式 B：模块方式运行 `python -m mypkg.core.engine`

```
sys.path[0] = ''                                  ← CWD = /home/you/myproject/
__package__  = 'mypkg.core'                       ← 模块模式，知道自己的包位置
```

- `from ..utils.helpers import xxx` → **能成功**（`..` = `mypkg`）
- `from mypkg.utils.helpers import xxx` → **能成功**（项目根在 sys.path）

### 方式 C：被别的模块 import（`from mypkg.core.engine import Engine`）

```
sys.path 取决于调用方的运行方式
__package__  = 'mypkg.core'
```

相对导入和绝对导入都能工作，只要 sys.path 包含项目根。

---

## 3. 一张表看清楚

| 运行方式 | 相对导入 `from ..utils` | 绝对导入 `from mypkg.utils` |
|----------|------------------------|-----------------------------|
| `python mypkg/core/engine.py` | **炸**（`__package__`=None） | 能跑（CWD 含 `mypkg/`） |
| `python -m mypkg.core.engine` | 能跑 | 能跑 |
| 被其他模块 import | 能跑 | 能跑 |

**相对导入只能在模块模式下使用。** 一旦用脚本方式运行（`python xxx.py`），`__package__` 是 None，任何 `from ..` 都会炸。

**绝对导入依赖 CWD。** `from mypkg.utils.helpers import xxx` 要求 CWD 恰好是项目根。如果你从别的目录运行，它就找不到 `mypkg/`。

---

## 4. 通用解决方案

在每一个需要跨子包导入的 `.py` 文件**最顶部**（在所有项目内导入之前），加一段：

```python
import sys
from pathlib import Path

# 计算项目根目录并注入 sys.path
# _PROJECT_ROOT = __file__ 向上 N 层到项目根，N = 文件在项目内的嵌套深度
_PROJECT_ROOT = Path(__file__).resolve().parents[N]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
```

`N` 的计算方法：

```
__file__                        →  mypkg/core/engine.py
__file__.parents[0] (=.parent)  →  mypkg/core/
__file__.parents[1]             →  mypkg/              ← 顶层包
__file__.parents[2]             →  myproject/          ← 项目根
```

以这个结构为例，`engine.py` 位于项目根下第 3 层（`mypkg/core/engine.py`），所以 `N = 2`（`parents[2]` 就是项目根）。

然后统一使用**绝对导入**：

```python
from mypkg.utils.helpers import helper_func
```

这样做的好处：
- `python mypkg/core/engine.py` → `_PROJECT_ROOT` 注入，能跑
- `python -m mypkg.core.engine` → 项目根本来就在 sys.path，重复插入无影响
- `pip install -e .` → 不需要这段也能跑，加了也无副作用

---

## 5. 常见错误速查表

| 错误信息 | 原因 | 解决 |
|----------|------|------|
| `ImportError: attempted relative import with no known parent package` | 用了 `from ..xxx` 但以脚本方式运行 | 改用绝对导入 + 注入项目根到 sys.path |
| `ModuleNotFoundError: No module named 'mypkg'` | 项目根不在 sys.path | sys.path 注入项目根 |
| `ModuleNotFoundError: No module named 'utils'` | 用了 `from utils import` 但 sys.path 里没有直接包含 `utils/` 的目录 | 改为 `from mypkg.utils import` |
| `SystemError: parent module 'xxx' not loaded` | 相对导入时父包没有正确加载 | 用绝对导入替代 |

---

## 6. 什么文件需要 sys.path 注入

**需要加的**：凡是 `from <顶层包>.xxx import yyy` 的文件，且可能被 `python path/to/file.py` 直接运行。

**不需要加的**：
- 叶子模块（不导入项目内其他模块，只导入标准库/第三方库）
- 顶层入口脚本（如 `scripts/run.py`，通常已经处理了 sys.path 或通过 `pip install -e .` 安装）
- `__init__.py` 如果只被 import 而非直接运行（但加了也无害）

---

## 7. 更长远的方案：可编辑安装

如果你不想在每个文件里加 sys.path 代码，项目规范化的方案是：

```bash
pip install -e .
```

这会把你的项目以"开发模式"安装到 site-packages，此后 `from mypkg.xxx import yyy` 在任何目录下都能工作。但这需要项目有 `pyproject.toml` 或 `setup.py`。

---

## 8. 一句话总结

> Python 导入的根源在 `sys.path`。**项目根目录必须在 sys.path 里**，`from <顶层包>.xxx` 才能工作。相对导入 `from ..xxx` 在脚本模式下必炸。最稳妥的方案：文件头部注入项目根路径 + 统一使用绝对导入。
