# AI 辅助学习能力演化仿真

本仓库保存两能力系统动力学模型的 Python 源码、仿真数据和图像结果。`h_1` 表示撤去 AI 后的独立能力，`h_2` 表示与 AI 协作、核验和整合输出的能力；二者是人的能力指数，范围为 `[0,1]`。时间 `t` 是未标定的**有效学习时间**。全部参数均为示意值，结果是确定性数值仿真，**不是学生实测数据或教学效果估计**。

## 1. 模型与边界

令 `u` 为分配给 AI 辅助活动的时间比例，`1-u` 为独立活动比例；每次仿真中 `u` 固定。模型为

$$
\begin{aligned}
\dot h_1 &= (1-h_1)\bigl[a(1-u)+\beta u h_2\bigr]-\delta_1h_1,\\
\dot h_2 &= cu\bigl[\varepsilon+(1-\varepsilon)h_1\bigr](1-h_2)-\delta_2h_2.
\end{aligned}
$$

第一式区分独立练习形成与 AI 输出经理解、核验后的内化形成；第二式描述协作技能形成。`\varepsilon+(1-\varepsilon)h_1` 是协作形成的支持条件，`\varepsilon` 表示在独立能力为零时仍存在的基础支持。衰减项在模型时间内持续作用。代码用四阶 Runge–Kutta（RK4）积分，不对状态做数值截断；输入越界或状态越界时会报错。

| 参数 | 代码键 | 基准值 | 含义 |
|---|---|---:|---|
| `a` | `a` | 0.25 | 独立学习形成率 |
| `β` | `beta` | 0.48 | AI 知识内化形成率 |
| `c` | `c` | 0.55 | AI 协作能力形成率 |
| `ε` | `eps` | 0.10 | 基础协作支持水平，取值 `[0,1]` |
| `δ₁` | `delta1` | 0.02 | 独立能力衰减率 |
| `δ₂` | `delta2` | 0.03 | AI 协作能力衰减率 |

基准初值为 `h₁(0)=0.20`、`h₂(0)=0.10`，常规观察期限为 `T=20`。六个速率参数的单位均为模型时间的倒数；`ε`、`u`、`h₁`、`h₂` 无量纲。`h₃=w h₁+(1-w)h₂` 仅是评价指标，**不是第三个微分方程或能力存量**。`w` 是评价权重，不进入动力学；它也不是 AI 使用比例 `u`。

## 2. 目录与代码职责

| 路径 | 功能 |
|---|---|
| `code/model.py` | 定义参数、方程右端、输入检查、向量化 RK4 积分、状态边界审计与解析特例。 |
| `code/experiments.py` | 运行八组实验，导出完整轨迹、终点、参数效果及数值核验结果。 |
| `code/sensitivity_analysis.py` | 主入口；默认重算数据、八幅图和两份文字报告；`--data-only` 只重算数值数据。 |
| `code/plot_figures.py`、`code/tick_display.py` | 将实验结果绘成八组 PNG、PDF、SVG；统一坐标与刻度，并触发图件检查。 |
| `code/figure_qa/` | 图形源码预检、PDF 文字检查、面板对齐与元素碰撞审计。 |
| `code/extend_joint_horizons.py` | 对四个期限的 `u×β` 联合扫描做轨迹及独立求解器复核。 |
| `code/h3_horizon_analysis.py` | 在固定 `u` 的轨迹上计算 `h₃`、期限对应的最优网格 `u` 和权重敏感性。 |
| `LATEX文档/scripts/draw_h3_horizons.py` | 读取 `h₃` 保存结果，绘制期限分析图；此处仅收录该绘图脚本及其图像，不包含论文。 |
| `preview_h3_3d_20261102/generate_preview.py` | 生成六联三维图、独立面板、离线交互 HTML 和对应轨迹数据。 |
| `preview_h3_3d_20261102/reproduction/` | 三维预览使用的模型与面板检查代码快照。它与主模型的动力学方程相同，但保留预览当时的常量集合。 |
| `code/redraw_saved_figures.py` | 从已保存 CSV 重绘八幅主图，不重新求解方程；论文图件同步部分需要完整原始项目目录。 |
| `code/write_report.py`、`code/parameter_definitions.json` | 将结果摘要写入文字报告的辅助代码与参数词表；主入口默认调用。 |
| `results/data/` | 主实验的轨迹、终点及联合扫描 CSV。CSV 使用 UTF-8 with BOM。 |
| `results/settings.json`、`results/summary.json`、`results/numerical_checks.json` | 运行配置、供图文共用的摘要和数值检查。 |
| `results/h3_horizon/` | `h₃` 的 CSV、压缩 NumPy 数组、摘要及数值检查。 |
| `images/` | 八组主实验图，每组提供 PNG、PDF 和 SVG。 |
| `LATEX文档/figures/09_h3_horizons.*` | `h₃` 期限图的 PNG、PDF、SVG、TIFF。 |
| `preview_h3_3d_20261102/` | 三维预览图、六张单独面板、离线交互 HTML 与数据。 |

`code/` 中用于 PPT、论文归档和跨文档同步的脚本没有进入此包；它们依赖本包以外的版式文件，不属于仿真计算。`output/` 的幻灯片渲染图、LaTeX 成稿、用户提供的原始示意图和历史备份也未纳入。

## 3. 环境与复现

保存结果记录的环境为 Python 3.11.7、NumPy 1.26.4、SciPy 1.15.1、Matplotlib 3.8.0、PyMuPDF 1.26.5。建议先在解压目录创建虚拟环境。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code\requirements.txt
.\.venv\Scripts\python.exe -B code\sensitivity_analysis.py
```

macOS / Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python -B code/sensitivity_analysis.py
```

只重算主实验数据时使用 `python -B code/sensitivity_analysis.py --data-only`；它不会更新图，不能把新数据与旧图当作同一次运行。完整运行会覆盖 `results/`、`images/` 的对应生成文件，并在根目录写出 `汇报交接总结.md`、`汇报要点.md`。若只需查看已有结果，无需运行脚本。

更多复现入口按顺序使用：

```powershell
python -B code\extend_joint_horizons.py
python -B code\h3_horizon_analysis.py
python -B LATEX文档\scripts\draw_h3_horizons.py
python -m pip install -r preview_h3_3d_20261102\requirements.txt
python -B preview_h3_3d_20261102\generate_preview.py
```

联合扫描与三维图会消耗较多时间和内存；字体替换也可能改变图件排版。三维预览可直接在浏览器中打开 `preview_h3_3d_20261102/h3_3d_interactive.html`，所需 Plotly JavaScript 已写入 HTML。`draw_h3_horizons.py` 默认读取 `results/h3_horizon/fixed_policy_trajectories.npz`，所以应先运行 `h3_horizon_analysis.py`。

## 4. 实验和图像索引

| 图号 | 文件名前缀 | 主要比较 |
|---|---|---|
| 1 | `01_time_allocation` | AI 时间比例 `u` |
| 2 | `02_internalization_strength` | 内化形成率 `β` |
| 3 | `03_mechanism_trajectories` | 六个参数分别上下扰动 20% 的过程 |
| 4 | `04_dynamic_coupling` | 两条跨能力动态输入的冻结诊断 |
| 5 | `05_observation_horizons` | 不同观察期限与全程独立活动基准 |
| 6 | `06_initial_collaboration` | 协作能力初值 `h₂(0)` |
| 7 | `07_local_sensitivity` | 六参数局部扰动对终点的跨度 |
| 8 | `08_joint_parameter_surface` | `u×β` 联合扫描 |
| 9 | `09_h3_horizons` | `h₁`、`h₂`、`h₃` 与各期限的固定比例选择 |
| 三维预览 | `h3_weights_beta_epsilon_3d` | `β`、`ε` 情景下的 `h₃(T,w)`；六面板中有一组故意重复的基准 |

图 1–8 位于 `images/`；图 9 位于 `LATEX文档/figures/`；三维预览位于 `preview_h3_3d_20261102/figures/`。PNG 适合直接查看，PDF/SVG 适合矢量编辑；图 9 与三维总图另有 TIFF。`preview_h3_3d_20261102/panels/` 保存六张单独面板。图中的曲线均为模型轨迹，没有抽样误差线、置信区间或实证显著性检验。

## 5. 校验与解释限制

主积分步长 `0.01`，减半复核步长 `0.005`，数值容差 `1e-7`。`results/numerical_checks.json` 记录解析特例、状态边界、步长减半和独立 DOP853 求解器比较；其 `all_checks_passed` 应为 `true`。`results/h3_horizon/summary.json` 记录 `h₃` 的独立校验。重新运行后请核对这两个文件及图件 QA 输出。

这些检查支持**代码数值一致性**，不能证明现实因果机制成立。当前没有经验证的能力量表，形成率与衰减率未用学习者数据标定；`T` 也不能直接换算为真实天数或学时。局部敏感性排名取决于基准点、扰动幅度和期限。图 9 的“最佳 `u`”只是在给定网格上为每个期限分别选出的**固定比例**，并非可随时间切换策略的全局最优。

## 6. 打包说明

仓库展示的是可浏览的源文件与结果；本地交付的 `仿真代码与图像结果.zip` 包含同一批文件和本 README。`MANIFEST.sha256` 逐项记录包内文件哈希，便于解压后核对。ZIP 不包含虚拟环境、缓存、原始 PPT、论文 PDF 或个人身份信息。
