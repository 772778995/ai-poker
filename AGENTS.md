# AGENTS.md — AI 德州扑克自动化系统

> **仅用于学术研究与技术学习，请在本地模拟器或私人环境中使用。**
> 文档版本：v2.0 | 更新：2026-03

---

## 目录

1. [系统总览](#1-系统总览)
2. [目录结构](#2-目录结构)
3. [模块规范](#3-模块规范)
4. [核心库选型](#4-核心库选型)
5. [数据结构定义](#5-数据结构定义)
6. [开发阶段与优先级](#6-开发阶段与优先级)
7. [Agent 协作约定](#7-agent-协作约定)
8. [测试策略](#8-测试策略)

---

## 1. 系统总览

```
┌──────────────────────────────────────────────────────────────────┐
│                       main.py (主控循环)                          │
└───────────┬──────────────────────────────────────┬──────────────┘
            │                                      │
    ┌───────▼───────┐                    ┌─────────▼────────┐
    │  Vision 层     │                    │  Analytics 层    │
    │  屏幕捕获+识别  │                    │  日志+仪表盘      │
    └───────┬───────┘                    └──────────────────┘
            │ GameFrame (原始图像数据)
    ┌───────▼───────┐
    │   State 层    │
    │  结构化状态解析 │
    └───────┬───────┘
            │ GameState (结构化对象)
    ┌───────▼───────┐
    │  Engine 层    │◄── GTO 引擎（实时决策，<150ms）
    │  GTO决策引擎   │
    └───────┬───────┘
            │ Decision (fold/call/raise + amount)
    ┌───────▼───────┐      ┌──────────────────────┐
    │  Executor 层  │      │  LLM 辅助层 ★新增     │
    │  鼠标键盘执行  │      │  GLM/Kimi/MiniMax     │
    └───────┬───────┘      │  仅用于离线分析复盘    │
            │ (穿透)        └──────────────────────┘
    ┌───────▼───────┐
    │  Stealth 层   │  ← 流程跑通后激活
    │  反检测包装器  │
    └───────────────┘
```

### ⚠️ LLM 在本系统中的正确定位

| 用途 | 是否适合 LLM | 推荐方案 |
|------|-------------|---------|
| 实时决策（fold/call/raise） | ❌ 延迟 >2s，成本高 | `treys` + 蒙特卡洛 + GTO 查找表 |
| 翻前范围策略 | ❌ 不稳定 | CSV 范围文件（GTO Wizard 导出） |
| 对手风格分析（离线） | ✅ 适合 | GLM-4-Flash / Kimi（免费额度） |
| 会话复盘报告生成 | ✅ 适合 | 任意免费 LLM |
| 决策逻辑解释（调试） | ✅ 适合 | 任意免费 LLM |
| 牌面识别 / OCR | ❌ | YOLOv8 + EasyOCR |

**结论：** GLM-5 / MiniMax M2.5 / Kimi2.5 均可接入 **LLM 辅助层**（`engine/llm_advisor.py`），为对手建模提供自然语言描述，并在会话结束后生成复盘报告。三者 API 均兼容 OpenAI SDK 格式，配置 `base_url` 即可切换，互为备份。

**主循环节奏：** 截图 → 识别 → 解析 → 决策 → 执行，每轮约 200–500ms。LLM 调用在独立线程中异步执行，不阻塞主循环。

---

## 2. 目录结构

```
poker-ai/
├── AGENTS.md
├── main.py                     # 主控循环入口
├── config.yaml                 # 全局配置
├── requirements.txt
│
├── vision/
│   ├── capture.py              # 屏幕捕获（mss）
│   ├── detector.py             # 牌面/UI 元素检测（OpenCV + YOLO）
│   └── ocr.py                  # 数字/文字识别（EasyOCR）
│
├── state/
│   ├── models.py               # Pydantic 数据模型（GameState/Decision/...）
│   ├── parser.py               # 将 Vision 输出解析为 GameFrame
│   └── tracker.py              # 跨轮状态追踪
│
├── engine/
│   ├── equity.py               # 胜率计算（treys + 蒙特卡洛）
│   ├── gto.py                  # GTO 策略（preflop 查找表 + postflop 规则）
│   ├── opponent.py             # 对手建模（VPIP/PFR/AF 统计）
│   └── llm_advisor.py          # ★ LLM 辅助层（离线分析，非实时）
│
├── executor/
│   ├── controller.py           # 动作执行总调度
│   └── mouse.py                # 鼠标操作（PyAutoGUI 封装）
│
├── stealth/                    # ⚠️ 流程跑通后再实现
│   ├── human_mouse.py          # 贝塞尔曲线鼠标轨迹（pyclick）
│   ├── timing.py               # 人类决策时间分布模拟
│   ├── session.py              # 会话管理
│   └── fingerprint.py          # 进程/内存特征隐藏
│
├── analytics/
│   ├── logger.py               # 手牌历史记录（SQLite + SQLAlchemy）
│   ├── hud.py                  # 实时 HUD 数据聚合
│   └── dashboard.py            # Streamlit 收益看板
│
├── data/
│   ├── preflop_ranges/         # 翻前范围 CSV（RFI/3bet/4bet）
│   ├── models/                 # YOLO 模型权重 (.pt)
│   ├── templates/              # OpenCV 模板匹配图片（fallback）
│   └── hands.db                # SQLite 手牌数据库
│
└── tests/
    ├── test_vision.py
    ├── test_equity.py
    ├── test_gto.py
    └── fixtures/               # 测试用截图样本
```

---

## 3. 模块规范

### 3.1 视觉感知层 — Vision

**职责：** 捕获屏幕，输出结构化的原始视觉数据（牌面 ID、数字、按钮位置）。

#### `vision/capture.py`

```python
class ScreenCapture:
    """
    捕获扑克客户端窗口区域。
    - 支持多显示器、指定窗口句柄（win32gui / Xlib）
    - grab_all_regions()：一次截图切分所有 ROI，减少重复调用
    - 输出：numpy.ndarray (BGR)
    """
    def grab(self) -> np.ndarray: ...
    def grab_region(self, region: tuple[float,float,float,float]) -> np.ndarray: ...
    def grab_named_region(self, name: str) -> np.ndarray | None: ...
    def grab_all_regions(self) -> dict[str, np.ndarray]: ...
```

**关键配置项（config.yaml）：**
```yaml
capture:
  monitor_index: 1
  window_title: "PokerTH"
  fps_limit: 8
  regions:                        # 预定义 ROI（相对坐标，0.0-1.0）
    hole_cards:      [0.42, 0.72, 0.58, 0.88]
    community_cards: [0.28, 0.42, 0.72, 0.58]
    pot:             [0.44, 0.38, 0.56, 0.44]
    stack:           [0.44, 0.88, 0.56, 0.94]
    action_buttons:  [0.30, 0.90, 0.70, 0.98]
```

#### `vision/detector.py`

```python
class CardDetector:
    """
    主路径：YOLOv8（poker-cards 预训练权重，支持 52 张牌分类）
    备用路径：OpenCV 模板匹配（YOLO 权重缺失时自动 fallback）
    输出格式："Ah", "Kd", "2c" ...（按 x 坐标从左到右）
    """
    def detect_cards(self, img: np.ndarray) -> list[str]: ...
    def detect_buttons(self, img: np.ndarray) -> DetectedButtons: ...
```

**YOLO 权重来源：**
- [roboflow/playing-cards](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d) 预训练模型
- 或使用 `labelImg` 自行标注 ~500 张样本训练

#### `vision/ocr.py`

```python
class PokerOCR:
    """
    提取底池、筹码量、下注额等数字信息。
    预处理：灰度化 → 自适应阈值 → 去噪 → 放大（提高小字识别率）
    后处理：正则匹配 $/BB 单位，统一转换为 float
    """
    def read_number(self, img: np.ndarray) -> float | None: ...
    def read_player_action(self, img: np.ndarray) -> str | None: ...
    def read_all(self, imgs: dict[str, np.ndarray]) -> dict[str, float | None]: ...
```

---

### 3.2 状态解析层 — State

**职责：** 将 Vision 层输出组合为完整、一致的 `GameState` 对象，负责跨帧状态追踪。

#### `state/models.py`（核心数据结构）

```python
class GameState(BaseModel):
    # 牌局信息
    hand_id: str                    # 每手牌唯一（UUID 短码）
    street: Street                  # preflop/flop/turn/river
    hole_cards: list[str]           # ["Ah", "Kd"]
    community_cards: list[str]      # ["Jc", "Ts", "9h"]

    # 筹码信息
    pot: float
    to_call: float                  # 0 = 可以 check
    min_raise: float
    max_raise: float

    # 玩家信息
    players: list[Player]
    hero_seat: int
    button_seat: int
    is_hero_turn: bool

    # 计算属性
    @property
    def pot_odds(self) -> float: ...   # 最低盈亏平衡胜率
    @property
    def spr(self) -> float: ...         # Stack-to-Pot Ratio

    # 元信息
    confidence: float               # 整体识别置信度（< 0.85 则跳过）

class Decision(BaseModel):
    action: Action
    amount: float
    equity: float
    ev: float
    confidence: float
    reasoning: str
    complexity: str                 # simple|medium|hard（影响 Stealth 思考时间）
```

#### `state/tracker.py`

```python
class StateTracker:
    """
    - 检测街道切换（preflop → flop → turn → river）
    - 维护手牌编号，检测新手牌（hole_cards 变化）
    - 缓存上一帧状态，识别失败时保持旧状态（最多 5 帧）
    - 检测英雄行动轮（Action Buttons 出现 = is_hero_turn=True）
    """
```

---

### 3.3 决策引擎层 — Engine

**职责：** 接收 `GameState`，输出最优 `Decision`。

#### `engine/equity.py`

```python
class EquityCalculator:
    """
    蒙特卡洛胜率模拟（treys 手牌评估）。
    - calculate()：10,000 次模拟，约 20ms，精度 ±0.5%
    - calculate_fast()：1,000 次，约 2ms，用于初筛
    - hand_class()：返回手牌类别（"Flush", "Two Pair" 等）
    """
```

#### `engine/gto.py`

```python
class GTOStrategy:
    """
    翻前：查找表驱动（position × hand_group → 混合策略）
    翻后：基于以下规则
      1. 弃牌赔率 = to_call / (pot + to_call)
      2. equity > pot_odds + 0.15 → raise
      3. equity > pot_odds → call
      4. equity ≤ pot_odds → fold
      5. 下注尺度由 SPR + 对手类型决定（0.33x/0.5x/0.67x/1.0x pot）
    """
    def decide(self, state: GameState, equity: float, opponent_type: PlayerType) -> Decision: ...
```

#### `engine/opponent.py`

```python
class OpponentModel:
    """
    HUD 核心统计：VPIP / PFR / AF / 3-bet% / Fold to C-bet
    玩家类型：Fish / TAG / LAG / Nit / Unknown
    """
    def update(self, seat: int, action: Action, state: GameState): ...
    def get_player_type(self, seat: int) -> PlayerType: ...
    def get_stats(self, seat: int) -> dict: ...    # 供 LLM 分析使用
```

#### `engine/llm_advisor.py` ★

```python
class LLMAdvisor:
    """
    LLM 辅助分析层（⚠️ 不参与实时决策路径）。

    支持的提供商（均兼容 OpenAI SDK 格式，修改 base_url 即可切换）：
    - 智谱 AI GLM-4-Flash：https://open.bigmodel.ai/api/paas/v4/
    - MiniMax M2.5：https://api.minimax.chat/v1
    - Kimi / Moonshot：https://api.moonshot.cn/v1

    功能：
    - opponent_summary()：根据 HUD 统计生成对手风格描述（100字）
    - session_report()：会话结束后生成复盘报告（200字）
    - explain_decision()：解释某手牌决策逻辑（调试用）

    调用方式：异步线程池，超时 10s，失败静默返回 None。
    """
    def opponent_summary(self, seat: int, stats: dict) -> str | None: ...
    def session_report(self, session_data: dict) -> str | None: ...
    def explain_decision(self, state, reasoning, equity, ev) -> str | None: ...
```

**配置示例（config.yaml）：**
```yaml
llm:
  enabled: true
  provider: "zhipuai"           # zhipuai | minimax | moonshot
  model: "glm-4-flash"          # 免费额度最大的选项
  api_key: ""                   # 或环境变量 POKER_AI_LLM_API_KEY
  timeout: 10
  use_for:
    - opponent_summary
    - session_report
```

---

### 3.4 执行控制层 — Executor

```python
class ActionController:
    """
    执行 fold / check / call / raise(amount) 动作。
    流程：
    1. 从 DetectedButtons 获取按钮坐标
    2. 若 raise：清空输入框 → 输入金额 → 点击 raise 按钮
    3. 执行后等待 UI 确认，最多重试 3 次
    """

class MouseController:
    """
    阶段一（调试）：pyautogui 直接点击
    阶段二（Stealth 激活后）：替换为 stealth/human_mouse.py
    接口保持稳定，上层无感知。
    """
    def click(self, x: int, y: int): ...
    def type_amount(self, amount: float): ...
    def select_all(self): ...
```

---

### 3.5 反检测层 — Stealth（流程跑通后实现）

> ⚠️ **本层在主流程完全跑通、胜率可验证后再开发。**
> 激活方式：`config.yaml` 中 `stealth.enabled: true`，无需修改其他代码。

#### 检测维度与对抗策略

| 检测维度 | 平台手段 | 对抗方案 |
|---------|---------|---------|
| 鼠标轨迹 | 直线移动、像素级精准 | pyclick 贝塞尔曲线 + 高斯抖动 |
| 决策时间 | 过快/过规律 | 对数正态分布延迟 |
| 操作节律 | 手牌间隔均匀 | 泊松过程模拟 |
| 在线时长 | 全天候不中断 | 会话管理 + 强制休息 |
| 进程扫描 | 已知 bot 进程名 | 进程名伪装 |
| 点击精度 | 总点按钮中心 | 落点高斯偏移（σ=3px） |

#### `stealth/timing.py`

```
决策延迟分布：
- 简单决策（confidence > 0.9）：对数正态 μ=1.2s, σ=0.4s
- 中等决策（0.7-0.9）：μ=3.5s, σ=1.2s
- 困难决策（< 0.7）：μ=7.0s, σ=2.5s
- 2% 概率触发 15-40s 超长停顿（模拟分心）
- Decision.complexity 字段由 Engine 层设置，Stealth 层消费
```

#### `stealth/session.py`

```
会话计划（可配置）：
- 单次会话：90–180 分钟
- 会话间休息：30–90 分钟
- 每日上限：6 小时
- 微休息：每 20-40 分钟 AFK 2-8 分钟
```

---

### 3.6 数据记录层 — Analytics

```python
class HandLogger:
    """SQLite 记录每手牌：手牌/公共牌/底池/行动/胜率/EV/理由"""

# analytics/dashboard.py
# streamlit run analytics/dashboard.py
# 看板指标：收益曲线 / 位置盈亏 / EV 泄露热力图 / 对手统计
```

---

## 4. 核心库选型

| 功能 | 选型 | 理由 |
|------|------|------|
| 屏幕捕获 | `mss` | 速度最快，跨平台 |
| 图像处理 | `opencv-python` | 模板匹配 + 预处理 |
| 目标检测 | `ultralytics` (YOLOv8) | 预训练扑克牌模型可用 |
| OCR | `easyocr` | 数字识别优于 Tesseract |
| 手牌评估 | `treys` | 纯 Python 最快 7 张牌评估器 |
| 数据模型 | `pydantic` v2 | 强类型验证 + 自动序列化 |
| 鼠标控制 | `pyautogui` → `pyclick` | 阶段一简单，阶段二换贝塞尔 |
| 数据存储 | `sqlalchemy` + SQLite | 零运维，结构化查询 |
| LLM 接入 | `openai` SDK | 兼容 GLM/Kimi/MiniMax 格式 |
| 日志 | `loguru` | 简洁，支持滚动文件 |
| 可视化 | `streamlit` + `plotly` | 零前端代码 |
| 测试 | `pytest` + `pytest-mock` | 标准，支持 fixture |
| 数值计算 | `numpy` | 蒙特卡洛加速 |

---

## 5. 数据结构定义

### 数据流（类型化）

```
Vision 层输出：
  GameFrame {hole_cards, community_cards, pot, stacks, bets, buttons, confidence}
      ↓ StateTracker.update()
State 层输出：
  GameState {street, hole_cards, community_cards, pot, to_call, players, spr, pot_odds, ...}
      ↓ GTOStrategy.decide()
Engine 层输出：
  Decision {action, amount, equity, ev, complexity, reasoning}
      ↓ ActionController.execute()
Executor 层：
  鼠标点击（click fold/call/raise）

Analytics 层（旁路记录）：
  HandRecord → SQLite → Streamlit 看板
  OpponentStats → LLMAdvisor → 自然语言对手描述
```

### Decision.complexity 与 Stealth 的联动

```python
# Engine 层设置（gto.py）
decision.complexity = "simple"   # equity > 0.80 或明显弃牌
decision.complexity = "medium"   # 正常价值/保护性跟注
decision.complexity = "hard"     # 薄价值/诈唬/多路底池

# Stealth 层消费（timing.py）
delay = timing.get_delay(decision.complexity)
```

---

## 6. 开发阶段与优先级

### Phase 1：视觉 + 状态（Week 1-3）

**目标：** 准确读取任意一手牌的完整状态。

- [ ] `vision/capture.py` — mss 截图，稳定帧率
- [ ] `vision/detector.py` — YOLOv8 识别 52 张牌（准确率 > 95%）
- [ ] `vision/ocr.py` — EasyOCR 提取底池/筹码数字（误差 < 1%）
- [ ] `state/models.py` — 完整 Pydantic 模型（已完成）
- [ ] `state/parser.py` — 组装 GameFrame，置信度校验
- [ ] `state/tracker.py` — 跨帧状态稳定性

**验收标准：** 100 张截图样本，GameState 准确率 > 92%，控制台输出正确 JSON。

---

### Phase 2：决策引擎（Week 4-6）

**目标：** 对任意合法 GameState 给出有理有据的决策。

- [ ] `engine/equity.py` — 蒙特卡洛胜率（10k 次 < 50ms）（已完成）
- [ ] `engine/gto.py` — 翻前查找表 + 翻后规则（已完成）
- [ ] `engine/opponent.py` — VPIP/PFR 统计追踪（已完成）
- [ ] 收集翻前范围 CSV 数据（data/preflop_ranges/）
- [ ] 单元测试：对比 GTO Wizard 参考答案，吻合率 > 80%

**验收标准：** PokerTH 中手动输入 20 个场景，决策准确率与 GTO 参考 > 80% 吻合。

---

### Phase 3：端到端联调（Week 7-8）

**目标：** 在本地模拟器中完整运行主循环，无人工干预打完一局。

- [ ] `executor/controller.py` — 接入 PyAutoGUI（已完成框架）
- [ ] `main.py` — 主循环完整联调（已完成）
- [ ] `state/parser.py` — 补全 players/stack/to_call 字段解析
- [ ] `state/tracker.py` — 补全多玩家状态追踪
- [ ] `analytics/logger.py` — SQLite 入库（已完成）
- [ ] 异常处理：识别失败 / 决策超时 / 执行失败 fallback

**验收标准：** PokerTH 中连续运行 50 手无崩溃，胜率统计可信。

---

### Phase 4：LLM 辅助层接入（Week 9，可与 Phase 3 并行）

**目标：** 接入免费 LLM，提供对手分析与复盘报告。

- [ ] `engine/llm_advisor.py` — 接口已完成，填入真实 API Key 测试
- [ ] 配置 `config.yaml` 中的 `llm` 部分
- [ ] `analytics/hud.py` — 整合 LLM 对手描述到 HUD
- [ ] `analytics/dashboard.py` — Streamlit 看板展示复盘报告

**LLM 提供商选择建议：**
- 优先：**智谱 AI GLM-4-Flash**（免费额度大，响应快）
- 备选：**Kimi Moonshot**（中文理解好）
- 备选：**MiniMax M2.5**（推理能力强）

---

### Phase 5：对手建模优化（Week 10）

- [ ] `engine/opponent.py` — 完整 HUD（AF / 3-bet% / Fold to C-bet）
- [ ] 玩家类型识别并调整策略权重
- [ ] 对鱼型玩家增大价值下注频率
- [ ] 对 Nit 型玩家减少三赌虚张

---

### Phase 6：Stealth 反检测层（Week 11-13，流程稳定后）

- [ ] `stealth/human_mouse.py` — pyclick 贝塞尔曲线
- [ ] `stealth/timing.py` — 对数正态决策延迟，消费 Decision.complexity
- [ ] `stealth/session.py` — 会话管理 + 微休息调度
- [ ] `stealth/fingerprint.py` — 进程伪装（按需）
- [ ] A/B 测试：Stealth 开/关时节奏对比

---

### Phase 7：看板与复盘（Week 14）

- [ ] `analytics/dashboard.py` — Streamlit 实时看板
- [ ] HH 格式导出（兼容 PT4 / HM3）
- [ ] LLM 自动生成 EV 泄露报告

---

## 7. Agent 协作约定

### 接口契约

- 公开接口（类名 + 方法签名）**只能新增，不能修改**
- 破坏性修改需先在 AGENTS.md 中提 RFC，review 后执行
- 接口变更必须同步更新 type stub（`*.pyi`）

### 数据流方向（严格单向）

```
Vision → State → Engine → Executor
                   ↓
               Analytics ← LLMAdvisor（异步旁路）
```

Engine 层**不得**持有 Vision/State 的引用，只接受函数参数。
LLMAdvisor 通过 `get_stats()` 接口从 OpponentModel 读取数据，不直接访问 State 层。

### 错误处理约定

```python
# 每层定义自己的异常类
class VisionError(Exception): ...
class StateError(Exception): ...
class EngineError(Exception): ...

# 主循环捕获并记录，不中断
try:
    state = tracker.update(frame)
except StateError as e:
    logger.warning(f"状态解析失败，跳过本帧: {e}")
    continue

# LLM 调用：失败静默返回 None，不抛出异常
result = llm.opponent_summary(seat, stats)  # None 表示跳过
```

### 配置优先级

`config.yaml` < 环境变量 (`POKER_AI_*`) < 命令行参数

```bash
# 环境变量示例
export POKER_AI_LLM_API_KEY="your_api_key_here"
export POKER_AI_LLM_PROVIDER="moonshot"
```

### 日志规范

```python
logger.info("[Vision] 识别到手牌: Ah Kd, 置信度: 0.97")
logger.debug("[Engine] Equity: 0.64, EV: +2.3BB → raise 67% pot")
logger.warning("[State] OCR 置信度过低 (0.71), 保持上一帧状态")
logger.error("[Executor] 点击失败，按钮坐标未找到: fold")
logger.info("[LLM] 对手 seat=3 分析完成: 典型鱼型玩家")
```

---

## 8. 测试策略

### 单元测试

```
tests/
├── test_vision.py      # 用 fixtures/ 截图测试识别准确率
├── test_equity.py      # 对比已知胜率（精度 < 0.5%）
├── test_gto.py         # 对比 GTO Wizard 参考决策
├── test_llm.py         # Mock LLM API，测试 prompt 格式
└── test_stealth.py     # 鼠标轨迹人类相似度评分
```

### 集成测试

- **PokerTH**（开源）作为主要测试环境
- 录制 50 手牌屏幕视频，回放测试 Vision + State 层
- 定期 1000 手蒙特卡洛模拟验证引擎 EV 为正

### 性能基准

| 组件 | 目标延迟 |
|------|---------|
| 截图 (mss) | < 5ms |
| 牌面识别 (YOLO) | < 30ms |
| OCR | < 50ms |
| 胜率计算 (10k 次) | < 50ms |
| GTO 决策 | < 10ms |
| **决策总延迟** | **< 150ms** |
| LLM 调用（异步，不计入主循环） | < 10s |

---

## 附录：参考资料

- **treys**：https://github.com/ihendley/treys
- **YOLOv8 扑克牌数据集**：https://universe.roboflow.com/augmented-startups/playing-cards-ow27d
- **pyclick（贝塞尔鼠标）**：https://github.com/patrikoss/pyclick
- **EasyOCR**：https://github.com/JaidedAI/EasyOCR
- **翻前范围数据**：https://www.pokercoaching.com/ranges
- **GTO 参考**：PioSOLVER / GTO+ 导出策略文件
- **智谱 AI API**：https://open.bigmodel.ai/dev/api
- **Kimi API**：https://platform.moonshot.cn/docs
- **MiniMax API**：https://platform.minimaxi.com/document/guides/chat-model/V2
