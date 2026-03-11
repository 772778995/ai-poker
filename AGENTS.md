# AGENTS.md — AI 德州扑克自动化系统

> 本文档为 Agent 协作开发规范，描述系统架构、模块职责、库选型与实现顺序。
> **仅用于学术研究与技术学习，请在本地模拟器或私人环境中使用。**

---

## 目录

1. [系统总览](#1-系统总览)
2. [目录结构](#2-目录结构)
3. [模块规范](#3-模块规范)
   - 3.1 [视觉感知层 — Vision](#31-视觉感知层--vision)
   - 3.2 [状态解析层 — State](#32-状态解析层--state)
   - 3.3 [决策引擎层 — Engine](#33-决策引擎层--engine)
   - 3.4 [执行控制层 — Executor](#34-执行控制层--executor)
   - 3.5 [反检测层 — Stealth（流程跑通后实现）](#35-反检测层--stealth流程跑通后实现)
   - 3.6 [数据记录层 — Analytics](#36-数据记录层--analytics)
4. [核心库选型](#4-核心库选型)
5. [数据结构定义](#5-数据结构定义)
6. [开发阶段与优先级](#6-开发阶段与优先级)
7. [Agent 协作约定](#7-agent-协作约定)
8. [测试策略](#8-测试策略)

---

## 1. 系统总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (主控循环)                        │
└────────────┬──────────────────────────────────────┬────────────┘
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
     │  Engine 层    │
     │  GTO决策引擎   │
     └───────┬───────┘
             │ Action (fold/call/raise + amount)
     ┌───────▼───────┐
     │  Executor 层  │
     │  鼠标键盘执行  │
     └───────┬───────┘
             │ (穿透)
     ┌───────▼───────┐
     │  Stealth 层   │  ← 流程跑通后激活
     │  反检测包装器  │
     └───────────────┘
```

**主循环节奏：** 截图 → 识别 → 解析 → 决策 → 执行，每轮约 200–500ms（Stealth 层激活后动态调整）。

---

## 2. 目录结构

```
poker-ai/
├── AGENTS.md                   # 本文件
├── main.py                     # 主控循环入口
├── config.yaml                 # 全局配置（客户端坐标、阈值等）
├── requirements.txt
│
├── vision/
│   ├── __init__.py
│   ├── capture.py              # 屏幕捕获（mss）
│   ├── detector.py             # 牌面/UI 元素检测（OpenCV + YOLO）
│   └── ocr.py                  # 数字/文字识别（EasyOCR）
│
├── state/
│   ├── __init__.py
│   ├── parser.py               # 将 Vision 输出解析为 GameState
│   ├── models.py               # Pydantic 数据模型
│   └── tracker.py              # 跨轮状态追踪（底池、历史行动）
│
├── engine/
│   ├── __init__.py
│   ├── equity.py               # 胜率计算（treys + 蒙特卡洛）
│   ├── gto.py                  # GTO 策略（preflop 查找表 + postflop CFR）
│   ├── opponent.py             # 对手建模（统计 VPIP/PFR/AF）
│   ├── postflop.py             # 翻后决策（弃牌率 / 赔率 / EV）
│   └── ranges.py               # 范围矩阵定义与操作
│
├── executor/
│   ├── __init__.py
│   ├── controller.py           # 动作执行总调度
│   └── mouse.py                # 鼠标操作（PyAutoGUI 封装）
│
├── stealth/                    # ⚠️ 流程跑通后再实现此模块
│   ├── __init__.py
│   ├── human_mouse.py          # 贝塞尔曲线鼠标轨迹
│   ├── timing.py               # 人类决策时间分布模拟
│   ├── session.py              # 会话管理（在线时长/休息节律）
│   └── fingerprint.py          # 进程/内存特征隐藏
│
├── analytics/
│   ├── __init__.py
│   ├── logger.py               # 手牌历史记录（SQLite）
│   ├── hud.py                  # 实时 HUD 数据聚合
│   └── dashboard.py            # Streamlit 收益看板
│
├── data/
│   ├── preflop_ranges/         # 翻前范围 CSV（RFI/3bet/4bet）
│   ├── models/                 # YOLO 模型权重 (.pt)
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
# 使用库：mss（比 pyautogui 快 3-5×）
import mss

class ScreenCapture:
    """
    捕获扑克客户端窗口区域。
    - 支持多显示器
    - 支持指定窗口句柄（win32gui / Xlib）
    - 输出：numpy.ndarray (BGR)
    """
    def __init__(self, config: dict): ...
    def grab(self) -> np.ndarray: ...
    def grab_region(self, region: tuple) -> np.ndarray: ...
```

**关键配置项（config.yaml）：**
```yaml
capture:
  monitor_index: 1
  window_title: "PokerTH"        # 目标窗口标题，用于自动定位
  fps_limit: 8                    # 最大截图帧率
  regions:                        # 预定义 ROI（相对坐标，0.0-1.0）
    hole_cards: [0.42, 0.72, 0.58, 0.88]
    community_cards: [0.28, 0.42, 0.72, 0.58]
    pot: [0.44, 0.38, 0.56, 0.44]
    stack: [0.44, 0.88, 0.56, 0.94]
    action_buttons: [0.30, 0.90, 0.70, 0.98]
```

#### `vision/detector.py`

```python
# 使用库：ultralytics (YOLOv8) + OpenCV 模板匹配（备用）
from ultralytics import YOLO

class CardDetector:
    """
    识别牌面（花色 + 点数）。
    主路径：YOLOv8（poker-cards 预训练权重，支持 52 张牌分类）
    备用路径：OpenCV 模板匹配（离线环境 fallback）
    输出格式："Ah", "Kd", "2c" ...
    """
    def detect_cards(self, img: np.ndarray) -> list[str]: ...
    def detect_buttons(self, img: np.ndarray) -> dict[str, tuple]: ...
    # 返回 {"fold": (x,y), "call": (x,y), "raise": (x,y)}
```

**YOLO 权重来源：**
- 使用 [roboflow/poker-cards](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d) 预训练模型
- 或自行标注 ~500 张样本使用 `labelImg` 训练

#### `vision/ocr.py`

```python
# 使用库：EasyOCR（比 Tesseract 对数字识别更准，支持 GPU 加速）
import easyocr

class PokerOCR:
    """
    提取底池、筹码量、下注额等数字信息。
    - 预处理：灰度化 → 阈值化 → 去噪（提高 OCR 准确率）
    - 后处理：正则匹配 $/BB 单位，统一转换为 float
    """
    def read_number(self, img: np.ndarray) -> float | None: ...
    def read_player_action(self, img: np.ndarray) -> str | None: ...
    # 返回 "fold" / "call $5" / "raise $20" / "all-in" / None
```

---

### 3.2 状态解析层 — State

**职责：** 将 Vision 层输出组合为完整、一致的 `GameState` 对象，负责跨帧状态追踪。

#### `state/models.py`

```python
from pydantic import BaseModel
from enum import Enum

class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

class Action(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"

class Player(BaseModel):
    seat: int
    stack: float
    bet: float = 0.0
    is_active: bool = True
    last_action: Action | None = None

class GameState(BaseModel):
    street: Street
    hole_cards: list[str]           # ["Ah", "Kd"]
    community_cards: list[str]      # ["Jc", "Ts", "9h"]
    pot: float
    to_call: float
    players: list[Player]
    hero_seat: int
    button_seat: int
    is_hero_turn: bool
    confidence: float               # 识别置信度，< 0.85 则跳过本轮
```

#### `state/tracker.py`

```python
class StateTracker:
    """
    - 检测街道切换（preflop → flop → turn → river）
    - 维护手牌编号（每次发牌递增）
    - 缓存上一帧状态，识别失败时保持旧状态
    - 检测英雄行动轮（Action Buttons 出现 = is_hero_turn=True）
    """
```

---

### 3.3 决策引擎层 — Engine

**职责：** 接收 `GameState`，输出最优 `(Action, amount)` 决策。

#### `engine/equity.py`

```python
# 使用库：treys（纯 Python，极快的手牌强度评估）
from treys import Card, Evaluator, Deck

class EquityCalculator:
    """
    蒙特卡洛胜率模拟。
    - 默认 10,000 次模拟（约 20ms）
    - 输出：equity（胜率）、tie_rate（平局率）
    - 翻前使用预计算查找表（PokerCoach ranges.csv）
    """
    def calculate(
        self,
        hole_cards: list[str],
        board: list[str],
        num_opponents: int,
        simulations: int = 10_000,
    ) -> dict:
        # 返回 {"equity": 0.65, "tie": 0.02, "win": 0.63}
```

#### `engine/gto.py`

```python
class GTOStrategy:
    """
    翻前策略：查找表驱动
    - 数据来源：GTO Wizard / PokerSnowie 导出的 CSV 范围文件
    - 格式：position × hand_category → {fold%, call%, raise%}
    - 支持：RFI（首次加注）、3-bet、4-bet、Squeeze

    翻后策略：基于以下启发规则（简化 GTO）
    1. 弃牌赔率 (Fold Equity) = to_call / (pot + to_call)
    2. 若 equity > 赔率要求 → call/raise
    3. 下注尺度选择：0.33pot / 0.67pot / 1.0pot（混合频率）
    4. SPR（Stack-to-Pot Ratio）驱动深筹码决策
    """
    def decide_preflop(self, state: GameState) -> tuple[Action, float]: ...
    def decide_postflop(self, state: GameState, equity: float) -> tuple[Action, float]: ...
```

#### `engine/opponent.py`

```python
class OpponentModel:
    """
    追踪对手统计数据（HUD 核心指标）：
    - VPIP：自愿入底池率（衡量松/紧）
    - PFR：翻前加注率（衡量主动性）
    - AF：攻击系数（翻后主动性）
    - 3-bet%：3-bet 频率
    - Fold to C-bet%：面对持续下注的弃牌率

    玩家类型分类：
    - Fish（鱼）：VPIP > 40%, PFR < 10%  → 价值下注为主，减少虚张
    - TAG：VPIP 15-25%, PFR 12-20%       → 标准 GTO 应对
    - LAG：VPIP > 25%, PFR > 20%         → 偏向弃牌，等强牌
    - Nit（石头）：VPIP < 12%             → 对其加注时直接弃牌
    """
    def update(self, seat: int, action: Action, state: GameState): ...
    def get_player_type(self, seat: int) -> str: ...
    def get_fold_to_cbet(self, seat: int) -> float: ...
```

---

### 3.4 执行控制层 — Executor

**职责：** 将决策转化为实际的鼠标点击与键盘输入。

#### `executor/controller.py`

```python
class ActionController:
    """
    执行 fold / call / raise(amount) 动作。
    流程：
    1. 从 Vision 获取按钮坐标
    2. 若 raise：先清空输入框 → 输入金额 → 点击 raise 按钮
    3. 所有操作通过 Stealth 层包装（流程跑通前直接调用 mouse.py）
    4. 执行后等待 UI 状态更新确认（最多重试 3 次）
    """
    def execute(self, action: Action, amount: float, buttons: dict): ...
```

#### `executor/mouse.py`

```python
# 使用库：PyAutoGUI（阶段一）→ 替换为 stealth/human_mouse.py（阶段二）
import pyautogui

class MouseController:
    """
    阶段一（调试用）：直接点击，无延迟
    pyautogui.click(x, y)
    pyautogui.typewrite(str(amount))

    接口保持稳定，Stealth 层激活后直接替换实现，上层无感知。
    """
    def click(self, x: int, y: int): ...
    def type_amount(self, amount: float): ...
    def move_to(self, x: int, y: int): ...
```

---

### 3.5 反检测层 — Stealth（流程跑通后实现）

> ⚠️ **本层在主流程完全跑通、胜率可验证后再开发。**
> 所有接口已在 executor 层预留，激活 Stealth 只需修改 `config.yaml` 中的 `stealth.enabled: true`。

#### 检测原理与对抗策略

现代扑克平台的反作弊主要通过以下维度检测：

| 检测维度 | 平台手段 | 我们的对抗方案 |
|---------|---------|--------------|
| 鼠标轨迹 | 检测直线移动、瞬移、像素级精准 | 贝塞尔曲线 + 抖动噪声 |
| 决策时间 | 统计决策延迟分布（太快/太规律） | 正态分布随机延迟 |
| 操作节律 | 每手牌时间间隔过于均匀 | 泊松过程模拟 |
| 在线时长 | 全天候不间断（人不可能） | 会话管理 + 强制休息 |
| 进程扫描 | 扫描已知 bot 进程名/内存特征 | 进程伪装 + 内存加密 |
| 屏幕分辨率 | 检测截图 API 调用（部分客户端） | 虚拟机 / OBS 虚拟摄像头中转 |
| 点击精度 | 总是点击按钮正中心 | 点击位置高斯偏移 |

#### `stealth/human_mouse.py`

```python
# 使用库：pyclick（内置贝塞尔曲线人类鼠标模拟）
# pip install pyclick
from pyclick import HumanClicker

class HumanMouse:
    """
    使用贝塞尔曲线生成自然鼠标轨迹。

    参数控制：
    - speed：移动速度（秒），从 N(0.3, 0.08) 采样
    - click_offset：落点高斯偏移，σ=3px（模拟人手抖动）
    - 偶发 double-move：5% 概率轻微过冲后回正

    实现要点：
    clicker = HumanClicker()
    clicker.move((x + offset_x, y + offset_y), duration)
    clicker.click()
    """
    def click(self, x: int, y: int): ...
    def move_to(self, x: int, y: int): ...
```

#### `stealth/timing.py`

```python
import numpy as np

class HumanTiming:
    """
    决策延迟模拟策略：

    1. 基础延迟：从对数正态分布采样
       - 简单决策（明显弃牌/跟注）：μ=1.2s, σ=0.4s
       - 中等决策：μ=3.5s, σ=1.2s
       - 困难决策（薄价值/虚张）：μ=7.0s, σ=2.5s

    2. 决策复杂度由引擎返回的置信度决定：
       confidence > 0.9  → 简单
       0.7 < confidence ≤ 0.9 → 中等
       confidence ≤ 0.7  → 困难

    3. 偶发"思考停顿"：
       - 2% 概率触发 15-40s 超长停顿（模拟分心）
       - 河牌圈大底池增加 50% 延迟基础值

    4. 时区感知：
       - 深夜（01:00-07:00）操作频率降低 30%，决策更慢
    """
    def get_delay(self, confidence: float, street: Street, pot_size: float) -> float: ...
```

#### `stealth/session.py`

```python
class SessionManager:
    """
    会话节律管理，模拟人类玩家在线行为：

    标准会话计划（可配置）：
    - 单次会话时长：90–180 分钟（正态分布采样）
    - 会话间强制休息：30–90 分钟
    - 每日最大在线：6 小时
    - 每周随机 1-2 天完全不在线

    微休息（在线期间）：
    - 每 20-40 分钟触发 2-8 分钟 AFK（上厕所/喝水）
    - AFK 期间不响应任何牌局（timeout 弃牌）

    实现：
    - 维护 session_start, session_end, break_schedule
    - 主循环每帧检查 should_pause() → 若是则 sleep + 不执行任何操作
    """
    def should_pause(self) -> bool: ...
    def log_session(self): ...
```

#### `stealth/fingerprint.py`

```python
class ProcessStealth:
    """
    进程与系统特征隐藏：

    1. 进程名伪装
       - Windows：使用 ctypes 修改进程描述符
       - Linux：修改 /proc/self/comm

    2. 窗口标题随机化
       - 若使用辅助窗口，随机命名

    3. 内存扫描对抗
       - 关键字符串（"poker bot", "equity"等）在内存中加密存储
       - 使用 ctypes 而非 Python 字符串常量

    4. 截图 API 隐藏（高级，按需实现）
       - 方案 A：通过 OBS 虚拟摄像头中转截图（最安全）
       - 方案 B：使用 DirectX DXGI 桌面复制 API（绕过 GDI 监控）
       - 方案 C：在独立 VM/容器中运行，主机侧截图

    5. 网络流量特征（在线平台）
       - 操作时间戳尽量与玩家决策时间对齐
       - 避免在发包窗口期外发送行动指令
    """
```

---

### 3.6 数据记录层 — Analytics

#### `analytics/logger.py`

```python
# 使用库：SQLAlchemy + SQLite（本地）/ PostgreSQL（多机）
class HandLogger:
    """
    记录每手牌完整信息：
    - hand_id, timestamp, street
    - hole_cards, board, pot, stack
    - hero_action, action_amount
    - equity_at_decision, ev_estimate
    - outcome（赢/输，筹码变化）

    自动导出为 HH（Hand History）格式，兼容 PokerTracker / Hold'em Manager
    """
```

#### `analytics/dashboard.py`

```python
# 使用库：Streamlit + Plotly
# 运行：streamlit run analytics/dashboard.py

"""
实时看板指标：
- 收益曲线（BB/100 滚动均值）
- 位置盈亏（按 BTN/CO/MP/UTG/BB/SB 分解）
- 每街道 EV 泄露热力图
- 对手统计排行（VPIP/PFR/AF）
- 今日会话时长与在线节律
"""
```

---

## 4. 核心库选型

| 功能 | 选型 | 理由 |
|------|------|------|
| 屏幕捕获 | `mss` | 速度最快，跨平台，零依赖 |
| 图像处理 | `opencv-python` | 工业级，模板匹配+预处理齐全 |
| 目标检测 | `ultralytics` (YOLOv8) | 预训练扑克牌模型可用，推理快 |
| OCR | `easyocr` | 对数字识别优于 Tesseract，支持 GPU |
| 手牌评估 | `treys` | 纯 Python 最快的 7 张牌评估器 |
| 数据模型 | `pydantic` v2 | 强类型验证，自动序列化 |
| 鼠标控制 | `pyautogui` → `pyclick` | 阶段一简单，阶段二换贝塞尔 |
| 数据存储 | `sqlalchemy` + SQLite | 零运维，结构化查询 |
| 配置管理 | `pydantic-settings` + YAML | 类型安全的配置读取 |
| 日志 | `loguru` | 比标准库 logging 简洁，支持滚动文件 |
| 可视化 | `streamlit` + `plotly` | 零前端代码快速出看板 |
| 测试 | `pytest` + `pytest-mock` | 标准，支持 fixture 截图测试 |
| 数值计算 | `numpy` | 蒙特卡洛模拟加速 |

**安装：**
```bash
pip install mss opencv-python ultralytics easyocr treys pydantic pydantic-settings \
            pyautogui pyclick sqlalchemy loguru streamlit plotly numpy pytest pytest-mock
```

---

## 5. 数据结构定义

### GameState 完整字段

```python
@dataclass
class GameState:
    # 牌局信息
    hand_id: str                    # UUID，每手牌唯一
    street: Street
    hole_cards: list[str]           # ["Ah", "Kd"]
    community_cards: list[str]      # 0-5 张

    # 筹码信息
    pot: float
    to_call: float                  # 0 = 可以 check
    min_raise: float
    max_raise: float                # 通常为英雄筹码量（全下上限）

    # 玩家信息
    players: list[Player]
    hero_seat: int
    button_seat: int
    is_hero_turn: bool

    # 元信息
    confidence: float               # 整体识别置信度
    timestamp: float                # time.time()
    raw_frame: np.ndarray | None    # 原始截图（调试用，生产关闭）
```

### Decision 输出

```python
@dataclass
class Decision:
    action: Action
    amount: float                   # fold/check/call 时为 0
    reasoning: str                  # 供日志记录的决策理由
    equity: float                   # 决策时的胜率
    ev: float                       # 期望值估算
    confidence: float               # 决策置信度
```

---

## 6. 开发阶段与优先级

### Phase 1：视觉 + 状态（Week 1-3）

**目标：** 能够准确读取任意一手牌的完整状态。

- [ ] `vision/capture.py` — 接入 mss，稳定截图
- [ ] `vision/detector.py` — YOLOv8 识别 52 张牌（准确率 > 95%）
- [ ] `vision/ocr.py` — EasyOCR 提取底池/筹码数字（误差 < 1%）
- [ ] `state/models.py` — 定义全部 Pydantic 模型
- [ ] `state/parser.py` — 组装 GameState，置信度校验
- [ ] `state/tracker.py` — 跨帧状态稳定性
- [ ] 单元测试：100 张截图样本，GameState 准确率 > 92%

**验收标准：** 将系统跑在测试截图集上，控制台能输出正确 GameState 的 JSON。

---

### Phase 2：决策引擎（Week 4-6）

**目标：** 能够对任意合法 GameState 给出有理有据的决策。

- [ ] `engine/equity.py` — 蒙特卡洛胜率（10k 次 < 50ms）
- [ ] `engine/ranges.py` — 翻前范围矩阵加载（169 手牌 × 位置）
- [ ] `engine/gto.py` — 翻前查找表策略
- [ ] `engine/postflop.py` — 基础翻后策略（赔率 + equity）
- [ ] `engine/opponent.py` — VPIP/PFR 统计追踪
- [ ] 单元测试：经典场景决策对齐（对比 GTO Wizard 参考答案）

**验收标准：** 在 PokerTH 模拟器中手动输入 20 个场景，决策准确率与 GTO 参考 > 80% 吻合。

---

### Phase 3：端到端联调（Week 7-8）

**目标：** 在本地模拟器中完整运行主循环，无人工干预打完一局。

- [ ] `executor/controller.py` — 接入 PyAutoGUI 执行动作
- [ ] `main.py` — 主循环（截图 → 解析 → 决策 → 执行）
- [ ] 异常处理：识别失败 / 决策超时 / 执行失败的 fallback
- [ ] 日志全链路：每手牌完整记录
- [ ] `analytics/logger.py` — SQLite 入库

**验收标准：** 系统在 PokerTH 中连续运行 50 手无崩溃，胜率统计可信。

---

### Phase 4：对手建模 + 策略优化（Week 9-10）

- [ ] `engine/opponent.py` — 完整 HUD 统计（AF / 3-bet% / Fold to C-bet）
- [ ] 玩家类型识别并调整策略权重
- [ ] 对鱼型玩家加大价值下注频率
- [ ] 漏洞检测（过度 bluff / 过度弃牌的对手）

---

### Phase 5：Stealth 反检测层（Week 11-13，流程稳定后）

- [ ] `stealth/human_mouse.py` — pyclick 贝塞尔曲线鼠标
- [ ] `stealth/timing.py` — 对数正态决策延迟
- [ ] `stealth/session.py` — 会话管理 + 微休息调度
- [ ] `stealth/fingerprint.py` — 进程伪装（按需）
- [ ] A/B 测试：Stealth 开/关时决策节奏对比
- [ ] 压测：连续 4 小时运行无异常

---

### Phase 6：看板与复盘（Week 14）

- [ ] `analytics/dashboard.py` — Streamlit 实时看板
- [ ] HH 格式导出（兼容 PT4 / HM3）
- [ ] EV 泄露报告自动生成

---

## 7. Agent 协作约定

多 Agent 并行开发时遵守以下规范：

### 接口契约

- 每个模块暴露的公开接口（类名 + 方法签名）**只能新增，不能修改**
- 破坏性修改需先在 AGENTS.md 中提 RFC，经 review 后执行
- 接口变更必须同步更新对应的 type stub（`*.pyi`）

### 数据流方向

```
Vision → State → Engine → Executor
                  ↓
              Analytics
```

- 严格单向，下游不得直接调用上游的内部方法
- Engine 层**不得**持有 Vision 或 State 的引用，只接受函数参数

### 错误处理约定

```python
# 每层定义自己的异常类
class VisionError(Exception): ...
class StateError(Exception): ...
class EngineError(Exception): ...

# 主循环捕获并记录，不中断
try:
    state = parser.parse(frame)
except StateError as e:
    logger.warning(f"状态解析失败，跳过本帧: {e}")
    continue
```

### 配置优先级

`config.yaml` < 环境变量 (`POKER_AI_*`) < 命令行参数

### 日志规范

```python
from loguru import logger

# 格式：[时间][模块][等级] 消息
logger.info("[Vision] 识别到手牌: Ah Kd, 置信度: 0.97")
logger.debug("[Engine] Equity: 0.64, EV: +2.3BB, 决策: raise 3x")
logger.warning("[State] OCR 置信度过低 (0.71), 保持上一帧状态")
logger.error("[Executor] 点击失败，按钮坐标未找到: fold")
```

---

## 8. 测试策略

### 单元测试

```
tests/
├── test_vision.py     # 用 fixtures/ 截图测试识别准确率
├── test_equity.py     # 对比已知胜率场景（精度 < 0.5%）
├── test_gto.py        # 对比 GTO Wizard 参考决策
└── test_stealth.py    # 鼠标轨迹人类相似度评分
```

### 集成测试

- 使用 **PokerTH**（开源，支持 AI 接口）作为主要测试环境
- 录制 50 手牌的屏幕视频，回放测试 Vision + State 层
- 定期运行 1000 手蒙特卡洛模拟验证引擎 EV 为正

### 性能基准

| 组件 | 目标延迟 | 测量方式 |
|------|---------|---------|
| 截图 (mss) | < 5ms | `time.perf_counter()` |
| 牌面识别 (YOLO) | < 30ms | 100 帧平均 |
| OCR | < 50ms | 100 帧平均 |
| 胜率计算 (10k) | < 50ms | `timeit` |
| 决策总延迟 | < 150ms | 端到端 |

---

## 附录：推荐参考资料

- **treys 文档**：https://github.com/ihendley/treys
- **YOLOv8 扑克牌数据集**：https://universe.roboflow.com/augmented-startups/playing-cards-ow27d
- **pyclick（人类鼠标）**：https://github.com/patrikoss/pyclick
- **EasyOCR**：https://github.com/JaidedAI/EasyOCR
- **翻前范围数据**：https://www.pokercoaching.com/ranges（可手动导出为 CSV）
- **GTO 参考**：Solver: PioSOLVER / GTO+ 导出的策略文件

---

*文档版本：v1.0 | 最后更新：2025-03*
