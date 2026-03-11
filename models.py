"""
state/models.py — 核心数据模型定义
所有层之间传递的结构化对象均定义于此。
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════
#  枚举类型
# ═══════════════════════════════════════════════════════

class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class Action(str, Enum):
    FOLD   = "fold"
    CHECK  = "check"
    CALL   = "call"
    RAISE  = "raise"
    ALL_IN = "all_in"


class PlayerType(str, Enum):
    """对手风格分类（由 OpponentModel 更新）"""
    UNKNOWN = "unknown"
    FISH    = "fish"     # VPIP > 40%, PFR < 10%：松被动，价值向倾斜
    TAG     = "tag"      # VPIP 15-25%, PFR 12-20%：标准 GTO 应对
    LAG     = "lag"      # VPIP > 25%, PFR > 20%：等强牌
    NIT     = "nit"      # VPIP < 12%：石头，遇加注直接弃牌


class Position(str, Enum):
    """相对于 Button 的位置"""
    BTN = "btn"
    CO  = "co"
    HJ  = "hj"
    MP  = "mp"
    UTG = "utg"
    SB  = "sb"
    BB  = "bb"


# ═══════════════════════════════════════════════════════
#  玩家模型
# ═══════════════════════════════════════════════════════

class Player(BaseModel):
    seat: int                               # 座位号（0-based）
    stack: float                            # 当前筹码量（BB 为单位）
    bet: float = 0.0                        # 当前街道的下注额
    is_active: bool = True                  # 是否仍在手牌中
    is_all_in: bool = False
    last_action: Optional[Action] = None
    position: Optional[Position] = None
    player_type: PlayerType = PlayerType.UNKNOWN


# ═══════════════════════════════════════════════════════
#  游戏状态（Vision → State 层输出）
# ═══════════════════════════════════════════════════════

class GameState(BaseModel):
    # 手牌标识
    hand_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)

    # 牌局信息
    street: Street
    hole_cards: list[str]           # ["Ah", "Kd"]
    community_cards: list[str] = [] # 0–5 张

    # 筹码信息
    pot: float
    to_call: float = 0.0            # 0 = 可以 check
    min_raise: float = 0.0
    max_raise: float = 0.0          # 英雄筹码上限

    # 玩家信息
    players: list[Player]
    hero_seat: int
    button_seat: int
    is_hero_turn: bool = False

    # 元信息
    confidence: float = 1.0         # 整体识别置信度（< 0.85 则跳过）
    raw_frame: Optional[bytes] = None  # 原始截图（调试用，序列化为 bytes）

    @field_validator("hole_cards")
    @classmethod
    def validate_hole_cards(cls, v: list[str]) -> list[str]:
        assert len(v) in (0, 2), f"手牌数量应为 0 或 2，得到 {len(v)}"
        return v

    @field_validator("community_cards")
    @classmethod
    def validate_community_cards(cls, v: list[str]) -> list[str]:
        assert len(v) in (0, 3, 4, 5), f"公共牌数量应为 0/3/4/5，得到 {len(v)}"
        return v

    @property
    def hero(self) -> Optional[Player]:
        """返回英雄玩家对象"""
        return next((p for p in self.players if p.seat == self.hero_seat), None)

    @property
    def active_opponents(self) -> list[Player]:
        """返回所有活跃的对手"""
        return [p for p in self.players if p.seat != self.hero_seat and p.is_active]

    @property
    def spr(self) -> float:
        """Stack-to-Pot Ratio：用于深筹码决策"""
        if self.pot <= 0:
            return 999.0
        hero = self.hero
        return (hero.stack if hero else 0) / self.pot

    @property
    def pot_odds(self) -> float:
        """底池赔率：需要的最低胜率以盈亏平衡"""
        if self.to_call <= 0:
            return 0.0
        return self.to_call / (self.pot + self.to_call)


# ═══════════════════════════════════════════════════════
#  决策输出（Engine → Executor）
# ═══════════════════════════════════════════════════════

class Decision(BaseModel):
    action: Action
    amount: float = 0.0             # fold/check/call 时为 0

    # 决策依据（供日志 & LLM 复盘使用）
    equity: float = 0.0             # 当前胜率
    ev: float = 0.0                 # 期望值（BB）
    confidence: float = 1.0         # 决策置信度
    reasoning: str = ""             # 文字说明

    # 复杂度分级（影响 Stealth 层的思考时间模拟）
    complexity: str = "medium"      # simple | medium | hard

    def __str__(self) -> str:
        if self.action in (Action.RAISE, Action.ALL_IN):
            return f"{self.action.value} {self.amount:.1f}BB (equity={self.equity:.1%}, EV={self.ev:+.2f}BB)"
        return f"{self.action.value} (equity={self.equity:.1%}, EV={self.ev:+.2f}BB)"


# ═══════════════════════════════════════════════════════
#  Vision 层中间结构
# ═══════════════════════════════════════════════════════

class DetectedButtons(BaseModel):
    """由 Vision 层输出，Executor 层消费"""
    fold:  Optional[tuple[int, int]] = None
    check: Optional[tuple[int, int]] = None
    call:  Optional[tuple[int, int]] = None
    raise_: Optional[tuple[int, int]] = Field(None, alias="raise")
    amount_input: Optional[tuple[int, int]] = None  # 金额输入框坐标

    model_config = {"populate_by_name": True}


class GameFrame(BaseModel):
    """Vision 层原始输出，传递给 State 层"""
    hole_cards: list[str]
    community_cards: list[str]
    pot: Optional[float]
    stacks: dict[int, float]        # seat → stack
    bets: dict[int, float]          # seat → bet
    buttons: DetectedButtons
    confidence: float
    timestamp: float = Field(default_factory=time.time)
