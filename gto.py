"""
engine/gto.py — GTO 策略决策引擎
翻前：查找表驱动（CSV 范围文件）
翻后：基于赔率 + Equity + SPR 的简化 GTO 规则
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from state.models import Action, Decision, GameState, PlayerType, Street


# 下注尺度选项（相对底池比例）
BET_SIZES = [0.33, 0.50, 0.67, 1.00]

# 翻前手牌强度分组（169 种起手牌 → 组别）
# 格式：hand_key → group (1=顶级, 5=最弱)
# 完整表在 ranges.py 中定义，这里是示例
_PREFLOP_GROUPS = {
    # 顶级手牌（Group 1）
    "AA": 1, "KK": 1, "QQ": 1, "JJ": 1, "AKs": 1,
    # 强手牌（Group 2）
    "TT": 2, "99": 2, "AQs": 2, "AJs": 2, "AKo": 2, "KQs": 2,
    # 中等手牌（Group 3）
    "88": 3, "77": 3, "ATs": 3, "KJs": 3, "QJs": 3, "AQo": 3,
    # 投机手牌（Group 4）
    "66": 4, "55": 4, "44": 4, "33": 4, "22": 4,
    "A9s": 4, "A8s": 4, "KTs": 4, "QTs": 4, "JTs": 4, "T9s": 4,
}


class GTOStrategy:
    """
    GTO 决策引擎。

    翻前查找表：position × hand_group → 混合策略 {fold%, call%, raise%}
    翻后：基于 equity + pot_odds + SPR + 对手类型的启发规则
    """

    def __init__(self, config: dict) -> None:
        self.ranges_dir = Path(config.get("preflop_ranges_dir", "data/preflop_ranges"))
        self.min_equity = config.get("min_equity_to_call", 0.30)
        self.bluff_freq = config.get("bluff_frequency", 0.25)
        self.bet_sizes = config.get("bet_sizes", BET_SIZES)

        self._preflop_tables: dict = {}
        self._load_preflop_tables()

    # ── 主决策接口 ────────────────────────────────────

    def decide(
        self,
        state: GameState,
        equity: float,
        opponent_type: PlayerType = PlayerType.UNKNOWN,
    ) -> Decision:
        """统一决策入口，根据街道分发"""
        if state.street == Street.PREFLOP:
            return self.decide_preflop(state, opponent_type)
        return self.decide_postflop(state, equity, opponent_type)

    # ── 翻前决策 ─────────────────────────────────────

    def decide_preflop(
        self,
        state: GameState,
        opponent_type: PlayerType = PlayerType.UNKNOWN,
    ) -> Decision:
        """
        翻前决策：查找表驱动的混合策略。
        """
        hand_key = self._hand_to_key(state.hole_cards)
        position = self._get_hero_position(state)
        group = _PREFLOP_GROUPS.get(hand_key, 5)

        # 根据组别与位置决定策略
        strategy = self._lookup_preflop_strategy(group, position, state)

        # 混合策略采样（GTO 不是确定性的）
        r = random.random()
        if r < strategy["fold"]:
            return Decision(
                action=Action.FOLD, amount=0,
                reasoning=f"翻前弃牌：{hand_key} group={group} pos={position}",
                complexity="simple",
            )
        elif r < strategy["fold"] + strategy["call"]:
            return Decision(
                action=Action.CALL, amount=state.to_call,
                reasoning=f"翻前跟注：{hand_key} group={group} pot_odds={state.pot_odds:.1%}",
                complexity="simple",
            )
        else:
            raise_size = self._preflop_raise_size(state)
            return Decision(
                action=Action.RAISE, amount=raise_size,
                reasoning=f"翻前加注：{hand_key} group={group} → {raise_size:.1f}BB",
                complexity="medium",
            )

    # ── 翻后决策 ─────────────────────────────────────

    def decide_postflop(
        self,
        state: GameState,
        equity: float,
        opponent_type: PlayerType = PlayerType.UNKNOWN,
    ) -> Decision:
        """
        翻后决策（简化 GTO 启发规则）：

        1. 计算底池赔率（pot odds）
        2. 若需要跟注：equity > pot_odds → call/raise，否则 fold
        3. 若可以 check：根据 equity 决定是否主动下注
        4. 下注尺度由 SPR 和手牌强度决定
        """
        pot_odds = state.pot_odds
        spr = state.spr
        can_check = state.to_call <= 0

        # ── 需要跟注的情况 ────────────────────────────
        if not can_check:
            # 胜率远高于赔率：考虑加注
            if equity > pot_odds + 0.15:
                return self._make_raise_decision(state, equity, opponent_type)
            # 胜率满足赔率：跟注
            elif equity > pot_odds:
                return Decision(
                    action=Action.CALL, amount=state.to_call,
                    equity=equity,
                    reasoning=f"跟注：equity={equity:.1%} > pot_odds={pot_odds:.1%}",
                    complexity="medium",
                )
            # 胜率不足：弃牌
            else:
                return Decision(
                    action=Action.FOLD, amount=0,
                    equity=equity,
                    reasoning=f"弃牌：equity={equity:.1%} < pot_odds={pot_odds:.1%}",
                    complexity="simple",
                )

        # ── 可以 check 的情况 ─────────────────────────
        # 强牌：价值下注
        if equity > 0.65:
            return self._make_raise_decision(state, equity, opponent_type)
        # 中等牌：偶尔下注（bluff / 半诈唬）
        elif equity > 0.40 and random.random() < self.bluff_freq:
            return self._make_raise_decision(state, equity, opponent_type)
        # 弱牌：check
        else:
            return Decision(
                action=Action.CHECK, amount=0,
                equity=equity,
                reasoning=f"过牌：equity={equity:.1%} 不足以下注",
                complexity="simple",
            )

    # ── 辅助方法 ──────────────────────────────────────

    def _make_raise_decision(
        self,
        state: GameState,
        equity: float,
        opponent_type: PlayerType,
    ) -> Decision:
        """构建加注决策，选择合适的下注尺度"""
        size_ratio = self._choose_bet_size(equity, state.spr, opponent_type)
        amount = state.pot * size_ratio
        amount = max(state.min_raise, min(amount, state.max_raise))

        # 全下判断
        hero = state.hero
        if hero and amount >= hero.stack * 0.9:
            return Decision(
                action=Action.ALL_IN, amount=hero.stack if hero else amount,
                equity=equity,
                reasoning=f"全下：equity={equity:.1%}, SPR={state.spr:.1f}",
                complexity="hard",
            )

        complexity = "simple" if equity > 0.80 else "medium" if equity > 0.60 else "hard"
        return Decision(
            action=Action.RAISE, amount=round(amount, 1),
            equity=equity,
            reasoning=f"加注 {size_ratio:.0%}底池：equity={equity:.1%}, SPR={state.spr:.1f}",
            complexity=complexity,
        )

    def _choose_bet_size(
        self,
        equity: float,
        spr: float,
        opponent_type: PlayerType,
    ) -> float:
        """
        根据 equity / SPR / 对手类型选择下注尺度。

        对鱼型玩家：加大价值下注（1.0x 底池）
        对石头玩家：减小尺度诱导（0.33x 底池）
        SPR < 4：倾向全下
        """
        if spr < 2:
            return 1.0  # 短筹码倾向全下

        # 对手类型调整
        size_map = {
            PlayerType.FISH: 1.0,    # 鱼型：大尺度价值
            PlayerType.NIT:  0.33,   # 石头：小尺度诱导
            PlayerType.LAG:  0.50,   # LAG：中等尺度
            PlayerType.TAG:  0.67,   # TAG：标准 GTO
        }
        base = size_map.get(opponent_type, 0.67)

        # 根据 equity 微调
        if equity > 0.80:
            base = min(base * 1.3, 1.0)
        elif equity < 0.55:
            base = base * 0.8  # 弱牌小尺度诈唬

        # 选最接近的标准尺度
        available = [s for s in self.bet_sizes if s <= 1.5]
        return min(available, key=lambda s: abs(s - base))

    def _lookup_preflop_strategy(
        self,
        group: int,
        position: str,
        state: GameState,
    ) -> dict[str, float]:
        """
        从查找表获取翻前策略，返回 {fold, call, raise} 概率。
        若查找表不存在则使用内置规则。
        """
        key = f"{position}_{group}"
        if key in self._preflop_tables:
            return self._preflop_tables[key]

        # 内置简化策略（查找表未加载时的 fallback）
        if group == 1:
            return {"fold": 0.0, "call": 0.1, "raise": 0.9}
        elif group == 2:
            return {"fold": 0.0, "call": 0.3, "raise": 0.7}
        elif group == 3:
            # 位置越好，越激进
            if position in ("btn", "co"):
                return {"fold": 0.1, "call": 0.3, "raise": 0.6}
            return {"fold": 0.2, "call": 0.5, "raise": 0.3}
        elif group == 4:
            if position in ("btn", "co", "hj"):
                return {"fold": 0.3, "call": 0.5, "raise": 0.2}
            return {"fold": 0.6, "call": 0.3, "raise": 0.1}
        else:  # group 5（弱牌）
            if position == "btn":
                return {"fold": 0.5, "call": 0.3, "raise": 0.2}
            return {"fold": 0.8, "call": 0.15, "raise": 0.05}

    def _preflop_raise_size(self, state: GameState) -> float:
        """翻前标准加注尺度（相对于大盲注）"""
        # 标准开池：2.5BB（6人桌），3BB（全满桌）
        num_players = len([p for p in state.players if p.is_active])
        base = 2.5 if num_players <= 6 else 3.0
        # 有人已经入底池：+1BB/limper
        limpers = sum(1 for p in state.players
                      if p.seat != state.hero_seat and p.bet > 0 and p.last_action != Action.RAISE)
        return base + limpers

    def _hand_to_key(self, hole_cards: list[str]) -> str:
        """将手牌转换为查找表 key，如 ["Ah", "Kd"] → "AKo" 或 "AKs" """
        if len(hole_cards) != 2:
            return "XX"
        c1, c2 = hole_cards
        r1, s1 = c1[0], c1[1]
        r2, s2 = c2[0], c2[1]

        rank_order = "23456789TJQKA"
        if rank_order.index(r1) < rank_order.index(r2):
            r1, r2 = r2, r1
            s1, s2 = s2, s1

        if r1 == r2:
            return f"{r1}{r2}"  # 口袋对
        suffix = "s" if s1 == s2 else "o"
        return f"{r1}{r2}{suffix}"

    def _get_hero_position(self, state: GameState) -> str:
        """计算英雄的位置"""
        num_players = len([p for p in state.players if p.is_active])
        seats = [p.seat for p in state.players if p.is_active]
        seats_sorted = sorted(seats)
        btn_idx = seats_sorted.index(state.button_seat) if state.button_seat in seats_sorted else 0
        hero_idx = seats_sorted.index(state.hero_seat) if state.hero_seat in seats_sorted else 0

        relative = (hero_idx - btn_idx) % num_players
        position_map = {
            0: "btn", 1: "sb", 2: "bb",
            3: "utg", 4: "mp", 5: "hj", 6: "co",
        }
        return position_map.get(relative, "mp")

    def _load_preflop_tables(self) -> None:
        """从 CSV 文件加载翻前范围表"""
        if not self.ranges_dir.exists():
            logger.warning(f"[Engine] 翻前范围目录不存在: {self.ranges_dir}，使用内置规则")
            return

        count = 0
        for csv_file in self.ranges_dir.glob("*.csv"):
            try:
                with open(csv_file, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = f"{row.get('position', '')}_{row.get('group', '')}"
                        self._preflop_tables[key] = {
                            "fold":  float(row.get("fold", 0)),
                            "call":  float(row.get("call", 0)),
                            "raise": float(row.get("raise", 0)),
                        }
                        count += 1
            except Exception as e:
                logger.warning(f"[Engine] 加载翻前范围失败 {csv_file}: {e}")

        if count > 0:
            logger.info(f"[Engine] 加载了 {count} 条翻前范围策略")
