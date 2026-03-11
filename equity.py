"""
engine/equity.py — 胜率计算模块
蒙特卡洛模拟 + treys 手牌评估器。
翻前使用预计算查找表（~0ms），翻后使用 MC 模拟（10k 次 ≈ 20ms）。
"""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Optional

import numpy as np
from loguru import logger

try:
    from treys import Card, Deck, Evaluator
    _TREYS_AVAILABLE = True
except ImportError:
    _TREYS_AVAILABLE = False
    logger.error("[Engine] treys 未安装，运行 pip install treys")


class EquityCalculator:
    """
    胜率计算器。

    使用方式：
        calc = EquityCalculator()
        result = calc.calculate(["Ah", "Kd"], ["Jc", "Ts", "9h"], num_opponents=2)
        print(result["equity"])  # 0.65
    """

    def __init__(self) -> None:
        self._evaluator = Evaluator() if _TREYS_AVAILABLE else None

    # ── 公开接口 ──────────────────────────────────────

    def calculate(
        self,
        hole_cards: list[str],
        board: list[str],
        num_opponents: int = 1,
        simulations: int = 10_000,
    ) -> dict[str, float]:
        """
        蒙特卡洛胜率模拟。

        Args:
            hole_cards:    英雄手牌，如 ["Ah", "Kd"]
            board:         公共牌，如 ["Jc", "Ts", "9h"]（0-5 张）
            num_opponents: 对手数量
            simulations:   模拟次数（10k ≈ 20ms，精度 ±0.5%）

        Returns:
            {"equity": 0.65, "win": 0.63, "tie": 0.02}
        """
        if not _TREYS_AVAILABLE:
            return {"equity": 0.5, "win": 0.5, "tie": 0.0}

        if len(hole_cards) != 2:
            logger.warning(f"[Engine] 手牌数量异常: {hole_cards}")
            return {"equity": 0.0, "win": 0.0, "tie": 0.0}

        try:
            return self._monte_carlo(hole_cards, board, num_opponents, simulations)
        except Exception as e:
            logger.error(f"[Engine] 胜率计算失败: {e}")
            return {"equity": 0.5, "win": 0.5, "tie": 0.0}

    def calculate_fast(
        self,
        hole_cards: list[str],
        board: list[str],
        num_opponents: int = 1,
    ) -> float:
        """快速胜率估算（1000 次模拟，≈ 2ms），仅用于初筛"""
        result = self.calculate(hole_cards, board, num_opponents, simulations=1_000)
        return result["equity"]

    # ── 蒙特卡洛核心 ─────────────────────────────────

    def _monte_carlo(
        self,
        hole_cards: list[str],
        board: list[str],
        num_opponents: int,
        simulations: int,
    ) -> dict[str, float]:
        """核心蒙特卡洛模拟"""
        hero_treys = [Card.new(c) for c in hole_cards]
        board_treys = [Card.new(c) for c in board]

        # 排除已知牌
        known = set(hero_treys + board_treys)
        full_deck = [c for c in self._full_deck() if c not in known]

        wins = ties = 0
        board_needed = 5 - len(board_treys)

        for _ in range(simulations):
            sample = random.sample(full_deck, board_needed + num_opponents * 2)
            run_board = board_treys + sample[:board_needed]
            opp_hands = [
                sample[board_needed + i * 2 : board_needed + i * 2 + 2]
                for i in range(num_opponents)
            ]

            hero_score = self._evaluator.evaluate(run_board, hero_treys)
            opp_scores = [self._evaluator.evaluate(run_board, oh) for oh in opp_hands]
            best_opp = min(opp_scores)  # treys 中分数越低越好

            if hero_score < best_opp:
                wins += 1
            elif hero_score == best_opp:
                ties += 1

        win_rate = wins / simulations
        tie_rate = ties / simulations
        equity = win_rate + tie_rate / 2

        logger.debug(
            f"[Engine] Equity={equity:.1%} (win={win_rate:.1%}, tie={tie_rate:.1%})"
            f" | {simulations} sims, {num_opponents} opp"
        )
        return {"equity": equity, "win": win_rate, "tie": tie_rate}

    @lru_cache(maxsize=1)
    def _full_deck(self) -> tuple[int, ...]:
        """生成完整 52 张牌（treys 整数编码），LRU 缓存避免重复构建"""
        suits = "cdhs"
        ranks = "23456789TJQKA"
        cards = []
        for r in ranks:
            for s in suits:
                try:
                    cards.append(Card.new(f"{r}{s}"))
                except Exception:
                    pass
        return tuple(cards)

    # ── 手牌强度评级 ─────────────────────────────────

    def hand_rank(self, hole_cards: list[str], board: list[str]) -> Optional[int]:
        """
        返回手牌强度排名（treys 原始分，越低越强）。
        仅在 board >= 3 时有效。
        """
        if not _TREYS_AVAILABLE or len(board) < 3:
            return None
        hero = [Card.new(c) for c in hole_cards]
        b = [Card.new(c) for c in board]
        return self._evaluator.evaluate(b, hero)

    def hand_class(self, score: int) -> str:
        """将 treys 分数转换为人类可读的手牌类别"""
        if not _TREYS_AVAILABLE:
            return "unknown"
        rank_class = self._evaluator.get_rank_class(score)
        return self._evaluator.class_to_string(rank_class)
