"""
tests/test_equity.py — 胜率计算单元测试
"""
import pytest
import sys
sys.path.insert(0, ".")


class TestEquityCalculator:
    """胜率计算精度验证"""

    def setup_method(self):
        from engine.equity import EquityCalculator
        self.calc = EquityCalculator()

    def test_aces_vs_one_opponent_preflop(self):
        """AA 翻前单挑胜率应约为 85%"""
        result = self.calc.calculate(["Ah", "Ad"], [], num_opponents=1, simulations=5000)
        assert result["equity"] > 0.80, f"AA 胜率 {result['equity']:.1%} 低于预期"

    def test_72o_worst_hand_preflop(self):
        """72o 翻前单挑胜率约 35%"""
        result = self.calc.calculate(["7h", "2d"], [], num_opponents=1, simulations=5000)
        assert result["equity"] < 0.45, f"72o 胜率 {result['equity']:.1%} 高于预期"

    def test_flush_draw_on_flop(self):
        """同花听牌翻后胜率约 35-40%"""
        # 手牌：Ah Kh，公共牌：2h 5h 9c（同花听牌）
        result = self.calc.calculate(
            ["Ah", "Kh"], ["2h", "5h", "9c"],
            num_opponents=1, simulations=5000
        )
        assert 0.30 < result["equity"] < 0.55

    def test_multiway_equity_decreases(self):
        """多路底池中胜率下降"""
        single = self.calc.calculate(["Ah", "Kd"], [], num_opponents=1, simulations=3000)
        multi  = self.calc.calculate(["Ah", "Kd"], [], num_opponents=3, simulations=3000)
        assert multi["equity"] < single["equity"], "多路底池胜率应低于单挑"

    def test_nut_flush_on_river(self):
        """坚果同花河牌圈胜率应接近 100%"""
        result = self.calc.calculate(
            ["Ah", "Kh"], ["2h", "5h", "9h", "3c", "7d"],
            num_opponents=1, simulations=3000
        )
        assert result["equity"] > 0.90


class TestGTOStrategy:
    """GTO 策略决策测试"""

    def setup_method(self):
        from engine.gto import GTOStrategy
        self.gto = GTOStrategy({})

    def test_aces_should_raise_preflop(self):
        """AA 翻前应大概率加注"""
        from state.models import GameState, Street, Player, Action
        state = GameState(
            street=Street.PREFLOP,
            hole_cards=["Ah", "Ad"],
            community_cards=[],
            pot=1.5,
            to_call=1.0,
            players=[
                Player(seat=0, stack=100, bet=0.5),
                Player(seat=1, stack=100, bet=1.0),
            ],
            hero_seat=0,
            button_seat=0,
        )
        # 多次采样，AA 应至少 80% 的情况下选择加注
        raises = sum(
            1 for _ in range(100)
            if self.gto.decide_preflop(state).action == Action.RAISE
        )
        assert raises >= 70, f"AA 加注频率 {raises}% 过低"

    def test_fold_when_equity_below_pot_odds(self):
        """胜率低于底池赔率时应弃牌"""
        from state.models import GameState, Street, Player, Action
        state = GameState(
            street=Street.FLOP,
            hole_cards=["7h", "2d"],
            community_cards=["Ah", "Kd", "Qc"],
            pot=10.0,
            to_call=8.0,   # 底池赔率 = 8/18 ≈ 44%
            players=[Player(seat=0, stack=50), Player(seat=1, stack=50)],
            hero_seat=0,
            button_seat=1,
        )
        decision = self.gto.decide_postflop(state, equity=0.10)
        assert decision.action == Action.FOLD
