"""
engine/llm_advisor.py — LLM 辅助分析层
⚠️  不参与实时决策路径，仅用于离线分析与复盘报告。

支持的 LLM 提供商（均兼容 OpenAI SDK 格式）：
- 智谱 AI (GLM-4-Flash)：免费额度，推荐
- MiniMax (M2.5)：https://api.minimax.chat/v1
- Kimi / Moonshot：https://api.moonshot.cn/v1
"""
from __future__ import annotations

import json
from typing import Optional

from loguru import logger

from state.models import GameState, PlayerType


# 各 LLM 提供商配置
PROVIDER_CONFIGS = {
    "zhipuai": {
        "base_url": "https://open.bigmodel.ai/api/paas/v4/",
        "default_model": "glm-4-flash",
    },
    "minimax": {
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "MiniMax-Text-01",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
    },
}


class LLMAdvisor:
    """
    LLM 辅助分析器。

    功能：
    1. opponent_summary()   — 根据 HUD 统计生成对手风格描述
    2. session_report()     — 会话结束后生成复盘报告
    3. explain_decision()   — 解释某手牌的决策逻辑（调试用）

    注意：所有方法均为异步 + 有超时保护，失败时静默返回 None，不影响主循环。
    """

    def __init__(self, config: dict) -> None:
        self.enabled = config.get("enabled", False)
        self.provider = config.get("provider", "zhipuai")
        self.model = config.get("model", "glm-4-flash")
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 10)

        provider_cfg = PROVIDER_CONFIGS.get(self.provider, {})
        self.base_url = config.get("base_url", provider_cfg.get("base_url", ""))

        self._client = None
        if self.enabled:
            self._init_client()

    # ── 公开接口 ──────────────────────────────────────

    def opponent_summary(self, seat: int, stats: dict) -> Optional[str]:
        """
        根据 HUD 统计数据生成对手风格描述。

        Args:
            seat:  对手座位号
            stats: {"vpip": 0.35, "pfr": 0.08, "af": 1.2, "hands": 120, ...}

        Returns:
            自然语言描述，如 "典型鱼型玩家，VPIP 极高，几乎不主动加注，
            适合对其进行价值下注，减少虚张声势。"
        """
        if not self.enabled or self._client is None:
            return None

        prompt = self._build_opponent_prompt(seat, stats)
        return self._call_llm(prompt, max_tokens=200)

    def session_report(self, session_data: dict) -> Optional[str]:
        """
        会话结束后生成复盘报告。

        Args:
            session_data: {
                "hands_played": 150,
                "profit_bb": 12.5,
                "vpip": 0.22,
                "pfr": 0.17,
                "biggest_pot_won": 45.0,
                "biggest_pot_lost": 38.0,
                "streets_breakdown": {...},
            }
        """
        if not self.enabled or self._client is None:
            return None

        prompt = self._build_session_prompt(session_data)
        return self._call_llm(prompt, max_tokens=500)

    def explain_decision(
        self,
        state: GameState,
        decision_reasoning: str,
        equity: float,
        ev: float,
    ) -> Optional[str]:
        """解释某手牌决策，用于调试与学习（调试模式下写入日志）"""
        if not self.enabled or self._client is None:
            return None

        prompt = f"""你是一位德州扑克 GTO 专家，请用简体中文简洁解释以下决策：

手牌: {state.hole_cards}
公共牌: {state.community_cards}
街道: {state.street.value}
底池: {state.pot:.1f}BB，需跟注: {state.to_call:.1f}BB
胜率: {equity:.1%}，期望值: {ev:+.2f}BB

决策系统给出的理由：{decision_reasoning}

请用 2-3 句话解释这个决策是否合理，以及有什么需要注意的地方。"""

        return self._call_llm(prompt, max_tokens=150)

    # ── 内部方法 ──────────────────────────────────────

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """调用 LLM API，带超时保护"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.debug(f"[LLM] API 调用失败（不影响主循环）: {e}")
            return None

    def _build_opponent_prompt(self, seat: int, stats: dict) -> str:
        return f"""你是德州扑克分析专家，请根据以下 HUD 数据，用简体中文生成一段简洁的对手风格描述（100字以内）：

座位: {seat}
样本手数: {stats.get('hands', 0)}
VPIP（自愿入底池率）: {stats.get('vpip', 0):.1%}
PFR（翻前加注率）: {stats.get('pfr', 0):.1%}
AF（攻击系数）: {stats.get('af', 0):.2f}
3-bet 频率: {stats.get('three_bet', 0):.1%}
面对持续下注弃牌率: {stats.get('fold_to_cbet', 0):.1%}

请给出：1）玩家类型判断 2）针对性策略建议（各一句话）"""

    def _build_session_prompt(self, data: dict) -> str:
        return f"""你是德州扑克教练，请分析以下会话数据并给出简短复盘报告（200字以内，使用简体中文）：

本次会话统计：
- 共打 {data.get('hands_played', 0)} 手
- 盈亏：{data.get('profit_bb', 0):+.1f} BB
- VPIP: {data.get('vpip', 0):.1%} | PFR: {data.get('pfr', 0):.1%}
- 最大赢池: {data.get('biggest_pot_won', 0):.1f}BB
- 最大输池: {data.get('biggest_pot_lost', 0):.1f}BB

请指出：1）表现亮点 2）需要改进的地方 3）一条具体建议"""

    def _init_client(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key or "placeholder",
                base_url=self.base_url,
            )
            logger.info(f"[LLM] 初始化成功，provider={self.provider}, model={self.model}")
        except ImportError:
            logger.warning("[LLM] openai 库未安装，LLM 功能禁用。运行: pip install openai")
        except Exception as e:
            logger.error(f"[LLM] 初始化失败: {e}")
