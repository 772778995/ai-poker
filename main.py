"""
main.py — poker-ai 主控循环
截图 → 识别 → 解析 → 决策 → 执行，每轮约 200–500ms。

用法：
    python main.py                    # 正常运行
    python main.py --debug            # 调试模式（保存问题帧）
    python main.py --config my.yaml   # 指定配置文件
    python main.py --dry-run          # 仅决策，不执行鼠标操作
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml
from loguru import logger

# ── 延迟导入（避免启动时 ImportError 终止程序）─────────
def _import_modules(config: dict):
    """导入所有模块，返回实例化后的组件"""
    from vision.capture import ScreenCapture
    from vision.detector import CardDetector
    from vision.ocr import PokerOCR
    from state.parser import StateParser
    from state.tracker import StateTracker
    from engine.equity import EquityCalculator
    from engine.gto import GTOStrategy
    from engine.opponent import OpponentModel
    from engine.llm_advisor import LLMAdvisor
    from executor.controller import ActionController
    from analytics.logger import HandLogger

    return {
        "capture":   ScreenCapture(config.get("capture", {})),
        "detector":  CardDetector(config.get("vision", {})),
        "ocr":       PokerOCR(config.get("vision", {})),
        "parser":    StateParser(config.get("state", {})),
        "tracker":   StateTracker(),
        "equity":    EquityCalculator(),
        "gto":       GTOStrategy(config.get("engine", {})),
        "opponent":  OpponentModel(),
        "llm":       LLMAdvisor(config.get("llm", {})),
        "executor":  ActionController(config.get("executor", {})),
        "logger":    HandLogger(config.get("analytics", {})),
    }


def main(args: argparse.Namespace) -> None:
    # ── 配置加载 ─────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── 日志初始化 ───────────────────────────────────
    log_level = "DEBUG" if args.debug else config.get("log_level", "INFO")
    log_file = config.get("log_file", "logs/poker_ai.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=log_level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(log_file, level="DEBUG", rotation=config.get("log_rotation", "100 MB"),
               encoding="utf-8")

    if args.debug:
        config.setdefault("vision", {})["debug_save_frames"] = True

    logger.info("=" * 60)
    logger.info("  poker-ai 启动")
    logger.info(f"  配置: {config_path} | 调试: {args.debug} | 试跑: {args.dry_run}")
    logger.info("=" * 60)

    # ── 模块初始化 ───────────────────────────────────
    try:
        mods = _import_modules(config)
    except Exception as e:
        logger.error(f"模块初始化失败: {e}")
        raise

    # ── Stealth 会话管理 ─────────────────────────────
    stealth_enabled = config.get("stealth", {}).get("enabled", False)
    session_mgr = None
    if stealth_enabled:
        from stealth.session import SessionManager
        session_mgr = SessionManager(config.get("stealth", {}).get("session", {}))
        logger.info("[Stealth] 会话管理已激活")

    # ── 主循环 ───────────────────────────────────────
    fps_limit = config.get("capture", {}).get("fps_limit", 8)
    frame_interval = 1.0 / fps_limit
    min_confidence = config.get("state", {}).get("min_confidence", 0.85)

    logger.info(f"主循环启动，帧率上限: {fps_limit} FPS")
    logger.info("按 Ctrl+C 退出")

    last_hand_id = None

    try:
        while True:
            loop_start = time.perf_counter()

            # ── Stealth：检查是否应暂停 ──────────────
            if session_mgr and session_mgr.should_pause():
                time.sleep(1.0)
                continue

            # ── 1. 截图 ──────────────────────────────
            try:
                frame = mods["capture"].grab()
                regions = mods["capture"].grab_all_regions()
            except Exception as e:
                logger.error(f"[Main] 截图失败: {e}")
                time.sleep(1.0)
                continue

            # ── 2. Vision 识别 ────────────────────────
            try:
                game_frame = mods["parser"].parse_frame(
                    frame, regions,
                    mods["detector"],
                    mods["ocr"],
                )
            except Exception as e:
                logger.warning(f"[Main] 帧解析失败，跳过: {e}")
                _sleep_remaining(loop_start, frame_interval)
                continue

            # ── 3. 状态追踪 ──────────────────────────
            try:
                state = mods["tracker"].update(game_frame)
            except Exception as e:
                logger.warning(f"[Main] 状态追踪失败: {e}")
                _sleep_remaining(loop_start, frame_interval)
                continue

            if state is None:
                _sleep_remaining(loop_start, frame_interval)
                continue

            # ── 置信度检查 ───────────────────────────
            if state.confidence < min_confidence:
                logger.debug(f"[Main] 识别置信度过低 ({state.confidence:.2f})，跳过")
                _sleep_remaining(loop_start, frame_interval)
                continue

            # ── 非英雄回合，跳过决策 ─────────────────
            if not state.is_hero_turn:
                # 仍然更新对手模型
                _update_opponent_model(mods, state, last_hand_id)
                _sleep_remaining(loop_start, frame_interval)
                continue

            # ── 4. 胜率计算 ──────────────────────────
            try:
                equity_result = mods["equity"].calculate(
                    state.hole_cards,
                    state.community_cards,
                    num_opponents=len(state.active_opponents),
                )
                equity = equity_result["equity"]
            except Exception as e:
                logger.error(f"[Engine] 胜率计算失败: {e}")
                equity = 0.5

            # ── 5. GTO 决策 ──────────────────────────
            try:
                # 获取主要对手类型
                opp_seat = state.active_opponents[0].seat if state.active_opponents else -1
                opp_type = mods["opponent"].get_player_type(opp_seat)

                decision = mods["gto"].decide(state, equity, opp_type)
                logger.info(f"[Engine] 决策: {decision}")
            except Exception as e:
                logger.error(f"[Engine] 决策失败，默认弃牌: {e}")
                from state.models import Action, Decision
                decision = Decision(action=Action.FOLD, amount=0, reasoning="决策异常 fallback")

            # ── 6. 执行 ──────────────────────────────
            if not args.dry_run:
                try:
                    mods["executor"].execute(decision, game_frame.buttons)
                except Exception as e:
                    logger.error(f"[Executor] 执行失败: {e}")
            else:
                logger.info(f"[DryRun] 跳过执行: {decision}")

            # ── 7. 记录 ──────────────────────────────
            try:
                mods["logger"].log_decision(state, decision, equity_result)
                last_hand_id = state.hand_id
            except Exception as e:
                logger.warning(f"[Analytics] 记录失败（不影响运行）: {e}")

            # ── LLM 辅助（异步，不阻塞主循环）────────
            if mods["llm"].enabled and state.hand_id != last_hand_id:
                # 此处可改为线程池异步调用
                pass

            _sleep_remaining(loop_start, frame_interval)

    except KeyboardInterrupt:
        logger.info("\n用户中断，正在退出...")
    finally:
        if session_mgr:
            session_mgr.log_session()
        logger.info("poker-ai 已停止")


def _update_opponent_model(mods, state, last_hand_id):
    """在非英雄回合更新对手模型"""
    try:
        from state.models import Action
        for player in state.players:
            if player.seat != state.hero_seat and player.last_action:
                mods["opponent"].update(player.seat, player.last_action, state)
    except Exception:
        pass


def _sleep_remaining(start: float, interval: float) -> None:
    """补足帧间隔剩余时间"""
    elapsed = time.perf_counter() - start
    remaining = interval - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="poker-ai 德州扑克自动化系统")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--dry-run", action="store_true", help="仅决策，不执行鼠标")
    args = parser.parse_args()
    main(args)
