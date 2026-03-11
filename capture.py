"""
vision/capture.py — 屏幕捕获模块
使用 mss（比 pyautogui 快 3-5×），支持多显示器与窗口自动定位。
"""
from __future__ import annotations

import sys
from typing import Optional

import mss
import mss.tools
import numpy as np
from loguru import logger


class ScreenCapture:
    """
    捕获扑克客户端窗口区域。
    支持：多显示器 / 指定窗口句柄 / ROI 区域截图
    输出：numpy.ndarray (BGR, HWC)
    """

    def __init__(self, config: dict) -> None:
        self.monitor_idx = config.get("monitor_index", 1)
        self.window_title = config.get("window_title", "")
        self.fps_limit = config.get("fps_limit", 8)
        self.regions = config.get("regions", {})

        self._sct = mss.mss()
        self._monitor = self._resolve_monitor()
        logger.info(f"[Vision] ScreenCapture 初始化完成，监视器: {self._monitor}")

    # ── 公开接口 ──────────────────────────────────────

    def grab(self) -> np.ndarray:
        """截取整个目标监视器"""
        return self._grab_monitor(self._monitor)

    def grab_region(self, region: tuple[float, float, float, float]) -> np.ndarray:
        """
        截取相对坐标区域 (x1, y1, x2, y2)，范围 0.0–1.0。
        返回该区域的 BGR numpy 数组。
        """
        mon = self._monitor
        w, h = mon["width"], mon["height"]
        x1, y1, x2, y2 = region
        abs_region = {
            "left":   mon["left"] + int(x1 * w),
            "top":    mon["top"]  + int(y1 * h),
            "width":  int((x2 - x1) * w),
            "height": int((y2 - y1) * h),
        }
        return self._grab_monitor(abs_region)

    def grab_named_region(self, name: str) -> Optional[np.ndarray]:
        """按配置名称截取 ROI（hole_cards / community_cards / pot / ...）"""
        region = self.regions.get(name)
        if region is None:
            logger.warning(f"[Vision] 未知区域名称: {name}")
            return None
        return self.grab_region(tuple(region))

    def grab_all_regions(self) -> dict[str, np.ndarray]:
        """一次性截取所有预定义 ROI，减少重复截图开销"""
        full = self.grab()
        result = {}
        mon = self._monitor
        w, h = mon["width"], mon["height"]
        for name, (x1, y1, x2, y2) in self.regions.items():
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            result[name] = full[py1:py2, px1:px2]
        return result

    # ── 内部方法 ──────────────────────────────────────

    def _grab_monitor(self, monitor: dict) -> np.ndarray:
        """截图并转换为 BGR numpy 数组"""
        sct_img = self._sct.grab(monitor)
        # mss 返回 BGRA，去掉 alpha 通道
        frame = np.array(sct_img)[:, :, :3]
        return frame

    def _resolve_monitor(self) -> dict:
        """
        解析目标监视器。
        优先尝试通过窗口标题定位；失败则使用 monitor_index。
        """
        # 尝试通过窗口标题定位（仅 Windows）
        if self.window_title and sys.platform == "win32":
            mon = self._find_window_monitor(self.window_title)
            if mon:
                return mon

        monitors = self._sct.monitors  # monitors[0] 是虚拟全屏，从 1 开始
        idx = min(self.monitor_idx, len(monitors) - 1)
        return monitors[idx]

    def _find_window_monitor(self, title: str) -> Optional[dict]:
        """Windows 专属：通过窗口标题获取窗口边界作为截图区域"""
        try:
            import win32gui

            def callback(hwnd, ctx):
                if title.lower() in win32gui.GetWindowText(hwnd).lower():
                    rect = win32gui.GetWindowRect(hwnd)
                    ctx.append({
                        "left":   rect[0],
                        "top":    rect[1],
                        "width":  rect[2] - rect[0],
                        "height": rect[3] - rect[1],
                    })

            windows = []
            win32gui.EnumWindows(callback, windows)
            if windows:
                logger.info(f"[Vision] 找到窗口 '{title}': {windows[0]}")
                return windows[0]
        except ImportError:
            pass
        return None

    def __del__(self):
        try:
            self._sct.close()
        except Exception:
            pass
