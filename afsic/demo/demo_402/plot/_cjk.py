"""matplotlib 中文字体设置（本机系统自带 Noto Sans CJK）。

在画图脚本里调用 `setup_cjk()`；找不到字体时静默降级（标签会变成方框，
但不会报错）。
"""
import os


def setup_cjk():
    try:
        import matplotlib
        from matplotlib import font_manager as fm
    except Exception:
        return False
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"):
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                matplotlib.rcParams["font.family"] = \
                    fm.FontProperties(fname=path).get_name()
                matplotlib.rcParams["axes.unicode_minus"] = False
                return True
            except Exception:
                continue
    return False
