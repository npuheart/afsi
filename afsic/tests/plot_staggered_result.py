"""解析交错版 VTU 并画结果图。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

vtu = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sphere_interp_staggered.vtu")
tree = ET.parse(vtu)
root = tree.getroot()
ns = {"v": "http://www.w3.org/1999/xhtml"}
# 无命名空间，直接找
piece = root.find(".//Piece")
def get_data_array(piece, name):
    for da in piece.findall(".//DataArray"):
        if da.get("Name") == name:
            return np.array([float(x) for x in da.text.split()]).reshape(-1, int(da.get("NumberOfComponents", "1")))
pts = get_data_array(piece, None) if False else None
# 手动：Points 是第一个无 Name 的 DataArray
for da in piece.findall("./Points/DataArray"):
    pts = np.array([float(x) for x in da.text.split()]).reshape(-1, 3)
h = get_data_array(piece, "h_ib")
g = get_data_array(piece, "g_exact")
err = get_data_array(piece, "error")

print("点数:", pts.shape[0])
print("h 范围:", h.min(), h.max(), "| g 范围:", g.min(), g.max())
en = np.linalg.norm(err, axis=1)
print("error 范数: max=%.4e mean=%.4e median=%.4e" % (en.max(), en.mean(), np.median(en)))
print("L2 相对误差:", np.linalg.norm(err)/np.linalg.norm(g))

out = os.path.dirname(os.path.abspath(__file__))
# ---- 图1: y≈0.5 切片上 error 范数分布 ----
sl = np.abs(pts[:,1] - 0.5) < 0.05
fig, ax = plt.subplots(figsize=(7,5))
sc = ax.scatter(pts[sl,0], pts[sl,2], c=en[sl], s=6, cmap="jet")
plt.colorbar(sc, label="|error|")
ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_title("staggered IB interp: |error| on y≈0.5 slice")
ax.set_aspect("equal"); fig.tight_layout()
fig.savefig(out + "/staggered_error_slice.png", dpi=150); plt.close(fig)

# ---- 图2: h vs g 散点 + 45° 线（3 分量合并） ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for c, col in zip(range(3), ["tab:blue","tab:orange","tab:green"]):
    ax[0].scatter(g[:,c], h[:,c], s=3, alpha=0.4, color=col, label=f"comp {c}")
lim = [g.min(), g.max()]
ax[0].plot(lim, lim, "k--", lw=1, label="y=x")
ax[0].set_xlabel("g (exact)"); ax[0].set_ylabel("h (IB interp)"); ax[0].set_title("h vs g")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].hist(en, bins=60, color="tab:red", alpha=0.8)
ax[1].set_xlabel("|error|"); ax[1].set_ylabel("count"); ax[1].set_title("error histogram")
fig.tight_layout(); fig.savefig(out + "/staggered_h_vs_g.png", dpi=150); plt.close(fig)
print("图已保存:", out + "/staggered_error_slice.png", out + "/staggered_h_vs_g.png")
