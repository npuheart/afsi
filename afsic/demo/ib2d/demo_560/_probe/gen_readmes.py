"""Generate the Chinese readme.md of the Jellyfish convergence-ladder demos (560-569, 633-639)."""
import os, re
import numpy as np

ROOT = "/Users/pengfei.ma/afsi/afsic/demo/ib2d"
L = os.path.join(ROOT, "demo_560", "_probe", "logs")

# demo -> (Re, Res, first_row, mu, ship_nx, ship_ny, name, nv, ns, nb, nt, kmax, kb, rlmin, rlmax)
F = {
 560: ("Re37pt5", "Res_96_120x36", 59, 26.666, 288 if os.environ.get("CAP") else 36, 960 if os.environ.get("CAP") else 120, "jelly96", 91, 65, 57, 18, 2.4e8, 3.46e9, 0.0, 4.23e-2),
 561: ("Re37pt5", "Res_768_960x288", 477, 26.666, 288, 960, "jelly768", 739, 535, 475, 144, 1.92e9, 1.77e12, 0.0, 5.22e-3),
 562: ("Re37pt5", "Res_512_640x192", 319, 26.666, 192, 640, "jelly512", 493, 357, 317, 96, 1.28e9, 5.24e11, 0.0, 7.83e-3),
 563: ("Re37pt5", "Res_48_60x18", 29, 26.666, 18, 60, "jelly48", 44, 31, 27, 9, 1.2e8, 4.32e8, 0.0, 8.58e-2),
 564: ("Re37pt5", "Res_384_480x144", 239, 26.666, 144, 480, "jelly384", 369, 267, 237, 72, 9.6e8, 2.21e11, 0.0, 1.05e-2),
 565: ("Re37pt5", "Res_256_320x96", 159, 26.666, 96, 320, "jelly256", 245, 177, 157, 48, 6.4e8, 6.55e10, 0.0, 1.57e-2),
 566: ("Re150", "Res_192_240x72", 119, 6.667, 72, 240, "jelly192", 183, 132, 117, 36, 4.8e8, 2.76e10, 0.0, 2.1e-2),
 567: ("Re150", "Res_64_80x24", 39, 6.667, 24, 80, "jelly64", 59, 42, 37, 12, 1.6e8, 1.02e9, 0.0, 6.39e-2),
 568: ("Re150", "Res_32_40x12", 19, 6.667, 12, 40, "jelly32", 29, 20, 17, 6, 8.0e7, 1.28e8, 0.0, 1.31e-1),
 569: ("Re150", "Res_128_160x48", 79, 6.667, 48, 160, "jelly128", 121, 87, 77, 24, 3.2e8, 8.19e9, 0.0, 3.16e-2),
 633: ("Re150", "Res_1024_1280x384", 637, 6.667, 384, 1280, "jelly1024", 987, 715, 635, 192, 2.56e9, 4.19e12, 0.0, 3.91e-3),
 634: ("Re150", "Res_96_120x36", 59, 6.667, 36, 120, "jelly96", 91, 65, 57, 18, 2.4e8, 3.46e9, 0.0, 4.23e-2),
 635: ("Re150", "Res_768_960x288", 477, 6.667, 288, 960, "jelly768", 739, 535, 475, 144, 1.92e9, 1.77e12, 0.0, 5.22e-3),
 636: ("Re150", "Res_512_640x192", 319, 6.667, 192, 640, "jelly512", 493, 357, 317, 96, 1.28e9, 5.24e11, 0.0, 7.83e-3),
 637: ("Re150", "Res_48_60x18", 29, 6.667, 18, 60, "jelly48", 44, 31, 27, 9, 1.2e8, 4.32e8, 0.0, 8.58e-2),
 638: ("Re150", "Res_384_480x144", 239, 6.667, 144, 480, "jelly384", 369, 267, 237, 72, 9.6e8, 2.21e11, 0.0, 1.05e-2),
 639: ("Re150", "Res_256_320x96", 159, 6.667, 96, 320, "jelly256", 245, 177, 157, 48, 6.4e8, 6.55e10, 0.0, 1.57e-2),
}
DT, T, PD, NSTEPS = 1e-5, 2e-3, 20, 200
BATCH_RATIO = "×1.37（36x120 与 48x160 网格）… ×1.67（12x40 网格）"


BST = {}


def _fmt(sec):
    return "%.0f s" % sec if sec is not None else "(未记录)"


def tail_summary(path):
    txt = open(path).read()
    i = txt.rfind("=== shipped setup")
    if i < 0:
        i = txt.rfind("===")
    blk = txt[i:]
    j = blk.find("\nmarkers")
    return blk[:j].strip().splitlines(), [l for l in blk[j:].strip().splitlines()
                                          if not l.startswith("wrote")]


def aligned_pair(demo):
    """Last (AFSI, IB2d) frame pair with round(t_a/dt) == round(t_r/dt) + 1."""
    A = np.load(os.path.join(ROOT, demo, "plot", "afsi_result.npz"))
    R = np.load(os.path.join(ROOT, demo, "ib2d_reference.npz"))
    dt = float(R["dt"])
    ia = {int(round(t / dt)): k for k, t in enumerate(A["t"])}
    for k in range(len(R["t"]) - 1, -1, -1):
        j = int(round(R["t"][k] / dt)) + 1
        if j in ia:
            return ia[j], k, A, R
    return None, None, A, R


def field_stats(demo):
    ja, jr, A, R = aligned_pair(demo)
    if ja is None:
        return None
    ua, ur = A["u"][ja], R["u"][jr]
    du = np.linalg.norm(ua - ur, axis=-1)
    nu = np.linalg.norm(ur, axis=-1)
    m = nu > 0.01 * nu.max()
    r = du[m] / nu[m]
    return dict(med=float(np.median(r)), p90=float(np.percentile(r, 90)), mx=float(r.max()),
                relL2=float(np.linalg.norm(du[m]) / np.linalg.norm(ur[m])), n=int(m.sum()),
                ratio=float(np.abs(ua).max() / np.abs(ur).max()),
                t=float(R["t"][jr]), dX=float(np.abs(A["X"][ja] - R["X"][jr]).max()))


def batch_status():
    """demo -> (finite_ref, umax_ref) parsed from the batch driver logs."""
    out = {}
    for f in ("batch1.txt", "batch2.txt"):
        fp = os.path.join(ROOT, "demo_560", "_probe", f)
        if not os.path.exists(fp):
            continue
        for line in open(fp):
            m = re.search(r"(demo_\d{3}) ref rc=(\d+) ref frames=(\d+) finite=(\w+) umax=([\d.eE+-]+)", line)
            if m:
                out[m.group(1)] = (m.group(4) == "True", float(m.group(5)))
    return out


def real_time(path):
    mt = re.findall(r"^real\s+(\d+)m([\d.]+)s", open(path).read(), re.M) if os.path.exists(path) else None
    return int(mt[-1][0]) * 60 + float(mt[-1][1]) if mt else None


BST.update(batch_status())

for num, (re_lab, res, first, mu, snx, sny, name, nv, ns, nb, nt, kmax, kb, rlmin, rlmax) in sorted(F.items()):
    demo = "demo_%03d" % num
    d = os.path.join(ROOT, demo)
    cl = os.path.join(L, demo + ".compare.log")
    if not os.path.exists(cl):
        print("skip (no compare)", demo)
        continue
    table, summary = tail_summary(cl)
    s = field_stats(demo)
    if s is None:
        print("skip (no aligned frames)", demo)
        continue
    rt_ref, rt_afsi = real_time(os.path.join(L, demo + ".ref.log")), real_time(os.path.join(L, demo + ".afsi.log"))
    fin, umax_ref = BST.get(demo, (None, None))
    stable = ("本 demo 实测：出厂 `dt = 1e-5`、200 步内参考解保持有限（末步 `max|u| = %.3e`）。" % umax_ref
              if fin else "本 demo 实测：该 dt 下参考解出现 NaN（见“不能验证”）。")
    rnx = min(snx, 48)
    rny = int(round(rnx * 10.0 / 3.0))
    h = 3.0 / rnx
    re_str = "Re = %g" % {"Re37pt5": 37.5, "Re150": 150.0}[re_lab]
    capped = snx > 48
    if capped:
        grid_line = (f"4. **运行网格**（本 demo 唯一的“降规模”操作）：上游出厂网格 **{snx}x{sny}**"
                     f"（{snx * sny // 1000}k 个节点，`h = {3.0 / snx:.4f}`）远超本机可行范围，"
                     f"本 demo 降到 **{rnx}x{rny}**（`h = {h:.4f}`），即收敛阶梯里 **Res_128** 那一档的网格；\n"
                     f"   上游 `input2d`、`{name}.*` 几何/结构文件**逐字保留**在 `ib2d_input/`。")
    else:
        grid_line = (f"4. **运行网格**：**{rnx}x{rny}**（`h = {h:.4f}`），与出厂网格一致"
                     f"（该档本来就是阶梯里较小的若干档之一）；上游文件逐字保留在 `ib2d_input/`。")
    txt = f"""# {demo} — IB2d `Examples_Education/Convergence/Jellyfish/Simulation_Skeletons/{re_lab}/{res}`

IB2d 官方**水母游动收敛性阶梯**（`Convergence/Jellyfish/Simulation_Skeletons`）的一个成员：
同一只水母在 4 个 Reynolds 数（`Re37pt5`、`Re75`、`Re150`、`Re300`）下各做 11 档
拉格朗日/网格分辨率的收敛性研究（`Res_32 … Res_1024`）。
本 demo 是 **{re_lab}（{re_str}）**、**{res}** 那一档；本搬运批次覆盖 `Re37pt5` 的 6 档
与 `Re150` 的全部 11 档（demo_560–569 / 633–639），`Re75`/`Re300` 不在本批次内。

模型出处（上游 `README.txt`）：A. Hoover, L. Miller, *A numerical study of the benefits of driving
jellyfish bells at their natural frequency*, J. Theor. Biol. 374: 13–25 (2015)；
matIB2d 版本由 N. A. Battista 制作（Feb. 2018, TCNJ）。

## 算例内容

* 周期盒 `3 x 10`（`Lx = 3`、`Ly = 10`）、`rho = 1000`、**`mu = {mu:g}`（{re_str}）**。
* **出厂设置**：网格 **{snx}x{sny}**（`h = {3.0 / snx:.4f}`）、`dt = 1e-5`、`Tfinal = 5.0`
  （= **5e5 步**）、`print_dump = 5000`（1000 帧）、`supp = 4`。
* **结构**（`ib2d_input/{name}.vertex / .spring / .nonInv_beam / .target`，逐字取自上游）：
  **{nv}** 个拉格朗日点、**{ns}** 根弹簧（`k ≤ {kmax:.3g}`、静长 `{rlmin:g} … {rlmax:.3g}`）、
  **{nb}** 根非不变梁（`kb = {kb:.3g}`）、**{nt}** 个钉死目标点。
  水母钟体几乎占满整个盒子（`x ∈ [0.004, 2.99]`、`y ∈ [1.75, 9.98]`）——它是"贴在周期边界上的大水母"。
* **收缩律**（`update_Springs.m`，逐行移植为 `ib2d_input/update_Springs.py`，两个代码**共用同一份**）：
  钟体边缘的 **{ns - first + 1}** 根弹簧（MATLAB 行 `{first}:end`）静长每步重置为

  ```
  RL_i = |cos(freq t pi)|,   freq = 1
  ```

  周期 1 s、取绝对值 ⇒ **每秒收缩两次**。它只改静长、不改刚度，是纯"肌肉"驱动。
  （阶梯各档的起止行号随分辨率变化，出厂 `.m` 里写的就是 `{first}:end`。）

## 移植要点

1. **机制**：`springs + update_springs + nonInvariant_beams + target_pts` —— 合并后的模板全部支持
   （同族参考 demo_480/482）；`update_Springs.py` 的写法与 demo_478/480 同族。
   本档唯一的参数是 `FIRST_ROW = {first}`。
2. **两端共用同一份用户模块**：`run_reference.py` 把 `ib2d_input/*.py` 拷进运行目录给 pyIB2d 用，
   `main.py` 从 `ib2d_input/` 导入 ⇒ "收缩律"不可能两边不一致。
3. **dt 与稳定性**（实测）：该几何刚度极大（`k ≤ {kmax:.3g}`、`kb = {kb:.3g}`），
   pyIB2d 的显式 IB 在 `dt` 过小时步长会失稳。本批次在 `jelly96` 几何 + 48x160 网格上的实测：
   `dt = 2e-5` 时 `t ≈ 0.005–0.02 s` 内出现除零 → NaN；`dt = 1e-4` 时 100 步内 `max|u|` 已达 36 m/s（发散）；
   `dt = 1e-5`（出厂值）稳定。{stable}
{grid_line}
5. **无法复现游泳**：出厂 `Tfinal = 5 s`（5e5 步），即便 `dt = 1e-5` 也是小时级运行；
   本 demo 只跑 **{T:g} s（{NSTEPS} 步，`print_dump = {PD}` → 10 帧）**，
   即收缩周期（1 s）的 **{100 * T:.1f}%**，只覆盖**初期瞬态**。

## 结果

`run_reference.py` + `main.py` + `compare.py`（两边各 10 帧，`compare.py` 按驱动器的 dump 约定对齐）：

```
{chr(10).join(table)}

{chr(10).join(summary)}
```

按节点统计的速度场相对误差（取 `|u_ref| > 1% max|u_ref|` 的 {s['n']} 个节点，
末对齐帧 `t = {s['t']:.5f}`）：中位数 **{100 * s['med']:.1f}%**、90 分位 **{100 * s['p90']:.1f}%**、
最大 **{100 * s['mx']:.0f}%**；该子集上的相对 $L_2$ = **{s['relL2']:.3f}**。

**能验证**（本 demo 的正面结论）：

* **结构运动学**：标记点位置一致到 `max|dX| ≈ {s['dX'] / h:.0e} h`（= {s['dX']:.2e} m），
  弹簧平均长度相对误差 ~1e-6 ⇒ 水母受到的力、以及被力推着走的标记点轨迹，两边是同一条。
* **瞬态流场的形状**：绝大多数节点的相对误差在 {100 * s['med']:.0f}%–{100 * s['p90']:.0f}% 量级。

**不能验证（如实记录）**：

* **流场绝对幅值**：AFSI 的 `max|u|` 系统性高于 pyIB2d，**本例 ×{s['ratio']:.2f}**，
  本批次各档落在 {BATCH_RATIO}；两套网格上量级不变 ⇒ 不是时间步/网格分辨率问题，而是
  **两种离散对同一个力场的高波数响应不同**（AFSI：Taylor–Hood P2/P1 + 力密度经 Q2 插值；
  IB2d：FFT 谱投影 + 共置罚力）。该窗口内 `mu ∇²u ≪ f`，流场几乎完全由"冲量启动"
  `du/dt = P(f)/rho` 决定，所以投影算子的差异被直接放大。因此本 demo **不宣称**速度场幅值可比。
* **游泳/推进**：要看到完整的收缩—回复循环与尾涡需要 `T ~ 1 s`（10^5 步）以上，
  本 demo 只跑 {T:g} s（收缩周期的 {100 * T:.1f}%）。也没有做阶梯的跨档收敛率分析
  （那需要全部 4 个 Re × 11 档、且各档都在自己的出厂网格上跑完整周期）。

## 运行

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate afsi-dolfinx

# 参考解（pyIB2d，出厂 dt）
python run_reference.py --nx {rnx} --dt {DT:g} --tend {T:g} --print-dump {PD}   # 约 {_fmt(rt_ref)}

# AFSI 移植
NX={rnx} DT={DT:g} TFINAL={T:g} PRINT_DUMP={PD} python main.py                    # 约 {_fmt(rt_afsi)}

# 对比（图 + compare_table.csv）
python compare.py
```

* 上游出厂：`dt = 1e-5`、`Tfinal = 5.0`、网格 {snx}x{sny}；本 demo 只改运行规模（见"移植要点"），
  数据文件一律逐字保留。
* 运行时长是**共享机器高负载**下测得的 wall time（含 ~5 s 的 numpy/dolfinx 导入），仅供参考。
"""
    fn = os.environ.get("README_OUT", "readme.md")
    open(os.path.join(d, fn), "w").write(txt)
    print("wrote", os.path.join(d, fn))
