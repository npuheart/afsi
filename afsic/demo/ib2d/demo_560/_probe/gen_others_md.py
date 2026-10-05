"""Generate readme.md for demo_640..646 (Example_Jellyfish_Swimming family)."""
import os, re
import numpy as np

ROOT = "/Users/pengfei.ma/afsi/afsic/demo/ib2d"
L = os.path.join(ROOT, "demo_560", "_probe", "logs")

# demo -> dict of everything the readme needs
G = {
 640: dict(sub="Hoover_Jellyfish", title="Hoover_Jellyfish", ship=(192, 640, 3.0, 10.0, 1e-5, 5.0, 5000),
       run=(48, 1e-5, 2e-3, 20),
       phys="Hoover & Miller 的**原始水母模型**（`Example_Jellyfish_Swimming` 的主算例）："
            "`jelly.*` 512 分辨率几何、415 个拉格朗日点、357 根弹簧、317 根非不变梁、96 个目标点，"
            "钟体边缘 39 根弹簧以 `freq = 2 Hz` 收缩（`|cos(2 t pi)|`，每秒两次）",
       extra="本算例与水母收敛阶梯里的 `Re150/Res_512_640x192` **几何完全相同**（只差 `freq`：这里 2、阶梯里 1）。"
             "本批次另有一个并行 agent 的 demo_480 也搬了这个算例（不同设置），本 demo 是独立的一份。"),
 641: dict(sub="Tentacle_Jelly", title="Tentacle_Jelly", ship=(320, 768, 5.0, 12.0, 1e-5, 10.0, 5000),
       run=(48, 1e-5, 2e-3, 20),
       phys="带**触手**的水母（`5 x 12` 盒子）：`jelly512.*` 几何有 2267 根弹簧、2221 根非不变梁、"
            "**1910 个多孔弹性点**（`poroelastic = 1`），触手弹簧（MATLAB 行 `318+1 : 318+39`）以 "
            "`freq = 0.8 Hz` 摆动（`|cos(0.8 t pi)|`，周期 1.25 s）",
       extra="这是本批次唯一用到 `poroelastic` 机制的水母算例；模板的 poroelastic 分支与 pyIB2d 一致"
             "（同一份 `update_Springs.py` 两边共用）。"),
 642: dict(sub="Failed_Attempts/Jellyfish_Spring_RL_Interpolation", title="Failed_Attempts / Jellyfish_Spring_RL_Interpolation",
       ship=(128, 128, 1.0, 1.0, 2.5e-4, 0.2, 50), run=(64, 2.5e-4, 0.05, 20),
       phys="作者的**失败尝试**之一：`Lx = Ly = 1` 的小方盒里，把所有静长在两个“相位”之间**线性插值**"
            "（`RL = RL1 + 5 t (RL2 - RL1)`，相位差 0.2 s）——想用“直接改静长”来驱动水母",
       extra="**上游自己的不一致（本次实测）**：`jelly.spring` 里写的静长与插值表的“相位 1”并不相等"
             "（尾部 41 根：`max|file - RL1| = 0.124`；前 106 根：`max|file - ds1| = 0.146`），"
             "所以第 1 步就会把静长整体跳变一次，第一个帧 `max|u|` 立刻冲到 14 m/s——这大概正是它“失败”的原因之一。"
             "另外该几何的 `.spring` 只有 4 列（没有 `alpha`），两个代码都按线性弹簧（`alpha = 1`）处理。"),
 643: dict(sub="Failed_Attempts/Jellyfish_Springs_Only", title="Failed_Attempts / Jellyfish_Springs_Only",
       ship=(128, 128, 1.0, 1.0, 2.5e-4, 1.0, 50), run=(64, 2.5e-4, 0.05, 20),
       phys="作者的**失败尝试**之一：**只有弹簧**（没有梁、没有目标点，钟体没有弯曲刚度），"
            "钟体边缘 25 根弹簧的静长按 `distVec_i (1 - 0.75 sin(4 pi t))` 摆动"
            "（注意没有取绝对值：静长在 0.25x 与 1.75x 之间来回）",
       extra="本算例两个代码**吻合得最好**（速度幅值 ×0.99、相对 $L_2$ 7.8%、标记点 0.024 h）："
             "因为没有梁、没有目标点，力场完全来自弹簧，两边实现完全一致。"),
 644: dict(sub="Failed_Attempts/Jellyfish_With_Beams/Forward_Swimmer", title="Failed_Attempts / Jellyfish_With_Beams / Forward_Swimmer",
       ship=(128, 128, 1.0, 1.0, 5e-4, 2.0, 20), run=(64, 5e-4, 0.05, 10),
       phys="作者的**失败尝试**之一（`Forward_Swimmer`）：带 **77 根不变梁**（`jelly.beam`、`beams = 1`）的水母，"
            "但驱动律**只改变最后一根弹簧**的静长：`RL(end) = 0.1250077 (1 - 0.5 |sin(10 pi t)|)`"
            "（单弹簧、10 Hz）",
       extra="单弹簧驱动 + `mu = 0.001`（几乎无粘）的极端组合：两边都在“高速 + 欠分辨”的区间里，"
             "流场只能算部分可比。"),
 645: dict(sub="Failed_Attempts/Jellyfish_With_Beams/Backwards_Swimmer_for_LOLz", title="Failed_Attempts / Jellyfish_With_Beams / Backwards_Swimmer_for_LOLz",
       ship=(128, 128, 1.0, 1.0, 1e-3, 2.0, 10), run=(64, 1e-3, 0.05, 5),
       phys="作者的**失败尝试**之一（文件夹名就叫 `Backwards_Swimmer_for_LOLz`）：带 **105 根不变梁**，"
            "但驱动律**只改最后一根弹簧**：`RL(end) = 0.08649829 |cos(10 pi t)|`（20 Hz）",
       extra="`dt = 1e-3`（本批次最大的时间步）下两个代码仍稳定，且弹簧平均长度一致到 3.7e-4 相对误差。"),
 646: dict(sub="Failed_Attempts/Jellyfish_IBAMR_Convert", title="Failed_Attempts / Jellyfish_IBAMR_Convert",
       ship=(128, 1024, 2.0, 16.0, 1e-5, 12.0, 5000), run=(48, 1e-5, 2e-3, 20),
       phys="作者的**失败尝试**之一：一个从 **IBAMR** 转换过来的水母设置（`2 x 16` 的长条盒），"
            "钟体边缘 39 根弹簧的静长按 `|sin(0.875 t 2 pi)|` 驱动"
            "（注意这里是**完整的 `2 pi`** 正弦；其它水母算例用 `|cos(freq t pi)|`），周期 1.143 s",
       extra="上游目录里还有 `input2d_IBAMR`、`update_Springs_Cpp.m`（C++ 版驱动律）——"
             "搬运时只保留 `input2d` 与 Python 端需要的文件，`.m` 驱动律留作对照。"),
}


def tail_summary(path):
    txt = open(path).read()
    i = txt.rfind("=== shipped setup")
    if i < 0:
        i = txt.rfind("===")
    blk = txt[i:]
    j = blk.find("\nmarkers")
    return blk[:j].strip().splitlines(), [l for l in blk[j:].strip().splitlines()
                                          if not l.startswith("wrote")]


def stats(demo):
    A = np.load(os.path.join(ROOT, demo, "plot", "afsi_result.npz"))
    R = np.load(os.path.join(ROOT, demo, "ib2d_reference.npz"))
    dt = float(R["dt"])
    ia = {int(round(t / dt)): k for k, t in enumerate(A["t"])}
    ja = jr = None
    for k in range(len(R["t"]) - 1, -1, -1):
        j = int(round(R["t"][k] / dt)) + 1
        if j in ia:
            ja, jr = ia[j], k
            break
    ua, ur = A["u"][ja], R["u"][jr]
    Xa, Xr = A["X"][ja], R["X"][jr]
    du = np.linalg.norm(ua - ur, axis=-1)
    nu = np.linalg.norm(ur, axis=-1)
    m = nu > 0.01 * nu.max()
    r = du[m] / nu[m]
    c0 = R["X"][0].mean(0)
    d = dict(med=float(np.median(r)), p90=float(np.percentile(r, 90)), mx=float(r.max()),
             relL2=float(np.linalg.norm(du[m]) / np.linalg.norm(ur[m])), n=int(m.sum()),
             ratio=float(np.abs(ua).max() / np.abs(ur).max()),
             umax_a=float(np.abs(ua).max()), umax_r=float(np.abs(ur).max()),
             drift_a=float(np.linalg.norm(Xa.mean(0) - c0)),
             drift_r=float(np.linalg.norm(Xr.mean(0) - c0)),
             dy_a=float(Xa.mean(0)[1] - c0[1]), dy_r=float(Xr.mean(0)[1] - c0[1]),
             dX=float(np.abs(Xa - Xr).max()), h=float(A["dx"]), t=float(R["t"][jr]))
    return d


def real_time(path):
    mt = re.findall(r"^real\s+(\d+)m([\d.]+)s", open(path).read(), re.M) if os.path.exists(path) else None
    return int(mt[-1][0]) * 60 + float(mt[-1][1]) if mt else None


for num, g in sorted(G.items()):
    demo = "demo_%03d" % num
    cl = os.path.join(L, demo + ".compare.log")
    if not os.path.exists(cl):
        print("skip", demo)
        continue
    table, summary = tail_summary(cl)
    s = stats(demo)
    rt_ref, rt_afsi = real_time(os.path.join(L, demo + ".ref.log")), real_time(os.path.join(L, demo + ".afsi.log"))
    snx, sny, slx, sly, sdt, stend, spd = g["ship"]
    rnx, rdt, rtend, rpd = g["run"]
    rny = int(round(rnx * sly / slx))
    nsteps = int(round(rtend / rdt))
    amp = ("本例两个代码的**幅值一致**（AFSI %.3e / pyIB2d %.3e，**×%.2f**）"
           % (s["umax_a"], s["umax_r"], s["ratio"]) if abs(s["ratio"] - 1) < 0.05 else
           "AFSI 的 `max|u|` = %.3e，pyIB2d = %.3e（**×%.2f**）" % (s["umax_a"], s["umax_r"], s["ratio"]))
    txt = f"""# {demo} — IB2d `Example_Jellyfish_Swimming/{g['sub']}`

{g['phys']}。

{g['extra']}

`Example_Jellyfish_Swimming/` 是 IB2d 官方水母游动算例的总目录（模型出自
A. Hoover, L. Miller, *J. Theor. Biol.* 374: 13–25 (2015)）；其中 `Failed_Attempts/`
放的是**作者自己标注为失败的尝试**——搬运时一并保留并注明"失败在哪"，这是上游历史的一部分。

## 算例内容

* **出厂设置**（`ib2d_input/input2d` 原样保留）：网格 **{snx}x{sny}**（`{slx:g} x {sly:g}` 盒子）、
  `dt = {sdt:g}`、`Tfinal = {stend:g}`（= {int(round(stend / sdt)):,} 步）、`print_dump = {spd}`。
* **本 demo 运行设置**：`NX = {rnx}`（`NY` 由盒子长宽比自动推出 = {rny}）、`dt = {rdt:g}`（= 出厂值）、
  `Tfinal = {rtend:g}`（= {nsteps} 步）、`print_dump = {rpd}`。
  出厂网格/Tfinal 都远超本机可行范围（网格最大 128x1024、Tfinal 最大 12 s），见"移植要点"。
* **结构/驱动**：几何与驱动律文件（`*.vertex/.spring/.beam/.poroelastic`、`update_Springs.m`）逐字取自上游；
  `update_Springs.m` 已逐行移植成 `ib2d_input/update_Springs.py`，**两个代码共用同一份**。

## 移植要点

1. **机制**：`springs + update_springs`（按算例再加 `beams` / `nonInvariant_beams` / `target_pts` /
   `poroelastic`）——合并后的模板全部支持（同族参考 demo_480/482/520）。
2. **出厂 `dt` 是"参考解能用"的关键**：jelly 系列几何刚度极大，pyIB2d 的显式 IB 在大 `dt` 下不稳定
   （本批次实测：`jelly96` 几何在 `dt = 2e-5` 时 0.005–0.02 s 内除零 → NaN；`dt = 1e-4` 时 100 步内
   `max|u| ≈ 36 m/s`）。本家族出厂 `dt` 取 `1e-5 … 1e-3`，本 demo 一律沿用出厂值。
3. **`Failed_Attempts/` 的算例本来就跑不出推进**（作者如此标注）：本 demo 的价值是
   "驱动机制/耦合是否被正确搬运"，而不是"水母有没有游起来"。

## 结果

```
{chr(10).join(table)}

{chr(10).join(summary)}
```

按节点统计的速度场相对误差（`|u_ref| > 1% max|u_ref|` 的 {s['n']} 个节点，末对齐帧 `t = {s['t']:.5f}`）：
中位数 **{100 * s['med']:.1f}%**、90 分位 **{100 * s['p90']:.1f}%**、最大 **{100 * s['mx']:.0f}%**。

结构方面：标记点位置一致到 **{s['dX'] / s['h']:.1e} h**（= {s['dX']:.2e} m）；质心位移
AFSI **{s['drift_a']:.3e} m**（y 方向 {s['dy_a']:+.3e}）/ IB2d **{s['drift_r']:.3e} m**（y 方向 {s['dy_r']:+.3e}）。

**能验证**：结构运动学 / 弹簧（+梁）力学的搬运正确性，以及（除个别档外）瞬态流场的形状。

**不能验证（如实记录）**：

* 流场**绝对幅值**：{amp}。本窗口内 `mu ∇²u ≪ f`，流场主要由"冲量启动" `du/dt = P(f)/rho` 决定，
  AFSI（Taylor–Hood + Q2 插值力密度）与 IB2d（FFT 谱投影 + 共置罚力）在投影算子上的差异会被直接放大
  （与水母收敛阶梯一族现象相同：那一族系统性 ×1.37…×1.67）。
* **推进距离 / 游泳**：要看到完整周期需要 `T` 远大于本 demo 取的 {rtend:g} s；
  且 `Failed_Attempts/` 里的算例在上游就没有成功推进（本 demo 的质心位移只有
  {1e3 * s['drift_r']:.2f} mm 量级）。

## 运行

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate afsi-dolfinx

python run_reference.py --nx {rnx} --dt {rdt:g} --tend {rtend:g} --print-dump {rpd}   # 约 {("%.0f s" % rt_ref) if rt_ref else "?"}
NX={rnx} DT={rdt:g} TFINAL={rtend:g} PRINT_DUMP={rpd} python main.py                    # 约 {("%.0f s" % rt_afsi) if rt_afsi else "?"}
python compare.py
```

* 运行时长在**共享机器高负载**下测得（wall，含 ~5 s 导入）。
"""
    open(os.path.join(ROOT, demo, "readme.md"), "w").write(txt)
    print("wrote", os.path.join(ROOT, demo, "readme.md"))
