#!/usr/bin/env python
"""固体实现核对工具（论文 rigid penalty / 刚体假设 / 体积变化）。

用法：python plot/solid_rigidity_check.py <run_dir> [<run_dir> ...]
读取 <run_dir>/solid.h5（dolfinx 输出：Mesh/mesh/{topology,geometry} +
Function/solid_coords_io/<t> 与 solid_force_io/<t>），输出：

  1. 圆柱（圆心 (20,20) R=5 cm）的刚体运动分解：平移、转角 θ、相对刚体运动的
     残余变形 RMS（= 论文"刚性"假设的偏差度量）；
  2. 单元面积比 |J-1| 的分位数（圆柱 / 梁），对应论文 volumetric penalization 关注点；
  3. 系绳力方向检查：corr(F_cyl, X0 - chi)（>0 表示力拉回参考位形，即论文
     F = kappa(psi - chi) 的恢复力）。
"""
import sys

import numpy as np
import h5py

CYL_CENTER = (20.0, 20.0)
CYL_R = 5.0


def load(path):
    with h5py.File(path, "r") as f:
        ks = sorted(f["Function/solid_coords_io"].keys(),
                    key=lambda s: float(s.replace("_", ".")))
        fks = sorted(f["Function/solid_force_io"].keys(),
                     key=lambda s: float(s.replace("_", ".")))
        topo = f["Mesh/mesh/topology"][:].astype(np.int64)
        # 参考构型：h5 里写入的网格几何（与 run_compare 的 tabulate_dof_coordinates 口径一致）
        geom = f["Mesh/mesh/geometry"][:][:, :2]
        t = [float(k.replace("_", ".")) for k in ks]
        coords = [f[f"Function/solid_coords_io/{k}"][:][:, :2] for k in ks]
        Xf = f[f"Function/solid_force_io/{fks[-1]}"][:][:, :2]
    return topo, geom, t, coords, Xf


def area(p):
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def rigid_fit(Xm, dm):
    """最佳刚体运动（平移 t + 旋转 θ）拟合；返回 t, θ, 残余 RMS。"""
    c = Xm.mean(axis=0)
    r = Xm - c
    n = len(r)
    A = np.zeros((2 * n, 3))
    A[0::2, 0] = 1.0
    A[1::2, 1] = 1.0
    A[0::2, 2] = -r[:, 1]
    A[1::2, 2] = r[:, 0]
    sol, *_ = np.linalg.lstsq(A, dm.reshape(-1), rcond=None)
    res = dm - (A @ sol).reshape(-1, 2)
    return sol[0:2], sol[2], float(np.sqrt((res**2).mean()))


def main(runs):
    for run in runs:
        path = f"{run}/solid.h5"
        try:
            topo, X0, ts, coords, Xf = load(path)
        except (OSError, KeyError) as e:
            print(f"[skip] {run}: {e}")
            continue
        cen = X0[topo].mean(axis=1)
        cyl_pts = ((X0[:, 0] - CYL_CENTER[0])**2
                   + (X0[:, 1] - CYL_CENTER[1])**2) <= CYL_R**2 + 1e-9
        cyl_cell = (((cen[:, 0] - CYL_CENTER[0])**2
                     + (cen[:, 1] - CYL_CENTER[1])**2) < CYL_R**2)
        print(f"\n=== {run} （帧数 {len(ts)}，t_end={ts[-1]:.4f} s）===")
        print(f"{'t':>6} {'漂移|t| [cm]':>12} {'θ [rad]':>10} "
              f"{'残余变形RMS [cm]':>16} {'梁尖|dy| [cm]':>13}")
        for i in range(0, len(ts), max(1, len(ts) // 10)):
            d = coords[i] - X0
            tr, th, res = rigid_fit(X0[cyl_pts], d[cyl_pts])
            tip = X0[:, 0] > 59.0
            print(f"{ts[i]:6.2f} {np.linalg.norm(tr):12.3e} {th:10.3e} "
                  f"{res:16.3e} {np.abs(d[tip][:, 1]).max():13.4f}")
        d = coords[-1] - X0
        for tag, pts, cells in [("cylinder", cyl_pts, cyl_cell),
                                ("beam", ~cyl_pts, ~cyl_cell)]:
            J = area(coords[-1][topo])[cells] / area(X0[topo])[cells]
            q = np.percentile(np.abs(J - 1), [50, 90, 99, 100])
            print(f"{tag:9s} 单元 |J-1|: 中位 {q[0]:.4f} | 90% {q[1]:.4f} | "
                  f"99% {q[2]:.4f} | max {q[3]:.4f}")
        corr = np.corrcoef(Xf[cyl_pts].ravel(), (-d[cyl_pts]).ravel())[0, 1]
        print(f"corr(F_cyl, X0-chi) = {corr:+.4f}  (>0: 系绳为恢复力)")


if __name__ == "__main__":
    main(sys.argv[1:])
