# demo_426 系绳符号修复 — 验证结果（2026-09-28）

全部运行：serial、`SMOKE=1`、显式指定 `SOLVER`/`BETA`（tether 路线为
`USE_IMPLICIT_DRAG=0`）。配置：N=32, dx=0.03125, dt=0.2*dx, ipcs 默认。
数字口径：`err_L2_rel_channel` 为通道内相对 L2（解析解），滑移=板面 max|u|，
漂移=板相对参考位置的最大位移。

| 运行 | 配置（步数） | err_L2_rel_channel | 板滑移 [m/s] | 板漂移 [m] | 备注 |
|---|---|---|---|---|---|
| `before_tether_ipcs` | 改动前基准，ipcs tether β=8e3（320） | 0.00627728 | 0.00304348 | 0.01122926 | 与 r1 **逐位一致** |
| `r1` | 同上（改后默认载荷，320） | 0.00627728 | 0.00304348 | 0.01122926 | 编辑对外科式 |
| `r2` | ipcs tether + `IB_DIRECT_LOAD=1`（320） | 0.00756917 | 0.00331252 | 0.01128408 | 新方法；vs r1 轨迹差 ≤1.3%(max rel) |
| `r3` | chorin tether（调用修复后，320） | 0.02044277 | 0.01929792 | 0.02745412 | chorin 分支此前无法运行 |
| `r4b` | 隐式 drag 配方 ipcs（320） | 0.05334737 | 0.00525073 | 0.0 | 原推荐配方复现（5.33%） |
| `r5` | ipcs tether+direct 长跑（1600，t=10s） | 0.01143501 | 0.00444287 | 0.03964077 | 残余爬移 ~0.1 格/s |
| `r6` | ipcs tether+direct β=1.2e4（800，t=5s） | 0.00892936 | 0.00217700 | 0.01751301 | 预算内高刚度 |
| `r7` | ipcs tether β=**−8e3**（=旧符号等效，320） | 3.932（发散） | 1.2529 | 3.2664（209×h/2） | 反号闭环对照 💥 |

文件说明：
- `r*.log`：各运行的完整控制台报告（含 verify 明细与 elapsed）。
- `r*_hist.csv`：`step,t,max_u,plate_disp` 轨迹。
- `r4b_verify.json`：drag 配方完整 verify.json。
- `before_*`：改动前基准（ipcs tether 逐位一致）与改前 chorin 崩溃记录
  （缺 `bcp` 位置参数的 TypeError）。
- 注：`post426/r4.log|r4_hist.csv`（原 `/tmp/post426/`）为一次受 shell 残留
  `SOLVER` 污染的运行，**未收录**，正确结果见 r4b。
- `r7_hist.csv`：反号对照的发散轨迹（重跑补存；末态漂移 3.2664 m 与首次运行一致，
  max|u| 峰值 19.7 m/s）。
- `plot_signfix.py` → `figures/signfix_histories.png`：四联验证图（板漂移曲线、
  r7 发散曲线、max|u|、通道误差柱图）。复现：`cd signfix-20260928 && python plot_signfix.py`。

正式记录（已提交 91e43e1）：
- `afsic/demo/demo_426/readme.md` — “What limits the explicit (tether) route — revised 2026-09-28”
- `docs/demo-426-ib-coupling-findings.md` — §7（反号诊断与闭环证据）

原始位置备份：`post426/`（**仓库根目录**，由 `/tmp/post426/` 移入；日志/轨迹
全量 ~2.7 MB）、`/tmp/ref426_smoke/`、`/tmp/ref426_N32/`
（被运行覆盖前的已提交 plot 输出原件）。

## 相关：压力驱动测试（`DRIVING=pressure`，提交 `f50e998`）

运行数据在仓库根 **`post426/pp/`**：
- `p1` 压力 exact + ipcs、`p2` 压力 const + ipcs、`p3` 压力 exact + chorin、
  `p4` 速度驱动回归（与 r1 逐位一致）——各含 `.log` 与 `_hist.csv`（321 行）。
- `p1_fields/`、`r1_fields/`：完整场输出（velocity/pressure/solid 的 XDMF + h5，
  可直接进 ParaView）。
- 关键数字：通道内相对 L2 = **0.63%**（速度驱动）vs **7.9% / 17.2%**（压力
  exact / const）；窗口诊断全窗/中段/核心：速度 0.63/0.61/0.56%、压力
  7.87/4.78/4.20%。原因：30° 斜切口上零粘性牵引自然条件与解析解失配（边界条件
  固有代价），详见 `afsic/demo/demo_426/readme.md` “Driving modes”。
- 注（2026-09-29）：速度驱动新增压力基准（gauge）——纯 Neumann 系统钉一个参考
  压力自由度（p≡0，通道内对角 3 格节点）。修正前存出的压力场带 ~7.8e11 未定
  常数（`p_ += phi` 累积漂移）；本目录 `fields/` 与 `post426/pp/r1_fields/`
  已用修正后的运行重新生成（通道内 ≈ 0.989·p_解析 + 0.33，corr 0.9995；速度
  指标变化 ≤0.4%）。
