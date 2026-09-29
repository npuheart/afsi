# post426 — demo_426 原始运行备份

demo_426（斜通道 IB 基准）**系绳符号修复**与**压力驱动（DRIVING=pressure）验证**
的原始运行产物，由 `/tmp/post426` 移入仓库根目录（2026-09-28）。

- 整理归档（含 `summary.md` 说明与图）：`afsic/demo/demo_426/plot/signfix-20260928/`
- `r1–r7`、`r4b`：系绳修复阶段各运行的日志与轨迹（`*_hist.csv`；与归档目录
  内容一致）。
- `pp/`：压力驱动测试 —— `p1` 压力 exact+ipcs、`p2` 压力 const+ipcs、`p3` 压力
  exact+chorin、`p4` 速度驱动回归；`p1_fields/`、`r1_fields/` 为完整场输出
  （velocity/pressure/solid 的 XDMF + h5，可直接进 ParaView）。
- `field_run.log`、`r7b.log`：补跑记录。
- 注：`r1_fields/` 与归档 `fields/` 的场文件为 **2026-09-29 压力基准（gauge）
  修正**后重新生成的版本（提交 `8ef2c9a`）；历史指标见归档 `summary.md`。

相关提交：`91e43e1`（系绳符号修复）、`f50e998`（pressure 驱动）、`8ef2c9a`（速度驱动压力基准）。
