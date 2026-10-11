#!/usr/bin/env bash
# 论文（IJNMBE）所需全部 2D 计算。用法：
#   conda activate afsi-dolfinx
#   cd afsic/demo/paper_runs
#   NP=8 bash run_paper.sh            # 全部
#   NP=8 bash run_paper.sh annulus    # 只跑某一组：annulus | disk | valve
# 结果写到 paper_runs/results/，跑完把整个 results/ 留在原处即可（Claude 能直接读取）。
set -uo pipefail   # 单个工况失败不中断，见 results/run.log
HERE="$(cd "$(dirname "$0")" && pwd)"
DEMO="$(cd "$HERE/.." && pwd)"
RES="$HERE/results"; mkdir -p "$RES"
NP="${NP:-8}"
MPI="mpirun -n $NP"
GAMMA_DISK="${GAMMA_DISK:-1.0}"     # 圆盘算例的 grad-div 系数（与 mu=0.01 同量级起试）
GAMMA_VALVE="${GAMMA_VALVE:-0}"     # 瓣膜消融默认不加 grad-div，只比较铺展方式
WHAT="${1:-all}"
# 离线运行（不连 swanlab / 计数器服务，避免 rank 0 网络异常导致 MPI 挂死）
export AFSI_OFFLINE="${AFSI_OFFLINE:-1}"
# 动量预测步：gamma>0 时用直接法（grad-div 让迭代法极慢）；gamma=0 用原迭代法（LU 每步分解反而慢很多）
lu() { if [ "$1" = "0" ]; then echo "${MOMENTUM_LU_G0:-0}"; else echo 1; fi; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RES/run.log"; }

# ---------------------------------------------------------------------------
# 1) 环形固体 demo_423（串行即可，单个工况 < 1 min）
# ---------------------------------------------------------------------------
annulus() {
  cd "$DEMO/demo_423"
  # 1a 网格收敛：对偶铺展，gamma=0，IPCS 与 Chorin
  for S in ipcs chorin; do
    log "annulus convergence $S"
    IB_DIRECT_LOAD=1 GRAD_DIV=0 python convergence.py --solver $S -n 16 32 64 128 \
      --steps 100 --dt 1e-4 --json "$RES/annulus_conv_${S}_direct1_g0.json" \
      > "$RES/annulus_conv_${S}_direct1_g0.txt" 2>&1
  done
  # 1b 时间步研究：N=64，T=0.01 固定
  N=64 python generate_mesh.py > /dev/null
  for S in ipcs chorin; do
    for DT in 1e-4 2.5e-5 6.25e-6; do
      STEPS=$(python -c "print(round(0.01/$DT))")
      log "annulus dt $S dt=$DT"
      N=64 STEPS=$STEPS DT=$DT SOLVER=$S IB_DIRECT_LOAD=1 GRAD_DIV=0 python main.py \
        > "$RES/annulus_dt_${S}_${DT}.txt" 2>&1
    done
  done
  # 1c 铺展方式与 grad-div：N=64，IPCS
  for D in 1 0; do
    for G in 0 0.1 1 10; do
      log "annulus spread direct=$D gamma=$G"
      N=64 STEPS=100 DT=1e-4 SOLVER=ipcs IB_DIRECT_LOAD=$D GRAD_DIV=$G python main.py \
        > "$RES/annulus_N64_ipcs_direct${D}_g${G}.txt" 2>&1
    done
  done
}

# ---------------------------------------------------------------------------
# 2) 方腔驱动圆盘 demo_336（T=10，建议 MPI）
# ---------------------------------------------------------------------------
disk() {
  cd "$DEMO/demo_336"
  MESH="$RES/disk_mesh.xdmf"
  [ -f "$MESH" ] || OUT="$MESH" H=0.008 python generate_mesh.py
  run_disk() {  # $1 solver $2 direct $3 gamma
    local tag="disk_${1}_direct${2}_g${3}"
    log "disk $tag"
    MOMENTUM_LU=$(lu $3) MESH_PATH="$MESH" OUTPUT_PATH="$RES/$tag" SOLVER=$1 IB_DIRECT_LOAD=$2 GRAD_DIV=$3 \
      $MPI python fsi_paralell.py > "$RES/$tag.log" 2>&1
  }
  run_disk ipcs   1 0
  run_disk ipcs   1 "$GAMMA_DISK"
  run_disk chorin 1 0
  run_disk chorin 1 "$GAMMA_DISK"
  run_disk ipcs   0 0
}

# ---------------------------------------------------------------------------
# 3) 二维理想瓣膜 demo_340（T=3 s，48000 步；8 进程约 12 min/工况）
# ---------------------------------------------------------------------------
valve() {
  cd "$DEMO/demo_340"
  [ -f plot/mesh-340.xdmf ] || python generate_mesh.py
  run_valve() {  # $1 solver $2 direct $3 gamma $4 fiber angle
    local tag="valve_${1}_direct${2}_g${3}_f${4}"
    log "valve $tag"
    rm -f "$RES/$tag.csv"
    MOMENTUM_LU=$(lu $3) OUTPUT_PATH="$RES/$tag" HISTORY="$RES/$tag.csv" SOLVER=$1 IB_DIRECT_LOAD=$2 \
      GRAD_DIV=$3 FIBER_ANGLE=$4 $MPI python main.py > "$RES/$tag.log" 2>&1 || log "  FAILED $tag (see log)"
  }
  # 对偶 vs dof-to-dof（IPCS），以及 Chorin 基线
  [ "${SKIP_VALVE_IPCS1:-0}" = "1" ] || run_valve ipcs 1 "$GAMMA_VALVE" 45
  run_valve ipcs   0 "$GAMMA_VALVE" 45
  run_valve chorin 1 "$GAMMA_VALVE" 45
  run_valve chorin 0 "$GAMMA_VALVE" 45
}

# 圆盘 grad-div 加大系数扫描（面积守恒随 gamma 的变化）
disk_gamma() {
  cd "$DEMO/demo_336"
  MESH="$RES/disk_mesh.xdmf"
  for G in 10 100; do
    local tag="disk_ipcs_direct1_g${G}"
    log "disk $tag"
    MOMENTUM_LU=1 MESH_PATH="$MESH" OUTPUT_PATH="$RES/$tag" SOLVER=ipcs IB_DIRECT_LOAD=1 GRAD_DIV=$G \
      $MPI python fsi_paralell.py > "$RES/$tag.log" 2>&1 || log "  FAILED $tag"
  done
}

case "$WHAT" in
  annulus) annulus ;;
  disk) disk ;;
  disk_gamma) disk_gamma ;;
  valve) valve ;;
  all) annulus; disk; valve ;;
  *) echo "unknown group: $WHAT"; exit 1 ;;
esac
log "done: $WHAT"
