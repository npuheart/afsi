#!/usr/bin/env bash
set -e  # 出错即停止

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] === $* ==="; }
START=$(date +%s)

export SPACK_VERSION=e2c49f2b3a7aabc324a2d54f6c8319b854191bd4
export SPACK_DIR="$HOME/spack-${SPACK_VERSION}-$(hostname -s)"
export SPACK_USER_CONFIG_PATH="$HOME/.spack-$SPACK_VERSION-$(hostname -s)"
export SPACK_USER_CACHE_PATH="$HOME/.spack-$SPACK_VERSION-$(hostname -s)/cache"
export ENV_NAME="fepos"
export TMPDIR="$HOME/tmp"


log "Clone Spack"
git clone --filter=blob:none --no-checkout https://github.com/spack/spack.git $SPACK_DIR
cd $SPACK_DIR
git checkout $SPACK_VERSION
git switch -c $SPACK_VERSION

log "Load Spack environment"
source $SPACK_DIR/share/spack/setup-env.sh
spack bootstrap root "$HOME/.spack-$SPACK_VERSION-$(hostname -s)/bootstrap"

log "Install gcc@14 (needed for fenics-dolfinx@main)"
spack install gcc@14
spack load gcc@14
spack compiler find

log "Create and activate Spack env"
spack env create $ENV_NAME
spack env activate $ENV_NAME -p

log "Add FEniCSx with PETSc/SLEPc (using gcc@14)"
spack add py-fenics-dolfinx@main+petsc4py+slepc4py ^fenics-dolfinx+adios2 ^petsc+hypre+mumps %gcc@14
spack add py-pip

log "Concretize environment"
spack concretize -f

log "Install packages"
spack install

log "Install adios4dolfinx via pip"
python3 -m pip install adios4dolfinx[test] gmsh matplotlib
python3 -m pip install pytest
ELAPSED=$(( $(date +%s) - START ))
log "Done — total time: $(( ELAPSED/3600 ))h $(( ELAPSED%3600/60 ))m $(( ELAPSED%60 ))s"
