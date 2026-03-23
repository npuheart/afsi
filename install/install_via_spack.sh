#!/usr/bin/env bash
set -e  # 出错即停止

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] === $* ==="; }
START=$(date +%s)

log "Clean up old Spack installation"
rm -rf ~/.spack ~/spack

log "Clone Spack"
git clone https://github.com/spack/spack.git ~/spack

log "Load Spack environment"
source ~/spack/share/spack/setup-env.sh

log "Install gcc@14 (needed for fenics-dolfinx@main)"
spack install gcc@14
spack load gcc@14
spack compiler find

log "Create and activate Spack env"
spack env create fenicsx-env
spack env activate fenicsx-env -p

log "Add FEniCSx with PETSc/SLEPc (using gcc@14)"
spack add py-fenics-dolfinx@main+petsc4py+slepc4py+adios %gcc@14

log "Add pip"
spack add py-pip

log "Concretize environment"
spack concretize -f

log "Install packages"
spack install

log "Install adios4dolfinx via pip"
python3 -m pip install adios4dolfinx[test]

ELAPSED=$(( $(date +%s) - START ))
log "Done — total time: $(( ELAPSED/3600 ))h $(( ELAPSED%3600/60 ))m $(( ELAPSED%60 ))s"
