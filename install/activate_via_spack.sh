# source ~/spack/share/spack/setup-env.sh
# spack load gcc@14
# spack env activate fenicsx-adios-env -p
# spack load cmake 


export SPACK_VERSION=e2c49f2b3a7aabc324a2d54f6c8319b854191bd4
export SPACK_DIR="$HOME/spack-${SPACK_VERSION}-$(hostname -s)"
export SPACK_USER_CONFIG_PATH="$HOME/.spack-$SPACK_VERSION-$(hostname -s)"
export SPACK_USER_CACHE_PATH="$HOME/.spack-$SPACK_VERSION-$(hostname -s)/cache"
export ENV_NAME="fepos"
export TMPDIR="$HOME/tmp"

source $SPACK_DIR/share/spack/setup-env.sh
spack bootstrap root "$HOME/.spack-$SPACK_VERSION-$(hostname -s)/bootstrap"
spack load gcc@14
spack env activate $ENV_NAME -p
spack load cmake 

