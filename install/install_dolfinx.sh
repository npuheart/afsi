


mkdir -p ~/tmp &&cd ~/tmp
wget -nc https://github.com/gabime/spdlog/archive/refs/tags/v${SPDLOG_VERSION}.tar.gz
wget -nc https://github.com/kahip/kahip/archive/v${KAHIP_VERSION}.tar.gz
wget -nc https://github.com/doxygen/doxygen/archive/refs/tags/Release_${DOXYGEN_VERSION}.tar.gz
wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_${HDF5_VERSION}.tar.gz 
wget -nc https://github.com/ornladios/ADIOS2/archive/v${ADIOS2_VERSION}.tar.gz -O adios2-v${ADIOS2_VERSION}.tar.gz
git clone -b gmsh_${GMSH_VERSION} --single-branch --depth 1 https://gitlab.onelab.info/gmsh/gmsh.git
git clone --depth=1 -b v${PETSC_VERSION} https://gitlab.com/petsc/petsc.git ${PETSC_DIR}
git clone --depth=1 -b v${SLEPC_VERSION} https://gitlab.com/slepc/slepc.git ${SLEPC_DIR}
git clone https://github.com/FEniCS/dolfinx.git
git clone https://github.com/FEniCS/basix.git
git clone https://github.com/FEniCS/ufl.git
git clone https://github.com/FEniCS/ffcx.git

cd ~/tmp
tar xfz v${SPDLOG_VERSION}.tar.gz && \
    cd spdlog-${SPDLOG_VERSION} && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DSPDLOG_BUILD_SHARED=ON -DSPDLOG_BUILD_PIC=ON -DCMAKE_INSTALL_PREFIX=$HOME/software -B build-dir . && \
    cmake --build build-dir && \
    cmake --install build-dir
    rm -rf build-dir

cd ~/tmp
tar xfz Release_${DOXYGEN_VERSION}.tar.gz && \
    cd doxygen-Release_${DOXYGEN_VERSION} && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/software -B build-dir . && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    rm -rf build-dir

cd ~/tmp
python3 -m venv ${VIRTUAL_ENV}
pip install --no-cache-dir --upgrade pip setuptools wheel && \
pip install --no-cache-dir cython numpy==${NUMPY_VERSION} && \
pip install --no-cache-dir mpi4py

cd ~/tmp
tar -xf v${KAHIP_VERSION}.tar.gz
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DNONATIVEOPTIMIZATIONS=on -DCMAKE_INSTALL_PREFIX=$HOME/software -B build-dir -S KaHIP-${KAHIP_VERSION}
cmake --build build-dir
cmake --install build-dir
rm -r build-dir

tar xfz hdf5_${HDF5_VERSION}.tar.gz
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$HOME/software -DCMAKE_BUILD_TYPE=Release -DHDF5_ENABLE_PARALLEL=on -DHDF5_ENABLE_Z_LIB_SUPPORT=on -B build-dir -S hdf5-hdf5_${HDF5_VERSION}
cmake --build build-dir -j${BUILD_NP}
cmake --install build-dir
rm -r build-dir

cd ~/tmp
mkdir -p adios2-v${ADIOS2_VERSION} && tar -xf adios2-v${ADIOS2_VERSION}.tar.gz -C adios2-v${ADIOS2_VERSION} --strip-components 1 
cmake -G Ninja -DADIOS2_USE_HDF5=on -DCMAKE_INSTALL_PREFIX=$HOME/software -DCMAKE_INSTALL_PYTHONDIR=$HOME/software/lib/ -DADIOS2_USE_Fortran=off -DBUILD_TESTING=off -DADIOS2_BUILD_EXAMPLES=off -DADIOS2_USE_ZeroMQ=off -B build-dir-adios -S ./adios2-v${ADIOS2_VERSION}
cmake --build build-dir-adios -j${BUILD_NP} && cmake --install build-dir-adios
rm -r build-dir-adios

cd ~/tmp
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_BUILD_DYNAMIC=1 -DCMAKE_INSTALL_PREFIX=$HOME/software -DCMAKE_INSTALL_PYTHONDIR=$HOME/software/lib/ -DENABLE_OPENMP=1 -B build-dir -S gmsh
cmake --build build-dir -j32 && cmake --install build-dir
rm -r build-dir

cd ${PETSC_DIR}
./configure \
    PETSC_ARCH=linux-gnu-real64-32 \
    --COPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --CXXOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --FOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --with-64-bit-indices=no \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-hypre \
    --download-metis \
    --download-mumps-avoid-mpi-in-place \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-spai \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-scalar-type=real \
    --with-precision=double

make PETSC_ARCH=linux-gnu-real64-32 ${MAKEFLAGS} all -j${BUILD_NP}
make PETSC_DIR=/home/pengfei/software/petsc PETSC_ARCH=linux-gnu-real64-32 check 
cd src/binding/petsc4py   &&  pip -v install --no-cache-dir --no-build-isolation . 

cd ${SLEPC_DIR}
export PETSC_ARCH=linux-gnu-real64-32 && ./configure && make -j${BUILD_NP}
cd src/binding/slepc4py && pip -v install --no-cache-dir --no-build-isolation . 

cd ~/tmp
pip install --no-cache-dir -r dolfinx/python/build-requirements.txt
pip install --no-cache-dir pyamg pytest scipy matplotlib numba

cd ~/tmp/basix && cmake -G Ninja -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=$HOME/software -B build-dir -S ./cpp
cmake --build build-dir -j${BUILD_NP} && cmake --install build-dir

pip install ./python && \
    cd ../ufl && pip install --no-cache-dir . && \
    cd ../ffcx && pip install --no-cache-dir . && \
    cd ../ && pip install --no-cache-dir ipython

cd ~/tmp/dolfinx && mkdir -p build-real && cd build-real
PETSC_ARCH=linux-gnu-real64-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$HOME/software/dolfinx-real -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -DDOLFINX_ENABLE_PETSC=ON -DDOLFINX_ENABLE_SLEPC=ON -DDOLFINX_ENABLE_SCOTCH=ON -DDOLFINX_ENABLE_KAHIP=ON -DDOLFINX_ENABLE_ADIOS2=ON ../cpp 
ninja install -j ${BUILD_NP} && cd ../python
PETSC_ARCH=linux-gnu-real64-32 pip -v install --config-settings=cmake.build-type="${DOLFINX_CMAKE_BUILD_TYPE}" --config-settings=install.strip=false --no-build-isolation --check-build-dependencies --target $HOME/software/dolfinx-real/lib/python${PYTHON_VERSION}/dist-packages --no-dependencies --no-cache-dir '.'
