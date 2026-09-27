# AFSI 安装指南（Conda 环境）

本文档介绍如何使用 Conda 创建 `afsi-dolfinx` 环境，安装 FEniCSx 及 AFSI 项目。

---

## 环境要求

- Conda（Miniconda 或 Anaconda）
- 操作系统（均已实测验证）：
  - Linux x86_64（Ubuntu 24.04）
  - macOS arm64（Apple Silicon，需安装 Xcode Command Line Tools）
- macOS 提示：若装有 Homebrew，构建时需忽略 `/opt/homebrew` 前缀（见第 4 步），否则
  CMake 会优先找到 Homebrew 的串行 HDF5 等库，与 conda 环境冲突导致编译失败。

---

## 1. 创建 Conda 环境

```bash
# 建议显式指定 conda-forge，避免与 defaults 频道混用
conda create -n afsi-dolfinx -c conda-forge python=3.12 -y
```

> 若不加 `-c conda-forge`，第 2 步安装 `fenics-dolfinx` 时 Python 会被 conda-forge
> 版本替换（实测 3.12.14 → 3.12.13），虽能继续但不推荐。

## 2. 安装 FEniCSx

```bash
conda install -n afsi-dolfinx -c conda-forge fenics-dolfinx=0.10.0 --solver libmamba -y
```

## 3. 安装其他依赖

```bash
# 科学计算和可视化（python-gmsh 提供 import gmsh 的 Python 模块；
# 只装 gmsh 只有命令行程序，实测会报 ModuleNotFoundError: No module named 'gmsh'）
conda install -n afsi-dolfinx -c conda-forge \
    matplotlib scipy numba gmsh python-gmsh --solver libmamba -y

# 编译工具链（Linux）
conda install -n afsi-dolfinx -c conda-forge \
    cmake ninja gcc_linux-64 gxx_linux-64 make libstdcxx-ng \
    libcurl tbb-devel openssl --solver libmamba -y

# 编译工具链（macOS；gcc_linux-64/gxx_linux-64/libstdcxx-ng 在 macOS 上不存在，
# 使用 Xcode Command Line Tools 的 Apple clang 即可）
conda install -n afsi-dolfinx -c conda-forge \
    cmake ninja make libcurl tbb-devel openssl --solver libmamba -y

# 测试和工具
conda install -n afsi-dolfinx -c conda-forge \
    pytest tqdm --solver libmamba -y

# swanlab（pip，不在 conda-forge 中）
conda run -n afsi-dolfinx pip install swanlab
```

> **说明**：`petsc4py`、`mpi4py`、`numpy` 等已作为 `fenics-dolfinx` 的依赖自动安装。

## 4. 安装 AFSI 项目

```bash
# 激活环境
conda activate afsi-dolfinx

# 安装构建依赖
pip install nanobind "scikit-build-core[pyproject]"

# 编译安装 AFSI（开发模式）—— Linux
cd /path/to/afsic
pip install --no-build-isolation -ve .

# 编译安装 AFSI（开发模式）—— macOS（需附加以下参数，均为实测必需）
cd /path/to/afsic
SITE=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX;$SITE \
  -DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew \
  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_AR=/usr/bin/ar -DCMAKE_RANLIB=/usr/bin/ranlib -DCMAKE_LINKER=/usr/bin/ld" \
  pip install -C install.strip=false --no-build-isolation -ve .
```

> **macOS 参数说明**：
> - `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX;$SITE`：同时暴露 conda 依赖与 site-packages 中的
>   nanobind（scikit-build-core 会生成覆盖该变量的 CMake 初始化缓存，不传则 nanobind 找不到）。
> - `-DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew`：跳过 Homebrew 前缀，否则会命中 Homebrew
>   的串行 HDF5，报 `CMake Error: Found serial HDF5 build, MPI HDF5 build required`。
> - `-DCMAKE_C_COMPILER/-DCMAKE_CXX_COMPILER/-DCMAKE_AR/-DCMAKE_RANLIB/-DCMAKE_LINKER`：
>   统一为 Apple 工具链。conda 环境通常只有 `clang` 没有 `clang++`，CMake 会混用 conda 的
>   `ar/ranlib/ld`，导致链接报 `ld: archive member '/' not a mach-o file`。
> - `-C install.strip=false`：跳过 strip。否则 `install_name_tool` 修改 rpath 后代码签名
>   失效，`import afsic` 会被系统直接杀掉（`zsh: killed`）。若已出现，可手动修复：
>   `codesign --force -s - <site-packages>/afsic/afsic_ext.abi3.so`

> **注意**：`afsic` 包导入时（`src/afsic/common/utilities.py`）依赖 `swanlab`，
> 请确保第 3 步的 swanlab 已安装，否则报 `ModuleNotFoundError: No module named 'swanlab'`。

> **注意**：如果编译时遇到 `MPI::MPI_C not found` 错误，确保 `CMakeLists.txt` 中已声明 `C` 语言（`project(afsic LANGUAGES C CXX)`）并添加 `find_package(MPI REQUIRED COMPONENTS C CXX)`。

## 5. 验证安装

```bash
# 激活环境
conda activate afsi-dolfinx

# 测试导入
python -c "import afsic; print('afsic imported successfully')"

# 运行测试
cd /path/to/afsic
python tests/test_coupling_2D.py
python tests/test_coupling_3D.py
```

预期输出示例（Linux / macOS 实测一致）：

```
# 2D 测试
Result: 0.500000
Result: 0.500000

# 3D 测试
Result: 0.340000
Result: 0.670000
Result: 0.560000
```

> `Result:` 行由 C++ 扩展打印，测试脚本本身无其他输出。

---

## 已知兼容性修复

安装过程中对源码做了以下适配（经检查，本仓库当前代码已包含这两处修复，无需重复操作）：

### CMakeLists.txt

```cmake
# 原：
project(afsic LANGUAGES CXX)

# 改为：
project(afsic LANGUAGES C CXX)

# 在 add_subdirectory 之前添加：
find_package(MPI REQUIRED COMPONENTS C CXX)
```

### IPCSSolver.py（实际路径 `src/afsic/euler/IPCSSolver.py`）

```python
# 原：
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)

# 改为（dolfinx 0.10.0 中 gmshio 重命名为 gmsh）：
from dolfinx.io import (VTXWriter, distribute_entity_data, gmsh as gmshio)
```

---

## 环境信息参考

| 组件 | Linux x86_64（Ubuntu 24.04） | macOS arm64（Apple Silicon） |
|------|------|------|
| Python | 3.12.13 | 3.12.13 |
| fenics-dolfinx | 0.10.0 | 0.10.0 |
| PETSc | 3.24.4 | 3.25.5 |
| MPI | MPICH 4.3.2 | OpenMPI 5.0.11 |
| nanobind | 2.13.0 | 3.1.0 |
| scikit-build-core | 1.0.3 | 1.1.0 |
| CMake | 4.4.1 | 4.4.x |
| 编译器 | GCC 14.3.0 | Apple clang 17 |

> macOS 上 conda-forge 的 `fenics-dolfinx` 依赖 OpenMPI（而非 MPICH），PETSc 等版本
> 随 conda-forge 当前解析结果可能略有差异；nanobind / scikit-build-core 直接用 pip 安装最新版即可。
