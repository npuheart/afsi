# AFSI 安装指南（Conda 环境）

本文档介绍如何使用 Conda 创建 `afsi-dolfinx` 环境，安装 FEniCSx 及 AFSI 项目。

---

## 环境要求

- Conda（Miniconda 或 Anaconda）
- Linux x86_64（已在 Ubuntu 24.04 上验证）

---

## 1. 创建 Conda 环境

```bash
conda create -n afsi-dolfinx python=3.12 -y
```

## 2. 安装 FEniCSx

```bash
conda install -n afsi-dolfinx -c conda-forge fenics-dolfinx=0.10.0 --solver libmamba -y
```

## 3. 安装其他依赖

```bash
# 科学计算和可视化
conda install -n afsi-dolfinx -c conda-forge \
    matplotlib scipy numba gmsh --solver libmamba -y

# 编译工具链
conda install -n afsi-dolfinx -c conda-forge \
    cmake ninja gcc_linux-64 gxx_linux-64 make libstdcxx-ng \
    libcurl tbb-devel --solver libmamba -y

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

# 编译安装 AFSI（开发模式）
cd /path/to/afsic
pip install --no-build-isolation -ve .
```

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

预期输出示例：

```
# 2D 测试
order : 2
mesh size : 65, 61
cell size : 0.015625, 0.016667
Result: 0.500000
Result: 0.500000

# 3D 测试
order : 2
mesh size : 65, 63, 61
cell size : 0.015625, 0.016129, 0.016667
evaluate at (0.340000, 0.670000, 0.560000)
Result: 0.340000
Result: 0.670000
Result: 0.560000
```

---

## 已知兼容性修复

安装过程中对源码做了以下适配（已包含在仓库中则无需重复操作）：

### CMakeLists.txt

```cmake
# 原：
project(afsic LANGUAGES CXX)

# 改为：
project(afsic LANGUAGES C CXX)

# 在 add_subdirectory 之前添加：
find_package(MPI REQUIRED COMPONENTS C CXX)
```

### IPCSSolver.py

```python
# 原：
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)

# 改为（dolfinx 0.10.0 中 gmshio 重命名为 gmsh）：
from dolfinx.io import (VTXWriter, distribute_entity_data, gmsh as gmshio)
```

---

## 环境信息参考

| 组件 | 版本 |
|------|------|
| Python | 3.12.13 |
| fenics-dolfinx | 0.10.0 |
| PETSc | 3.24.4 |
| MPICH | 4.3.2 |
| nanobind | 2.13.0 |
| scikit-build-core | 1.0.3 |
| CMake | 4.4.1 |
| GCC | 14.3.0 |
