"""
Navier-Stokes 求解器抽象基类

三步投影法的公共接口:
  Step 1: 试探速度
  Step 2: 压力修正
  Step 3: 速度修正

子类需实现 _setup_forms() 定义各自变分形式。
"""

from abc import ABC, abstractmethod
from petsc4py import PETSc
from dolfinx.fem import Constant, Function
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector, apply_lifting,
                               create_vector, set_bc)


class BaseNSSolver(ABC):
    """三步投影法 NS 求解器抽象基类。

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        速度函数空间 (CG2 vector)。
    Q : dolfinx.fem.FunctionSpace
        压力函数空间 (CG1 scalar)。
    bcu : list of dolfinx.fem.DirichletBC
        速度边界条件。
    bcp : list of dolfinx.fem.DirichletBC
        压力边界条件。
    dt : float
        时间步长。
    rho : float
        密度。
    mu : float
        动力粘度。
    """

    def __init__(self, V, Q, bcu, bcp, dt, rho, mu):
        self.V = V
        self.Q = Q
        self.mesh = V.mesh
        self.bcu = bcu
        self.bcp = bcp

        # 物理参数
        self.dt = dt
        self.k = Constant(self.mesh, PETSc.ScalarType(dt))
        self.rho = Constant(self.mesh, PETSc.ScalarType(rho))
        self.mu = Constant(self.mesh, PETSc.ScalarType(mu))

        # 场变量 (子类可按需覆盖)
        self.u_ = Function(V, name="u")
        self.u_n = Function(V, name="u_n")
        self.p_ = Function(Q, name="p")
        self.p_n = Function(Q, name="p_n")
        self.f = Function(V, name="force")

        # 调用子类钩子设置变分形式
        self._setup_forms()

        # 组装矩阵和创建求解器
        self._assemble_matrices()
        self._create_solvers()

    @abstractmethod
    def _setup_forms(self):
        """子类实现：定义三步投影的变分形式。

        需设置:
          self.a1, self.L1  — Step 1: 试探速度
          self.a2, self.L2  — Step 2: 压力修正
          self.a3, self.L3  — Step 3: 速度修正
        """

    def _assemble_matrices(self):
        """组装各步的矩阵。"""
        self.A1 = assemble_matrix(self.a1, bcs=self.bcu)
        self.A1.assemble()
        self.b1 = create_vector(self.V)

        self.A2 = assemble_matrix(self.a2, bcs=self.bcp)
        self.A2.assemble()
        self.b2 = create_vector(self.Q)

        self.A3 = assemble_matrix(self.a3)
        self.A3.assemble()
        self.b3 = create_vector(self.V)

    @abstractmethod
    def _create_solvers(self):
        """子类实现：创建各步的 KSP 求解器。

        需设置:
          self.solver1, self.solver2, self.solver3
        """

    @abstractmethod
    def solve_one_step(self):
        """执行一个完整时间步（三步投影）。"""

    def cleanup(self):
        """释放 PETSc 资源。"""
        for attr in ("b1", "b2", "b3", "solver1", "solver2", "solver3"):
            if hasattr(self, attr):
                getattr(self, attr).destroy()
