import os

from petsc4py import PETSc

from dolfinx.fem import (Constant, Function, form, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix)
from ufl import (FacetNormal, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs)

# 实验开关：NO_CONVECTION=1 关闭对流项；IPCS_VISC_BE=1 粘性改用隐式欧拉（默认 CN）
_NO_CONVECTION = os.environ.get("NO_CONVECTION", "0").lower() not in ("0", "", "false", "no")
_VISC_BE = os.environ.get("IPCS_VISC_BE", "0").lower() not in ("0", "", "false", "no")
# KSP_MONITOR=1: 每步检查三个线性求解器的收敛原因/迭代数，失败立即打印
# KSP_STRICT=1 : 不收敛时直接报错（PETSc 异常）而不是静默继续
_KSP_MONITOR = os.environ.get("KSP_MONITOR", "0").lower() not in ("0", "", "false", "no")
_KSP_STRICT = os.environ.get("KSP_STRICT", "0").lower() not in ("0", "", "false", "no")
_KSP_STATE = {"step": 0, "fails": {}, "maxit": {}}
# IPCS_KSP_CHORIN=1: 线性求解策略对齐 ChorinSolver（动量 HYPRE、压力 BCGS+HYPRE）
_KSP_CHORIN = os.environ.get("IPCS_KSP_CHORIN", "0").lower() not in ("0", "", "false", "no")
# IPCS_NO_LAG=1: 消融实验——预测步去掉滞后压力项 -div(p_)（其余不变）
_NO_LAG = os.environ.get("IPCS_NO_LAG", "0").lower() not in ("0", "", "false", "no")

class IPCSSolver:
    def __init__(self, V, Q, bcu, bcp, dt_raw, rho_raw, mu_raw,
                 ds_p=None, p_traction=None, drag=None, ib_body_force=True):
        """Incremental pressure-correction solver.

        Parameters
        ----------
        ib_body_force : bool, default True
            True  —— 动量弱式包含 -∫f·v 项（与 ChorinSolver 符号一致；调用方直接
                     存放物理力 b1 的原值即可，无需符号补偿）。
            False —— 省略该弱式项，改由调用方通过 ``self.ib_load`` 每步提供一个
                     "已装配、已含符号"的 IB 载荷 PETSc 向量（直接载荷模式，
                     对应 main.py 的 IB_DIRECT_LOAD；两种方式不得同时使用）。        ds_p : ufl.Measure, optional
            Measure on the pressure-Dirichlet open-boundary facets.  When
            supplied, ``p_traction`` is added explicitly to the momentum
            predictor so the remaining natural condition on those facets is
            the correct homogeneous viscous traction ``mu du/dn = 0``.
        p_traction : dolfinx.fem.Function or Constant, optional
            Known pressure datum ``p_bc(t)`` on the ``ds_p`` facets.  The
            caller must update its value every time step.
        """
        self.bcu = bcu
        self.bcp = bcp
        self.p_traction = p_traction
        
        self.V = V
        self.Q = Q
        
        mesh = V.mesh
        self.mesh = mesh
        k = Constant(mesh, PETSc.ScalarType(dt_raw))
        mu = Constant(mesh, PETSc.ScalarType(mu_raw)) 
        rho = Constant(mesh, PETSc.ScalarType(rho_raw))

        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)
        u_ = Function(V,name = "u")
        u_s = Function(V)
        u_n = Function(V)
        u_n1 = Function(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)
        p_ = Function(Q,name = "p")
        phi = Function(Q)


        f = Function(V)
        F1 = rho / k * dot(u - u_n, v) * dx
        if not _NO_CONVECTION:
            F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
        if _VISC_BE:
            F1 += mu * inner(grad(u), grad(v)) * dx
        else:
            F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx
        if not _NO_LAG:
            F1 -= dot(p_, div(v)) * dx
        if ib_body_force:
            # 符号与 ChorinSolver 一致（-∫f·v）：调用方直接存放物理力 b1 的原值，
            # 无需任何 force_scale 类补偿（demo_402/423/424/425/336 已统一）。
            F1 -= dot(f, v) * dx
        if drag is not None:
            # Implicit linear damping: adds drag * u to the momentum LHS.
            # Used to make a fluid region behave like a porous/static medium.
            F1 += dot(drag * u, v) * dx
        if ds_p is not None:
            if p_traction is None:
                raise ValueError("p_traction must be given when ds_p is provided")
            n = FacetNormal(mesh)
            F1 += dot(p_traction * n, v) * ds_p
        a1 = form(lhs(F1))
        L1 = form(rhs(F1))
        A1 = create_matrix(a1)
        b1 = create_vector(V)
        # Pressure update
        a2 = form(dot(grad(p), grad(q)) * dx)
        L2 = form(-rho / k * dot(div(u_s), q) * dx)
        A2 = assemble_matrix(a2, bcs=self.bcp)
        A2.assemble()
        b2 = create_vector(Q)
        # Velocity update
        a3 = form(rho * dot(u, v) * dx)
        L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
        A3 = assemble_matrix(a3, bcs=self.bcu)
        A3.assemble()
        b3 = create_vector(V)

        # Solver for step 1
        solver1 = PETSc.KSP().create(mesh.comm)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        # Solver for step 2
        solver2 = PETSc.KSP().create(mesh.comm)
        solver2.setOperators(A2)
        solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver for step 3
        solver3 = PETSc.KSP().create(mesh.comm)
        solver3.setOperators(A3)
        solver3.setType(PETSc.KSP.Type.CG)
        pc3 = solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

        if _KSP_CHORIN:
            # 对齐 ChorinSolver 的线性求解策略：动量 HYPRE、压力 BCGS+HYPRE
            pc1.setType(PETSc.PC.Type.HYPRE)
            pc1.setHYPREType("boomeramg")
            solver2.setType(PETSc.KSP.Type.BCGS)
            pc2.setType(PETSc.PC.Type.HYPRE)
            pc2.setHYPREType("boomeramg")

        if _KSP_STRICT:
            for _ksp in (solver1, solver2, solver3):
                _ksp.setErrorIfNotConverged(True)
        
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.u_ = u_
        self.u_s = u_s
        self.u_n = u_n
        self.u_n1 = u_n1
        self.p_ = p_
        self.phi = phi
        self.a1 = a1
        self.L1 = L1
        self.A1 = A1
        self.b1 = b1
        self.a2 = a2
        self.L2 = L2
        self.A2 = A2
        self.b2 = b2
        self.a3 = a3
        self.L3 = L3
        self.A3 = A3
        self.b3 = b3
        self.f = f
        # 直接载荷接口（默认 None）：由调用方每步提供一个已装配、已含符号的 IB 载荷
        # PETSc 向量（与 b1 同布局）。在 lifting 之后、set_bc 之前加到动量右端 owned
        # 自由度（每个全局自由度只加一次；f 经全局 gather/scatter 已完整，此处不得
        # 再做反向 ghost 累加）。需配合 ib_body_force=False 使用，否则重复计入。
        self.ib_load = None

    def solve_one_step(self):
        # Step 1: Tentative velocity step
        self.A1.zeroEntries()
        assemble_matrix(self.A1, self.a1, bcs=self.bcu)
        self.A1.assemble()
        with self.b1.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        if self.ib_load is not None:
            self.b1.axpy(1.0, self.ib_load)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_s.x.petsc_vec)
        self.u_s.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with self.b2.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.phi.x.petsc_vec)
        self.phi.x.scatter_forward()

        if _NO_LAG:
            # 非增量模式：预测步无压力，phi 即本步全压，直接赋值
            # （不能累加，否则输出压力无物理意义）
            self.p_.x.array[:] = self.phi.x.array[:]
        else:
            self.p_.x.petsc_vec.axpy(1, self.phi.x.petsc_vec)
        self.p_.x.scatter_forward()

        # Step 3: Velocity correction step
        # Applied to the full velocity field, so bcu must be imposed again;
        # otherwise the pressure-correction gradient destroys wall no-slip.
        with self.b3.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b3, self.L3)
        apply_lifting(self.b3, [self.a3], [self.bcu])
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b3, self.bcu)
        self.solver3.solve(self.b3, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

        # Update variable with solution form this time step
        with self.u_.x.petsc_vec.localForm() as loc_, self.u_n.x.petsc_vec.localForm() as loc_n, self.u_n1.x.petsc_vec.localForm() as loc_n1:
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)

        if _KSP_MONITOR:
            _KSP_STATE["step"] += 1
            rank = self.mesh.comm.rank
            for tag, ksp in (("momentum", self.solver1), ("pressure", self.solver2),
                             ("velocity", self.solver3)):
                reason = ksp.getConvergedReason()
                it = ksp.getIterationNumber()
                _KSP_STATE["maxit"][tag] = max(_KSP_STATE["maxit"].get(tag, 0), it)
                if reason < 0:
                    n = _KSP_STATE["fails"].get(tag, 0) + 1
                    _KSP_STATE["fails"][tag] = n
                    if rank == 0 and n <= 3:
                        print(f"[KSP-FAIL] step={_KSP_STATE['step']} {tag}: reason={reason} "
                              f"iters={it} res={ksp.getResidualNorm():.3e}", flush=True)
            if rank == 0 and _KSP_STATE["step"] % 500 == 0:
                print(f"[KSP] step={_KSP_STATE['step']} maxit={_KSP_STATE['maxit']} "
                      f"fails={_KSP_STATE['fails']}", flush=True)