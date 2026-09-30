---
title: ""
description: ""
date: 2026-09-30
weight: 2
academic: true
---

## 1. Introduction

This demo is called from {{< cite "li2025local" "author" >}} ( **4.3.1 Slanted Channel Flow** ) and the most original from {{< cite "gruninger2024benchmarking" "author" >}} (**4.1. Oldroyd-B flow through an inclined channel**), but that is for viscoelastic flow.

We have extended the type of boundary condition combinations: the original one is the **velocity Dirichlet** driving — the analytical steady solution, Eq. (48), is imposed on the inlet and outlet cross-sections (ramped from rest) — for which the pressure field is kept unique with a **pin point** (one pressure degree of freedom fixed to zero inside the inlet opening); what we added is the **pressure Dirichlet** driving — the analytical opening pressure $p=-(\Delta p/L)\,(x\cos\theta+y\sin\theta)$ is imposed on both cross-sections and the velocity there is left free. In the incremental pressure-correction solver this requires the opening pressure-traction term to be added explicitly to the predictor (`ds_p`/`p_traction`, as in demo_424). The two slanted plates are held stationary with the tethering (penalty-spring) method, stiffness $\beta=8\times10^3$; all other boundary segments are no-slip.

Errors at $t=2$ ($N=32$, $\Delta t=0.15\Delta x$, 427 steps), measured inside the channel against the analytical solution:

| driving | relative $L_2$ error | profile error at $x=0.5$ | $U_{\max}$ error |
| --- | --- | --- | --- |
| velocity Dirichlet (+ pin point) | $0.46\%$ | $0.54\%$ | $+0.29\%$ |
| pressure Dirichlet | $11.9\%$ | $7.5\%$ | $+6.8\%$ |

The gap is intrinsic to the oblique ($30^\circ$) openings: the predictor's remaining natural condition on the cut (zero viscous traction) is not satisfied by the exact solution, whereas the velocity datum imposed on the same cut is exact by construction. Both variants are stable, with residual wall slip below $2.5\%$ of $U_{\max}$ and the plates staying within $0.7$ cell of their reference position. 

![image-20260930182645394](https://githubimages.pengfeima.cn/images/202609301826534.png)

![image-20260930182550326](https://githubimages.pengfeima.cn/images/202609301825491.png)

![**FIGURE 1**: The mesh of solid and background domain.](https://githubimages.pengfeima.cn/images/202609301818180.png)

{{< references >}}


## Apendix 1: Digest from {{< cite "li2025local" "author" >}} 

{{< color "red" >}}It should be noted that parameter D it the veritical space between two walls.{{< /color >}}

To evaluate the performance of different kernels in non-grid-aligned configurations, we examine flow through a slanted channel, following the two-dimensional benchmark presented in Gruninger et al.$^{46}$ The computational domain $\Omega=[0,1]\times[-0.25,2]$ contains two parallel plates separated by channel width $D.$ Figure 22 illustrates the channel geometry, inclined at angle $\theta=\pi/6$, and shows a representative velocity field computed using CBS$_32.$

![image-20260930175010567](https://githubimages.pengfeima.cn/images/202609301750002.png)

**Figure 22**: Computational setup of the slanted channel flow problem (inclination angle $\pi/6$). The color map shows the velocity field computed using CBS$_{32}$ kernel. The white vertical line at $x = 0.5$ indicates the location of velocity profile measurements, and the dots represent Lagrangian markers defining the top and bottom channel plates.

The channel is inclined rather than vertical or horizontal to avoid grid-aligned discretization of the channel walls. The exact steady-state solution for flow through the inclined channel can be derived through the coordinate transformation of the plane Poiseuille equation. As in previous work, {{< cite "gruninger2024benchmarking" "author" >}}  the analytic solution is given by:



$$
u(x,y)=-\frac{\Delta p\cos\theta}{2\mu L}\left(y\cos\theta-x\sin\theta\right)\left(y\cos\theta-x\sin\theta-h\right),\\\nu(x,y)=-\frac{\Delta p\sin\theta}{2\mu L}\left(y\cos\theta-x\sin\theta\right)\left(y\cos\theta-x\sin\theta-h\right),\tag{48}
$$

where $h$ denotes the height of the channel in its horizontal configuration and $\theta$ the counterclockwise rotation angle about the origin.

The simulation uses the following parameters: the horizontal channel width $h=1$, and the slanted channel width $D=h/\cos(\theta)$, dynamic viscosity $\mu=0.5$, density $\rho=1.0$, pressure gradient $\Delta p/L=1.0$, and maximum velocity $U_{\max}=0.25.$ The fine-grid Cartesian cell size is set to $\Delta x=L/N$,where $N=32$ with a time step size of $\Delta t=0.15\Delta x.$

We impose velocity boundary conditions at the inlet and outlet using the analytical steady-state solution from Eq. (48). To maintain the rigidity and stationary position of the structure, we employ both the penalty stiffness and the penalty body and damping forces described in Eq. (14). The penalty parameters are empirically determined as the largest values that maintain numerical stability for each combination of kernel, grid spacing, and time step size
This benchmark evaluates explicitly how different kernel choices affect the accuracy of flow computations within a confined, stationary geometry. Fig. 23 compares the velocity profiles at $x=0.5$ for different kernel types. Kernels with narrower support regions show better accuracy in capturing peak velocities, with BS and CBS kernels of equal support performing similarly and outperforming their IB counterparts. The width of the numerical boundary layer decreases with decreasing kernel support size, with CBS$_21$ demonstrating the thinnest numerical boundary layer among all tested

kernels.



![image-20260930175337520](https://githubimages.pengfeima.cn/images/202609301753653.png)

![image-20260930175352546](https://githubimages.pengfeima.cn/images/202609301753602.png)

**Figure 23**: Comparison of velocity profiles at x = 0.5 for different kernel types. Kernels with narrower support regions show better accuracy in capturing peak velocities, with BS and CBS kernels of equal support performing similarly and outperforming their IB counterparts. The width of the numerical boundary layer decreases with decreasing kernel support size, with CBS21 demonstrating the thinnest numerical boundary layer among all tested kernels