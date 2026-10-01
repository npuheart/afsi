# Solid formulation excerpts — rigid penalty force, volumetric penalization, modified invariants

> 论文摘录（原文照录，公式为 LaTeX/KaTeX 兼容写法；文献引用编号按原文保留），供 demo_402 的系绳/罚参数实现参考。

## Immersed rigid structure: discretized Lagrangian penalty force

For an immersed rigid structure, the discretized Lagrangian penalty force $\mathbf{F}_h(\mathbf{X}, t)$ is computed directly from the discrete deformation approximation $\boldsymbol{\chi}_h(\mathbf{X}, t)$:

$$
\mathbf{F}_h(\mathbf{X}, t) = \kappa \left( \boldsymbol{\psi}(\mathbf{X}, t) - \boldsymbol{\chi}_h(\mathbf{X}, t) \right) + \eta \left( \mathbf{V}(\mathbf{X}, t) - \frac{\partial \boldsymbol{\chi}_h}{\partial t}(\mathbf{X}, t) \right).
$$

Because $\mathbf{F}_h$ and $\boldsymbol{\chi}_h$ share the same finite element basis, the nodal coefficients of $\mathbf{F}_h$ are determined directly by those of $\boldsymbol{\chi}_h$:

$$
\mathbf{F}_l(t) = \kappa \left( \boldsymbol{\psi}_l(t) - \boldsymbol{\chi}_l(t) \right) + \eta \left( \mathbf{V}_l(t) - \frac{\partial \boldsymbol{\chi}_l}{\partial t}(t) \right).
$$

Throughout the rest of the paper, we drop the subscript $h$ and assume each Lagrangian variable is represented by its discrete approximation to prevent the notation from becoming overly cumbersome.

## 3.3.1 Volumetric Penalization

For flexible structures, traditional IB formulations decompose the Cauchy stress as

$$
\sigma = \sigma^v - p\mathbb{I} + \begin{cases} 0 & \mathbf{x} \in \Omega_t^f, \\ \sigma^e & \mathbf{x} \in \Omega_t^s, \end{cases}
$$

where $\sigma^v = \mu \left(\nabla \mathbf{u} + \nabla \mathbf{u}^\top \right)$ is the deviatoric viscous stress and $\sigma^e$ is the elastic stress.

This formulation can lead to poor numerical results in the discretized equations, including unphysical and sometimes extreme contractions of the immersed structure. To address this issue, the elastic stress tensor is decomposed into deviatoric and volumetric components:

$$
\sigma^e = \operatorname{dev}[\sigma^e] - \pi_{\text{stab}} \mathbb{I}
$$

where $\pi_{\text{stab}} \mathbb{I}$ acts as a stabilization term, providing an additional pressure in the solid domain that counteracts spurious compressible motions.

Following approaches used in nearly incompressible elasticity, the volumetric penalization term $\pi_{\text{stab}}$ can be derived from volumetric energy $U(J)$ that depends only on volumetric changes in the structure:

$$
\pi_{\text{stab}} = -\frac{\partial U(J)}{\partial J}
$$

Note that the negative sign is due to the pressure convention in the fluid mechanics.

To control the stabilization strength, a numerical Poisson ratio $\nu_{\text{stab}}$ is introduced to modulate the numerical bulk modulus $\kappa_{\text{stab}}$ through the relationship:

$$
\kappa_{\text{stab}} = \frac{2G(1+\nu_{\text{stab}})}{3(1-2\nu_{\text{stab}})}
$$

in which $G$ is the shear modulus. This relationship mirrors the connection between the physical Poisson ratio $\nu$ and bulk modulus $\kappa$ in compressible materials. Setting $\nu_{\text{stab}} = -1$ yields $\pi_{\text{stab}} = 0$, which recovers the unstabilized formulation. It is important to note that both $\kappa_{\text{stab}}$ and $\nu_{\text{stab}}$ are numerical parameters rather than physical ones, as the immersed structure remains incompressible in all cases.

## 3.3.2 Modified Invariants

The elastic Cauchy stress is related to the first Piola-Kirchhoff stress through

$$
\sigma^{\mathrm{e}} = \frac{1}{J} \mathbb{P}^{\mathrm{s}} \mathbb{F}^{\top}.
$$

For hyperelastic materials, the first Piola-Kirchhoff stress $\mathbb{P}^{\mathrm{s}}$ can be derived from a strain energy functional $\Psi(\mathbb{F})$:

$$
\mathbb{P}^{\mathrm{s}} = \frac{\partial \Psi(\mathbb{F})}{\partial \mathbb{F}}.
$$

To achieve the desired decomposition of the Cauchy stress, the strain energy functional into volume-preserving and volume-changing components:

$$
\Psi(\mathbb{F}) = W(\mathbb{F}) + U(J).
$$

To ensure material frame invariance,$^{32}$ $\Psi$ for isotropic materials is typically expressed in terms of the first two invariants of the right Cauchy-Green tensor $\mathbb{C} = \mathbb{F}^{\top}\mathbb{F}$:

$$
\Psi(I_1, I_2) = W(I_1, I_2) + U(J)
$$

with $I_1 = \operatorname{tr}(\mathbb{C})$ and $I_2 = \frac{1}{2}\left(I_1^2 - \operatorname{tr}(\mathbb{C}^2)\right)$. This is referred to as the unmodified invariants-based model.

For nearly incompressible elasticity, it is common to use invariants that only encode shearing deformations but not volume change. To eliminate volume change information, the modified invariants are introduced:$^{48}$

$$
\tilde{I}_1 = J^{-2/3} I_1
$$

$$
\tilde{I}_2 = J^{-4/3} I_2
$$

These are the invariants of the modified tensor $\tilde{\mathbb{C}} = \tilde{\mathbb{F}}^{\top} \tilde{\mathbb{F}}$. The resulting model takes the form:

$$
\Psi(\tilde{I}_1, \tilde{I}_2) = W(\tilde{I}_1, \tilde{I}_2) + U(J)
$$

This is referred to as the modified invariants-based model.
