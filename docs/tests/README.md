# Tests — Overview

This directory documents every test file in `afsic/tests/` to aid future
understanding and review.

---

## `tests/test_email.py`

**Purpose:** Integration test for the `afsic.send_email` / `afsic.EmailInfo`
nanobind bindings (backed by libcurl SMTP).

**What it tests:**

| Test | Description |
|---|---|
| `test_send_plain_email` | Sends a plain-text email to `Pengfei.Ma@glasgow.ac.uk` and asserts `send_email` returns `True`. |
| `test_send_html_email` | Sends an HTML email to the same address and asserts success. |

**Credentials:** Read from environment variables — tests are automatically
**skipped** if any variable is missing (no hard-coded secrets):

| Variable | Example value |
|---|---|
| `AFSIC_SMTP_URL` | `smtp://smtp.qq.com:587` |
| `AFSIC_SMTP_USERNAME` | `499908174@qq.com` |
| `AFSIC_SMTP_PASSWORD` | QQ SMTP authorisation code |
| `AFSIC_SMTP_FROM` | `499908174@qq.com` |

**How to run:**

```bash
source ~/.bashrc          # loads the AFSIC_SMTP_* variables
cd /home/staff4/pma/afsi/afsic
pytest tests/test_email.py -v
```

---

*Add a new section here whenever a new test file is created.*

---

## `tests/test_ns.py`

**Purpose:** Unit and convergence tests for the incompressible Navier-Stokes
solvers `ChorinSolver` and `IPCSSolver`.

**Benchmark:** 2-D Poiseuille (channel) flow on the unit square.
Exact steady-state solution:

```
u_x(y) = 4 * U_max * y * (1 - y),  u_y = 0,  p(x) = -8 * mu * x
```

Inlet: parabolic profile at x=0. No-slip: y=0 and y=1. Outlet: p=0 at x=1.

| Test | Checks |
|---|---|
| `test_chorin_construct` | `ChorinSolver` constructs without error; `u_` and `p_` exist. |
| `test_ipcs_construct` | Same for `IPCSSolver`. |
| `test_mass_conservation[ChorinSolver]` | `‖div u‖₂ < 1e-8` after 10 steps. |
| `test_mass_conservation[IPCSSolver]` | Same for IPCS. |
| `test_convergence[ChorinSolver]` | L² velocity error vs. Poiseuille exact `< 1e-2` after 400 steps (t=2). |
| `test_convergence[IPCSSolver]` | Same for IPCS. |

**How to run:**

```bash
cd /home/staff4/pma/afsi/afsic
pytest tests/test_ns.py -v
# or in parallel:
mpirun -n 4 pytest tests/test_ns.py -v
```
