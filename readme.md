
# AFSI - Automated Fluid-Structure Interaction Solver


<p align="center">
    <img src="https://githubimages.pengfeima.cn/images/202508220026326.jpg"  alt="valve"/>
</p>
<p align="center">
    <img src="https://githubimages.pengfeima.cn/images/202508220028530.png" width="300" alt="valve"/>
    <img src="https://githubimages.pengfeima.cn/images/202508220029045.png" width="300" alt="ventricle"/>
</p>
<p align="center">
  To the up we show an ideal valve, and to the down, the ideal left ventricle benchmark.
</p>

Description
-----------
AFSI is a automated fluid-structure interaction solver based on the immersed boundary method and developed within the FEniCS framework. Leveraging FEniCS’s support for automated solution of partial differential equations, AFSI offers out-of-the-box high-performance capabilities, making it a practical tool for research groups and individual researchers to investigate complex FSI problems—including those involving nonlinear solids with large deformations and large displacements.


Authors
-------
AFSI is developed by:

- Pengfei Ma
- Xuan Wang


Licence
-------
GPLv3


Documentation
-------------
Work in progress......

Some demos are under the directory: `afsic/demo`.

Installation
------------
Currently, AFSI requires FEniCSx version 0.10.0. The following steps outline one method for installing FEniCSx:

- Install the necessary dependencies using `install/install_env.sh` with sudo.
- Activate the environment by sourcing `install/activate_dolfinx`.
- Install FEniCSx by running `install/install_dolfinx`.

After installing FEniCSx, install AFSI by navigating to the `afsic` directory and running:
```
cd afsic && pip install .
```
The installation process has been tested and verified on a fresh installation of Ubuntu 24.04.
Before each use of AFSI, remember to activate the environment by sourcing `install/activate_dolfinx`.

Use
---

Work in progress......

Contact
-------
The latest version of this software can be obtained from

  https://github.com/npuheart/afsi

Please report bugs and other issues through the issue tracker at:

  https://github.com/npuheart/afsi/issues
