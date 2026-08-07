SetFactory("OpenCASCADE");

// Channel with a cylindrical hole — body-fitted mesh
// Fluid domain = rectangle minus cylinder disk.
// Physical Curve tags: 11=inlet, 12=outlet, 13=bottom, 14=top, 15=cylinder
// Physical Surface: "fluid" (tag 1)
//
// 修正说明:
//   1) 原来用 Duplicata 保留的圆盘曲面会同时被 gmsh 剖分，
//      导致 .msh 中圆盘与流体单元重叠，dolfinx read_from_msh 报
//      "Invalid rank ... less than 1"。这里 BooleanDifference 直接删除
//      圆盘输入，只保留带孔的流体曲面。
//   2) 显式指定物理标签 11-15，与 main.py 的 find(11)~find(15) 及
//      readme 一致（此前 gmsh 自动分配 1-6，边界条件全部失效）。

Lx = 2.2;
Ly = 0.41;
cx = 0.2;
cy = 0.2;
R  = 0.05;
eps = 1e-6;

Rectangle(1) = {0, 0, 0, Lx, Ly};
Disk(2) = {cx, cy, 0, R};

// 矩形减去圆盘，输入曲面均删除 → 只留下带孔流体曲面
BooleanDifference(4) = { Surface{1}; Delete; }{ Surface{2}; Delete; };

Physical Surface(1) = {4};   // "fluid"

// Boundary curves
inlet_c[]   = Curve In BoundingBox{ -eps, -eps, -eps,  eps,   Ly+eps, eps };
outlet_c[]  = Curve In BoundingBox{ Lx-eps, -eps, -eps, Lx+eps, Ly+eps, eps };
top_c[]     = Curve In BoundingBox{ -eps, Ly-eps, -eps, Lx+eps, Ly+eps, eps };
bottom_c[]  = Curve In BoundingBox{ -eps, -eps, -eps, Lx+eps, eps,    eps };
cylinder_c[]= Curve In BoundingBox{ cx-R-eps, cy-R-eps, -eps, cx+R+eps, cy+R+eps, eps };

Physical Curve(11) = inlet_c[];     // "inlet"
Physical Curve(12) = outlet_c[];    // "outlet"
Physical Curve(13) = bottom_c[];    // "bottom"
Physical Curve(14) = top_c[];       // "top"
Physical Curve(15) = cylinder_c[];  // "cylinder"

Mesh.MeshSizeMax = 0.01;   // far field ~220×41, 与均匀网格 h≈0.01 一致
Mesh.MeshSizeMin = 0.002;  // finer near cylinder
