// Turek–Hron 改版基准的固体几何 —— CGS 单位 (g, cm, s)
// 论文(SI, m): 圆心(0.2,0.2) 直径 0.1；梁 0.35 × 0.02 —— 换算 ×100 得 cm。
SetFactory("OpenCASCADE");

cx = 20;      // 圆柱圆心 x [cm]   (0.2  m)
cy = 20;      // 圆柱圆心 y [cm]   (0.2  m)
R = 5;        // 圆柱半径    [cm]   (d = 0.1 m = 10 cm)
flag_L = 35;  // 弹性梁长    [cm]   (0.35 m)
flag_h = 2;   // 弹性梁厚    [cm]   (0.02 m)
eps = 1e-4;   // 布尔运算容差 [cm]  (= 1e-6 m)

// ---------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------
// Ball: the cylinder, kept as its own surface (NOT unioned away).
Disk(1) = {cx, cy, 0, R};

// Tail (raw): rectangle starting at the cylinder CENTER so it fully
// swallows the right half of the disk -> after cutting away the disk
// part below, the tail's left edge is guaranteed to hug the circle
// exactly (no sliver gap), instead of just touching it at one point.
Rectangle(2) = {cx, cy - flag_h/2, 0, R + flag_L, flag_h};

// Tail = rectangle MINUS the disk (keep a copy of the disk so
// Surface{1} below still exists afterwards).
Duplicata { Surface{1}; }   // -> creates Surface{3}, a copy of the disk
BooleanDifference(4) = { Surface{2}; Delete; }{ Surface{3}; Delete; };

// Glue ball (1) and tail (4) so they share the interface curve
// conformally (watertight, no overlap, single shared edge for meshing).
BooleanFragments{ Surface{1}; Surface{4}; Delete; }{ }

// ---------------------------------------------------------------
// Helper: x-coordinate where the flag band intersects the circle
// (R, cy +/- flag_h/2) -> chord_x = cx + sqrt(R^2 - (flag_h/2)^2)
// ---------------------------------------------------------------
chord_x = cx + Sqrt(R^2 - (flag_h/2)^2);

// ---------------------------------------------------------------
// Physical Surfaces: ball vs tail, selected by bounding box
// ---------------------------------------------------------------
ball_surf[]  = Surface In BoundingBox{ cx-R-eps,        cy-R-eps,        -eps,
                                       cx+R+eps,        cy+R+eps,        eps };
tail_surf[]  = Surface In BoundingBox{ chord_x-eps,     cy-flag_h/2-eps, -eps,
                                       cx+R+flag_L+eps, cy+flag_h/2+eps, eps };

Physical Surface("ball") = ball_surf[];
Physical Surface("tail") = tail_surf[];

// ---------------------------------------------------------------
// Physical Curves: circle / upper edge / lower edge / right tip
// ---------------------------------------------------------------
// circle: the long arc, excluded x-range up to chord_x so the two
// tiny notch arcs (the ball-tail interface) are NOT picked up.
circle_c[]    = Curve In BoundingBox{ cx-R-eps, cy-R-eps, -eps,
                                       chord_x+eps, cy+R+eps, eps };

upper_c[]     = Curve In BoundingBox{ chord_x-eps,      cy+flag_h/2-eps, -eps,
                                       cx+R+flag_L+eps, cy+flag_h/2+eps, eps };

lower_c[]     = Curve In BoundingBox{ chord_x-eps,      cy-flag_h/2-eps, -eps,
                                       cx+R+flag_L+eps, cy-flag_h/2+eps, eps };

right_tip_c[] = Curve In BoundingBox{ cx+R+flag_L-eps, cy-flag_h/2-eps, -eps,
                                       cx+R+flag_L+eps, cy+flag_h/2+eps, eps };

Physical Curve("circle")    = circle_c[];
Physical Curve("upper_edge") = upper_c[];
Physical Curve("lower_edge") = lower_c[];
Physical Curve("right_tip")  = right_tip_c[];

// ---------------------------------------------------------------
// Mesh size [cm]: MFAC=0.5 × 流体细网格 Δx=1.9219 cm -> 0.96 cm
//   （论文 Δx = L/N = 2.46/128 m = 1.9219 cm；MFAC = 固体网格/Δx）
// ---------------------------------------------------------------
Mesh.MeshSizeMax = 0.96;
Mesh.MeshSizeMin = 0.48;
