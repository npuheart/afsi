// 球体连接三段直杆的几何模型
// 单位：cm

// 定义几何参数
R_sphere = 5.0      // 球体半径
R_rod = 1.0         // 直杆半径
L_rod1 = 10.0       // 第一段直杆长度
L_rod2 = 8.0        // 第二段直杆长度
L_rod3 = 12.0       // 第三段直杆长度

// 定义材料编号
m_void = 0          // 真空
m_sphere = 1        // 球体材料
m_rod1 = 2          // 第一段直杆材料
m_rod2 = 3          // 第二段直杆材料
m_rod3 = 4          // 第三段直杆材料

// 定义cell编号
c_void = 1          // 外部真空
c_sphere = 10       // 球体
c_rod1 = 20         // 第一段直杆
c_rod2 = 30         // 第二段直杆
c_rod3 = 40         // 第三段直杆

// 定义曲面
// 球体表面
surf_sph = SQ 0 0 0 R_sphere

// 第一段直杆（沿X轴正方向）
surf_cyl1_x = C/X 0 0 R_rod
surf_plane1a = PX 0
surf_plane1b = PX L_rod1

// 第二段直杆（沿Y轴正方向）
surf_cyl2_y = C/Y L_rod1 0 R_rod
surf_plane2a = PY 0
surf_plane2b = PY L_rod2

// 第三段直杆（沿Z轴正方向）
surf_cyl3_z = C/Z L_rod1 L_rod2 R_rod
surf_plane3a = PZ 0
surf_plane3b = PZ L_rod3

// 世界边界
surf_world = SO 100

// 定义cells
// 外部真空
cell c_void m_void 
    -surf_world 
    #(-surf_sph) 
    #(surf_cyl1_x -surf_plane1a surf_plane1b) 
    #(surf_cyl2_y -surf_plane2a surf_plane2b) 
    #(surf_cyl3_z -surf_plane3a surf_plane3b)

// 球体
cell c_sphere m_sphere 
    -surf_sph 
    #(surf_cyl1_x -surf_plane1a) 
    #(surf_cyl2_y -surf_plane2a) 
    #(surf_cyl3_z -surf_plane3a)

// 第一段直杆
cell c_rod1 m_rod1 
    -surf_cyl1_x surf_plane1a -surf_plane1b 
    #(-surf_sph)

// 第二段直杆
cell c_rod2 m_rod2 
    -surf_cyl2_y surf_plane2a -surf_plane2b

// 第三段直杆
cell c_rod3 m_rod3 
    -surf_cyl3_z surf_plane3a -surf_plane3b

// 定义facet标记（用于可视化或特殊处理）
// 球体表面标记
facet f_sph_outer 
    surf_sph
    
// 直杆1表面标记
facet f_rod1_lateral 
    surf_cyl1_x
    
facet f_rod1_base 
    surf_plane1a
    
facet f_rod1_top 
    surf_plane1b
    
// 直杆2表面标记
facet f_rod2_lateral 
    surf_cyl2_y
    
facet f_rod2_base 
    surf_plane2a
    
facet f_rod2_top 
    surf_plane2b
    
// 直杆3表面标记
facet f_rod3_lateral 
    surf_cyl3_z
    
facet f_rod3_base 
    surf_plane3a
    
facet f_rod3_top 
    surf_plane3b

// 材料定义（示例）
// 球体材料
material m_sphere 
    composition H 2 O 1
    density 1.0
    
// 直杆材料
material m_rod1 
    composition Fe 1
    density 7.8
    
material m_rod2 
    composition Al 1
    density 2.7
    
material m_rod3 
    composition Cu 1
    density 8.9

// 真空材料
material m_void 
    composition void
    density 0.0