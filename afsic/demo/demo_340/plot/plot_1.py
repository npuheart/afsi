
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np

# 分别读取两个文件
# AFSI 曲线数据来自新生成 data/ani_*.csv（45° = demo-340-000092），5.17_*.csv 已删除可再生成
df_ALE = pd.read_csv('x_dis_ALE.csv')
df_FSI = pd.read_csv('X_FSI.csv')
df_ma_t = pd.read_csv('data/ani_t.csv')
df_ma_x = pd.read_csv('data/ani_x.csv')
df_M2 = pd.read_csv('X_M2.csv')

t_ma = df_ma_t['demo-340-000092-time_step'].values
x_displacement_ma = df_ma_x['demo-340-000092-x_displacement_step'].values

# 提取数据列
t_ALE = df_ALE['x'].values
x_displacement_ALE = df_ALE['Curve1'].values

t_M2 = df_M2['x'].values
x_displacement_M2 = df_M2['Curve1'].values

t_FSI = df_FSI['x'].values
x_displacement_FSI = df_FSI['Curve1'].values

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(t_M2, x_displacement_M2, 'g:', label='Ryan et al.(M2)', markersize=5, linewidth=2.0)
plt.plot(t_FSI, x_displacement_FSI, 'b-.', label='Ryan et al.(M3)', markersize=5, linewidth=2.0)
plt.plot(t_ALE, x_displacement_ALE, 'k--', label='Kamensky et al.', markersize=5, linewidth=2.0)
plt.plot(t_ma, x_displacement_ma, 'r-', label='AFSI', markersize=5, linewidth=2.0)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (cm)')
plt.legend(loc='lower right')
plt.grid(True)
plt.xlim([0, 3.0])
plt.ylim([0, 0.6])
plt.tight_layout()
plt.savefig('smoothed_x.png', dpi=300)  # 保存图像
plt.show()

# 读取 CSV 文件
df_ALE_Y = pd.read_csv('Y_ALE.csv')  # 替换为你的文件路径
df_FSI_Y = pd.read_csv('Y_FSI.csv')  # 替换为你的文件路径
df_M2_Y = pd.read_csv('y_M2.csv')

df_ma_t = pd.read_csv('data/ani_t.csv')
df_ma_y = pd.read_csv('data/ani_y.csv')

t_ma = df_ma_t['demo-340-000092-time_step'].values
y_displacement_ma = df_ma_y['demo-340-000092-y_displacement_step'].values

t_ALE_Y = df_ALE_Y['x'].values
y_displacement_ALE = df_ALE_Y['Curve1'].values

t_FSI_Y = df_FSI_Y['x'].values
y_displacement_FSI = df_FSI_Y['Curve1'].values

t_M2_Y = df_M2_Y['x'].values
y_displacement_M2 = df_M2_Y['Curve1'].values

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(t_M2_Y, y_displacement_M2, 'g:', label='Ryan et al.(M2)', markersize=5, linewidth=2.0)
plt.plot(t_FSI_Y, y_displacement_FSI, 'b-.', label='Ryan et al.(M3)', markersize=5, linewidth=2.0)
plt.plot(t_ALE_Y, y_displacement_ALE, 'k--', label='Kamensky et al.', markersize=5, linewidth=2.0)
plt.plot(t_ma, y_displacement_ma, 'r-', label='AFSI', markersize=5, linewidth=2.0)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (cm)')
plt.legend(loc='lower right')
plt.grid(True)
plt.xlim([0, 3.0])
plt.ylim([0, 0.5])
plt.tight_layout()
plt.savefig('smoothed_y.png', dpi=300)  # 保存图像
plt.show()