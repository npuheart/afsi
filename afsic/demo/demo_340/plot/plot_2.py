import pandas as pd

import matplotlib.pyplot as plt

# 读取数据
x_df = pd.read_csv('data/ani_x.csv')
y_df = pd.read_csv('data/ani_y.csv')
t_df = pd.read_csv('data/ani_t.csv')

# 获取时间步
t = t_df['demo-340-000092-time_step']

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(t, x_df['demo-340-000092-x_displacement_step'], 'g:', label='$45^\circ$', markersize=5, linewidth=2.0)
plt.plot(t, x_df['demo-340-000091-x_displacement_step'], 'b-.', label='$60^\circ$', markersize=5, linewidth=2.0)
plt.plot(t, x_df['demo-340-000090-x_displacement_step'], 'k--', label='$75^\circ$', markersize=5, linewidth=2.0)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (cm)')
plt.legend(loc='lower right')
plt.grid(True)
plt.xlim([0, 3.0])
plt.ylim([0, 0.6])
plt.tight_layout()
plt.savefig('x_vs_t.png')
plt.close()

# 画 y 关于 t 的变化
plt.figure(figsize=(8, 5))
plt.plot(t, y_df['demo-340-000092-y_displacement_step'], 'g:', label='$45^\circ$', markersize=5, linewidth=2.0)
plt.plot(t, y_df['demo-340-000091-y_displacement_step'], 'b-.', label='$60^\circ$', markersize=5, linewidth=2.0)
plt.plot(t, y_df['demo-340-000090-y_displacement_step'], 'k--', label='$75^\circ$', markersize=5, linewidth=2.0)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (cm)')
plt.legend(loc='lower right')
plt.grid(True)
plt.xlim([0, 3.0])
plt.ylim([0, 0.5])
plt.tight_layout()
plt.savefig('y_vs_t.png')
