import pandas as pd

import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('demo-337-2025-8-31_14_15_57.csv')

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(df['num_processors'], [data*100 for data in df['timing/solid_force_p']], marker='^', markerfacecolor='none', label='Structural force evaluation')  # 空心三角形
plt.plot(df['num_processors'], [data*100 for data in df['timing/fluid_to_solid_p']], marker='x', label='Velocity interpolation')  # 叉
plt.plot(df['num_processors'], [data*100 for data in df['timing/solid_to_fluid_p']], marker='o', markerfacecolor='none', label='Force spreading')  # 空心圆
plt.plot(df['num_processors'], [data*100 for data in df['timing/ns_solver_p']], marker='s', markerfacecolor='none', label='Navier-Stokes solver')  # 空心方块

for col in ['timing/fluid_to_solid_p']:
    for x, y in zip(df['num_processors'][4:], df[col][4:]):
        plt.text(x, (y - 0.05)*100, f'{y*100:.0f}%', ha='center', va='bottom', fontsize=12)


for col in ['timing/solid_force_p']:
    for x, y in zip(df['num_processors'][4:], df[col][4:]):
        plt.text(x, (y + 0.02 * max(df[col]))*100, f'{y*100:.0f}%', ha='center', va='bottom', fontsize=12)


for col in ['timing/ns_solver_p']:
    for x, y in zip(df['num_processors'], df[col]):
        plt.text(x, (y - 0.07 * max(df[col]))*100, f'{y*100:.0f}%', ha='center', va='bottom', fontsize=12)


for col in ['timing/solid_to_fluid_p']:
    for x, y in zip(df['num_processors'], df[col]):
        plt.text(x, (y + 0.04 * max(df[col]))*100, f'{y*100:.0f}%', ha='center', va='bottom', fontsize=12)


plt.xlabel('Number of MPI processes')
plt.ylabel('Time consumption for each component (%)')
plt.xlim(-5, max(df['num_processors']) + 5)
plt.ylim(0, 100)
# plt.title('Speed up vs Number of Processors')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('your_plot_2.png', dpi=500)