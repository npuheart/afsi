import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 英文文章使用 Arial 字体，中文文章使用 Microsoft YaHei 字体
# properties = fm.FontProperties(fname=f'{home_dir}/afsi-data/msyh.ttc')
# print(properties.get_name())  # 输出字体名称以确认加载成功
# plt.rcParams['font.sans-serif'] = ['Arial']

home_dir = os.path.expanduser('~')
fm.fontManager.addfont(f'{home_dir}/afsi-data/Arial.ttf')
fm.fontManager.addfont(f'{home_dir}/afsi-data/msyh.ttc')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']



plt.rcParams.update({
    'font.size': 12,            
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
})


# 读取 CSV 文件
df = pd.read_csv('32x32x32.csv')

# 绘制图形
plt.figure(figsize=(8, 6))
# plt.plot(df['num_processors'], [data*100 for data in df['timing/solid_force_p']], marker='^', markerfacecolor='none', label='Structural force evaluation')  # 空心三角形
# plt.plot(df['num_processors'], [data*100 for data in df['timing/fluid_to_solid_p']], marker='x', label='Velocity interpolation')  # 叉
# plt.plot(df['num_processors'], [data*100 for data in df['timing/solid_to_fluid_p']], marker='o', markerfacecolor='none', label='Force spreading')  # 空心圆
# plt.plot(df['num_processors'], [data*100 for data in df['timing/ns_solver_p']], marker='s', markerfacecolor='none', label='Navier-Stokes solver')  # 空心方块
plt.plot(df['num_processors'], [data*100 for data in df['timing/solid_force_p']], marker='^', markerfacecolor='none', label='固体力')  # 空心三角形
plt.plot(df['num_processors'], [data*100 for data in df['timing/fluid_to_solid_p']], marker='x', label='速度插值')  # 叉
plt.plot(df['num_processors'], [data*100 for data in df['timing/solid_to_fluid_p']], marker='o', markerfacecolor='none', label='力延拓')  # 空心圆
plt.plot(df['num_processors'], [data*100 for data in df['timing/ns_solver_p']], marker='s', markerfacecolor='none', label='Navier-Stokes 求解')  # 空心方块

for col in ['timing/fluid_to_solid_p']:
    for x, y in zip(df['num_processors'][4:], df[col][4:]):
        plt.text(x, (y - 0.05)*100, f'{y*100:.0f}%', ha='center', va='bottom')


for col in ['timing/solid_force_p']:
    for x, y in zip(df['num_processors'][4:], df[col][4:]):
        plt.text(x, (y + 0.02 * max(df[col]))*100, f'{y*100:.0f}%', ha='center', va='bottom')


for col in ['timing/ns_solver_p']:
    for x, y in zip(df['num_processors'], df[col]):
        plt.text(x, (y - 0.07 * max(df[col]))*100, f'{y*100:.0f}%', ha='center', va='bottom')


for col in ['timing/solid_to_fluid_p']:
    for x, y in zip(df['num_processors'], df[col]):
        plt.text(x, (y + 0.04 * max(df[col]))*100, f'{y*100:.0f}%', ha='center', va='bottom')


# plt.xlabel('Number of MPI processes')
plt.xlabel('MPI 进程数')
plt.ylabel('耗时占比 (%)')
plt.xlim(-5, max(df['num_processors']) + 5)
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('your_plot_2.png', dpi=600)