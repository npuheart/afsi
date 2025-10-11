# NOTE: Still waiting the correct 64x64x64 data 

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
    'font.size': 16,            
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
})

# 读取 CSV 文件
df = pd.read_csv('32x32x32.csv')


plt.figure(figsize=(8, 6))
plt.plot(df['num_processors'], df['num_processors'], marker='o', linestyle='--')

# 32x32x32的网格
plt.plot(df['num_processors'], df['speed_up'], marker='^', markerfacecolor='none', color='tab:blue', label='$32^3$ AFSI 求解器')
plt.plot(df['num_processors'], df['speed_up_2'], marker='^', markerfacecolor='none', color='tab:orange', label='$32^3$ Navier-Stokes 求解器')

df = pd.read_csv('64x64x64.csv')
# 64x64x64的网格
plt.plot(df['num_processors'], df['speed_up'], marker='x', markerfacecolor='none', color='tab:green', label='$64^3$ AFSI 求解器')
plt.plot(df['num_processors'], df['speed_up_2'], marker='x', markerfacecolor='none', color='tab:red', label='$64^3$ Navier-Stokes 求解器')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('MPI 进程数')
plt.ylabel('加速比')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('your_plot.png', dpi=600)
plt.show()