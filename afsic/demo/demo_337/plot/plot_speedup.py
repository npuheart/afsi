import pandas as pd

import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('demo-337-2025-8-31_14_15_57.csv')

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(df['num_processors'], df['speed_up'], marker='o', label='Speed up')
plt.plot(df['num_processors'], df['speed_up_2'], marker='o', label='Speed up 2')
plt.plot(df['num_processors'], df['num_processors'], marker='o', linestyle='--', label='Ideal')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('num_processors')
plt.ylabel('Speed up')
plt.title('Speed up vs Number of Processors')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('your_plot.png')
plt.show()