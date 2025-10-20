import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd

# 读取为 DataFrame
df = pd.read_csv('../data/diastole-ibamr.csv')
initial = np.loadtxt("../data/ideal_middle_wall.txt")
displacements = np.loadtxt("../data/diastole-pulse-disp.txt")
displacements_afsi = np.loadtxt("../data/diastole-afsi.txt")
deformed = initial + displacements
deformed_afsi = displacements_afsi

for i in range(len(deformed_afsi)):
    deformed_afsi[i][0] = (deformed_afsi[i][0] - 3.5)*10.0
    deformed_afsi[i][1] = (deformed_afsi[i][1] - 2.5)*10.0

# 获取某一列
column_data_1 = df['x']
column_data_2= df['Curve1']

plt.subplots_adjust(right=0.50)  
main_ax = plt.subplot(1, 1, 1)
main_ax.plot(initial[:, 1],initial[:, 0], linewidth=2, linestyle='--', color='gray', label='Initial')
main_ax.plot(column_data_1, column_data_2,  alpha=1.0, label='IBAMR')
main_ax.plot(deformed[:, 1],deformed[:, 0], alpha=1.0, label='Pulse')
main_ax.plot(deformed_afsi[:, 1],deformed_afsi[:, 0], alpha=1.0, label='AFSI')


main_ax.set_xlim(-15, 4)
main_ax.set_ylim(-30, 4)
main_ax.set_xticks([-10, -5, 0])
main_ax.set_xlabel('x(mm)')
main_ax.set_ylabel('z(mm)')
main_ax.grid(True, linestyle='-.', alpha=0.5)
main_ax.legend(loc='upper right', fontsize=10)
main_ax.set_aspect('equal', adjustable='box')  

rect = Rectangle(
    (-13.5, -9), 1, 7,
    linewidth=1, edgecolor='black', facecolor='none', linestyle='--',
)
main_ax.add_patch(rect)

rect = Rectangle(
    (-5, -28), 5, 3,
    linewidth=1, edgecolor='black', facecolor='none', linestyle='--'
)
main_ax.add_patch(rect)

main_ax.text(
    -0.1, 1.05, '(a)', 
    transform=main_ax.transAxes,  
    fontsize=12, 
    fontweight='bold',
    va='top', ha='right'
)

sub_ur = plt.axes([0.65, 0.60, 0.3, 0.3])
sub_ur.plot(column_data_1, column_data_2,  alpha=1.0, linestyle='None', marker='+', markersize=5, markeredgewidth=0.8, label='IBAMR')
sub_ur.plot(deformed[:, 1],deformed[:, 0], alpha=1.0, linestyle='None', marker='x', markersize=5, markeredgewidth=0.8, label='Pulse')
sub_ur.plot(deformed_afsi[:, 1],deformed_afsi[:, 0], alpha=1.0, label='Present')

sub_ur.set_xticks([-13, -12.5])
# sub_ur.set_yticks([-2, -1, 0, 1, 2])
sub_ur.set_xlim(-13.5, -12.5)
sub_ur.set_ylim(-9, -2)
sub_ur.set_xlabel('x(mm)')
sub_ur.set_ylabel('z(mm)')

sub_ur.text(
    -0.25, 1.10, '(b)', 
    transform=sub_ur.transAxes,  
    fontsize=12, 
    fontweight='bold',
    va='top', ha='right'
)


sub_ax = plt.axes([0.65, 0.15, 0.3, 0.3])  
sub_ax.plot(column_data_1, column_data_2,  alpha=1.0, linestyle='None', marker='+', markersize=5, markeredgewidth=0.8, label='IBAMR')
sub_ax.plot(deformed[:, 1],deformed[:, 0], alpha=1.0, linestyle='None', marker='x', markersize=5, markeredgewidth=0.8, label='Pulse')
sub_ax.plot(deformed_afsi[:, 1],deformed_afsi[:, 0], alpha=1.0, label='Present')
sub_ax.set_xticks([-5, -2.5, 0])  
sub_ax.set_yticks([-28, -27.5, -27, -26.5, -26, -25.5, -25])  # 简化y轴刻度
sub_ax.set_xlim(-5, 0)
sub_ax.set_ylim(-28.0, -25.0)
sub_ax.set_xlabel('x(mm)')
sub_ax.set_ylabel('z(mm)')

sub_ax.text(
    -0.25, 1.10, '(c)', 
    transform=sub_ax.transAxes,  
    fontsize=12, 
    fontweight='bold',
    va='top', ha='right'
)

plt.savefig("../figures/diastole_plot.png", dpi=300, bbox_inches='tight')  