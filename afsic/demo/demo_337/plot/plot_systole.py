import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

data_ibamr = np.loadtxt('../data/systole-ibamr.txt', skiprows=1)
initial = np.loadtxt("../data/ideal_middle_wall.txt")
displacements = np.loadtxt("../data/systole-pulse-disp.txt")
displacements_afsi = np.loadtxt("../data/systole-afsi-4.txt")
deformed = initial + displacements
deformed_afsi = displacements_afsi

for i in range(len(deformed_afsi)):
    deformed_afsi[i][0] = (deformed_afsi[i][0] - 3.5)*10.0
    deformed_afsi[i][1] = (deformed_afsi[i][1] - 2.5)*10.0

plt.subplots_adjust(right=0.50)  
main_ax = plt.subplot(1, 1, 1)
# main_ax.plot(x_np, curve1_np, color='red', alpha=1.0, linestyle='None', marker='+', markersize=5, markeredgewidth=0.8, label='IBAMR')
# main_ax.plot(data_ibamr[:, 0],data_ibamr[:, 1], color='red', alpha=1.0, label='IBAMR')
# main_ax.plot(deformed[:, 1],deformed[:, 0], color='blue', alpha=1.0, label='Present')
# main_ax.plot(initial[:, 1],initial[:, 0], linewidth=2, linestyle='--', color='gray', label='Initial')
# plt.plot(initial[:, 1], initial[:, 0], label='Initial Position')
main_ax.plot(initial[:, 1],initial[:, 0], linewidth=2, linestyle='--', color='gray', label='Initial')
main_ax.plot(data_ibamr[:, 0],data_ibamr[:, 1],  alpha=1.0, label='IBAMR')
main_ax.plot(deformed[:, 1],deformed[:, 0], alpha=1.0, label='Pulse')
main_ax.plot(deformed_afsi[:, 1],deformed_afsi[:, 0], alpha=1.0, label='AFSI')

main_ax.set_xlim(-15, 4)
main_ax.set_ylim(-30, 4)
main_ax.set_xticks([-8, -6, -4, -2, 0])
main_ax.set_xlabel('x(mm)')
main_ax.set_ylabel('z(mm)')
main_ax.grid(True, linestyle='-.', alpha=0.5)
main_ax.legend(loc='upper right', fontsize=10)
main_ax.set_aspect('equal', adjustable='box')  



rect = Rectangle(
    (-8.75, -2), 0.5, 4,
    linewidth=1, edgecolor='black', facecolor='none', linestyle='--',
)
main_ax.add_patch(rect)

rect = Rectangle(
    (-2, -15), 2, 2,
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
sub_ur.plot(data_ibamr[:, 0],data_ibamr[:, 1],  alpha=1.0, linestyle='None', marker='+', markersize=5, markeredgewidth=0.8, label='IBAMR')
sub_ur.plot(deformed[:, 1],deformed[:, 0], alpha=1.0, linestyle='None', marker='x', markersize=5, markeredgewidth=0.8, label='Pulse')
sub_ur.plot(deformed_afsi[:, 1],deformed_afsi[:, 0], alpha=1.0, label='Present')

sub_ur.set_xticks([-8.75, -8.25])
sub_ur.set_yticks([-2, -1, 0, 1, 2])
sub_ur.set_xlim(-8.75, -8.25)
sub_ur.set_ylim(-2, 2)
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
sub_ax.plot(data_ibamr[:, 0],data_ibamr[:, 1],  alpha=1.0, linestyle='None', marker='+', markersize=5, markeredgewidth=0.8, label='IBAMR')
sub_ax.plot(deformed[:, 1],deformed[:, 0], alpha=1.0, linestyle='None', marker='x', markersize=5, markeredgewidth=0.8, label='Pulse')
sub_ax.plot(deformed_afsi[:, 1],deformed_afsi[:, 0], alpha=1.0, label='Present')
sub_ax.set_xticks([-2, 0])  
sub_ax.set_yticks([-15, -14, -13])  
sub_ax.set_xlim(-2, 0)
sub_ax.set_ylim(-15, -13)
sub_ax.set_xlabel('x(mm)')
sub_ax.set_ylabel('z(mm)')

sub_ax.text(
    -0.25, 1.10, '(c)', 
    transform=sub_ax.transAxes,  
    fontsize=12, 
    fontweight='bold',
    va='top', ha='right'
)

plt.savefig("../figures/systole_plot.png", dpi=300, bbox_inches='tight')  