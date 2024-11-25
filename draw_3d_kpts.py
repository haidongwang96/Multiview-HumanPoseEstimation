import os
import utility as su
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

kpts_3d = su.read_pickle_file("kpts_3d.pkl")
print(kpts_3d.shape)

sk_util = su.skeleton_util()

os.makedirs("pose", exist_ok=True)
for i, p3ds in enumerate(kpts_3d):
    plot_save_name = f"data/pose/pose_3d_{i}.png"
    su.pose_3d_plot(p3ds)
    plt.savefig(plot_save_name)
