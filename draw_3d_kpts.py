import os

import numpy as np
import pickle

import utility as su
import matplotlib.pyplot as plt



kpts_3d = su.read_pickle_file("kpts_3d_video1.pkl")

sk_util = su.skeleton_util()

for i, p3ds in enumerate(kpts_3d):

    print(p3ds)
    su.print_block()
    #plot_save_name = f"{pose_dir}/pose_3d_{i}.jpg"
    #fig = su.pose_3d_plot(p3ds)
    #plt.savefig(plot_save_name)
    #plt.close()
