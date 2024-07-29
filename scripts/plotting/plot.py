import argparse
import os
import pandas as pd
import util

import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process logfiles.')
parser.add_argument('datadir', type=str, help='Directory containing JSON files', default=".")
parser.add_argument('outdir', type=str, help='Directory to save plots', default=".")

args = parser.parse_args()
datadir = args.datadir
outdir = args.outdir

os.makedirs(outdir, exist_ok=True)

files=[x for x in os.listdir(datadir) if x.endswith(".json") and not x.startswith("run_log_fcs-x86")]
files.sort()
files=[os.path.join(datadir,file) for file in files]
df_all, launch_count_list = util.parse_json_files(files)

descriptive_names = {
    'sim_clean': 'workspace_reset',
    'sim_A': 'simulate',
    'sim_ct': 'reduce',
    'sim_cp': 'copy d->h',
}

df_all['kernel'] = df_all['kernel'].replace(descriptive_names)

fig, axes = plt.subplots(2, 1, figsize=(10, 9))

sns.barplot(ax=axes[0], x='kernel', y='total /s', hue='technology', data=df_all, errorbar=None, capsize=0.1)
axes[0].set_title('FastCaloSim Run time comparison for different technologies and kernels')
axes[0].set_ylabel('Total elapsed run time [s]')
axes[0].legend(title='Technology', loc='upper right', fontsize=12) 

sns.barplot(ax=axes[1], x='kernel', y='avg launch /us', hue='technology', data=df_all, errorbar=None , capsize=0.1)
axes[1].set_title('Average Kernel Run time in us')
axes[1].set_ylabel('Average Run time and standard deviation [us]')
axes[1].legend(title='Technology', loc='upper right', fontsize=12)
    
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
launch_count_str = util.make_launch_count_str(launch_count_list)
axes[1].text(0.02, 0.95, launch_count_str, transform=axes[1].transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='left', bbox=props)

df_errorbar = util.make_df_for_errorbar(df_all, axes[1])

axes[1].errorbar(df_errorbar['bar_x'], df_errorbar['bar_y'], yerr=df_errorbar['std dev /us'], fmt='none', c='red', capsize=5)
axes[1].set_ylim(bottom=0)  # Set y-axis minimum to 0

plt.tight_layout()

filename = os.path.join(outdir, 'Results.png')
# plt.savefig(filename)
plt.show()

