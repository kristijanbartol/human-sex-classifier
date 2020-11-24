import json
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from scipy.stats.stats import pearsonr


# JSON keys.
ACC = 'acc'
OPENPOSE = 'openpose'
MISSING = 'missing'


if __name__ == '__main__':
    with open('people3d-openpose.json') as rf:
        op_data = json.load(rf)
    with open('people3d-gt.json') as rf:
        gt_data = json.load(rf)

    op_accs = []
    op_openpose = []
    op_missing = []
    gt_accs = []
    gt_openpose = []
    gt_missing = []

    # Select a subset of actions.
    subset_keys = [x for x in op_data[ACC] if 'man' not in x]
#   subset_keys = sample(subset_keys, 20)
    # Edit names.
#   parts = [x.split('_') for x in subset_keys]
#   subset_names = [f'{x[0]}_{x[1]}' for x in parts] 
    subset_names = subset_keys

    for key in subset_keys:
        print(key)
        op_accs.append(op_data[ACC][key])
        op_openpose.append(op_data[OPENPOSE][key])
        op_missing.append(op_data[MISSING][key])

        gt_accs.append(gt_data[ACC][key])
        gt_openpose.append(gt_data[OPENPOSE][key])
        gt_missing.append(gt_data[MISSING][key])

    op_accs = np.array(op_accs, dtype=np.float32)
    op_openpose = np.array(op_openpose, dtype=np.float32)
    op_missing = np.array(op_missing, dtype=np.float32)

    gt_accs = np.array(gt_accs, dtype=np.float32)


    op_accs_mean = np.mean(op_accs)
    op_openpose_mean = np.mean(op_openpose)
    op_missing_mean = np.mean(op_missing)

    gt_accs_mean = np.mean(gt_accs)

#    op_accs *= (op_openpose_mean / op_accs_mean)
#    op_missing *= (op_openpose_mean / op_missing_mean) / 4.

#    gt_accs *= (op_openpose_mean / gt_accs_mean)

    # set width of bar
    barWidth = 0.45

    # Set position of bar on X axis
    r1 = np.arange(len(op_accs))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r1]
        
    '''
    # Make the plot
    plt.bar(r1, op_accs, color='#0d76bd', width=barWidth, edgecolor='white', label='Accuracy (noisy)')
#   plt.bar(r2, openpose, color='#bbbbbb', width=barWidth, edgecolor='white', label='MPJPE error')
#   plt.bar(r3, missing, color='#ed1c23', width=barWidth, edgecolor='white', label='Missing joints')

    # Make the plot
    plt.bar(r2, gt_accs, color='#bbbbbb', width=barWidth, edgecolor='white', label='Accuracy (GT)')
#   plt.bar(r2, openpose, color='#bbbbbb', width=barWidth, edgecolor='white', label='MPJPE error')
#   plt.bar(r3, missing, color='#ed1c23', width=barWidth, edgecolor='white', label='Missing joints')

    # Add xticks on the middle of the group bars
    plt.xlabel('Subject ID', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(op_accs))], subset_names, rotation=90)
    '''

    op_accs *= 100

#    fit_line = np.polyfit(op_accs, op_openpose, 1)
    fit_line = np.polyfit(op_accs, op_missing, 1)

#    plt.plot(op_accs, op_openpose, 'r^')
    plt.plot(op_accs, op_missing, 'g^')
    plt.plot(op_accs, fit_line[0] * op_accs + fit_line[1], color='darkblue', 
            linewidth=2)
#    plt.axis([0, 100, 0, 20])
    plt.axis([0, 100, 0, 6])
    plt.xlabel('Accuracy (per action)')
#    plt.ylabel('Average MPJPE (per action)')
    plt.ylabel('Average number of missing joints (per action)')

    print(np.corrcoef(op_openpose, op_accs))
    print(np.corrcoef(op_missing, op_accs))

#    plt.savefig('mpjpe-accuracy-plot.png')
    plt.savefig('missing-accuracy-plot.png')

