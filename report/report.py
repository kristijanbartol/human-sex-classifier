import json
import numpy as np
import matplotlib.pyplot as plt
from random import sample


# JSON keys.
ACC = 'acc'
OPENPOSE = 'openpose'
MISSING = 'missing'


if __name__ == '__main__':
    with open('people3d-openpose.json') as rf:
        data = json.load(rf)
        accs = []
        openpose = []
        missing = []

        # Select a subset of actions.
        subset_keys = sample([x for x in data[ACC] if 'man' not in x], 20)
        # Edit names.
        parts = [x.split('_') for x in subset_keys]
        edited_keys = [f'{x[0]}_{x[1]}' for x in parts] 

        for key in subset_keys:
            print(key)
            accs.append(data[ACC][key])
            openpose.append(data[OPENPOSE][key])
            missing.append(data[MISSING][key])

        accs = np.array(accs, dtype=np.float32)
        openpose = np.array(openpose, dtype=np.float32)
        missing = np.array(missing, dtype=np.float32)

        accs_mean = np.mean(accs)
        openpose_mean = np.mean(openpose)
        missing_mean = np.mean(missing)

        accs *= (openpose_mean / accs_mean)
        missing *= (openpose_mean / missing_mean) / 4.

        # set width of bar
        barWidth = 0.45

        # Set position of bar on X axis
        r1 = np.arange(len(accs))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r1]
        
        # Make the plot
        plt.bar(r1, accs, color='#0d76bd', width=barWidth, edgecolor='white', label='Accuracy')
#        plt.bar(r2, openpose, color='#bbbbbb', width=barWidth, edgecolor='white', label='MPJPE error')
        plt.bar(r3, missing, color='#ed1c23', width=barWidth, edgecolor='white', label='Missing joints')

        # Add xticks on the middle of the group bars
        plt.xlabel('Action ID', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(accs))], edited_keys, rotation=90)

        corrcoef_ = np.corrcoef(accs, openpose)
        print(corrcoef_)

        # Create legend & Show graphic
        plt.legend()
        plt.show()

