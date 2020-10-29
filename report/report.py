import matplotlib.pyplot as plt
import numpy as np
import os
import json


if __name__ == '__main__':
    data = {}
    json_names = [x for x in os.listdir('./') if 'peta-valid' in x]
    for json_name in json_names:
        with open(json_name) as fjson:
            acc_dict = json.load(fjson)
            for key in acc_dict:
                if key not in data:
                    data[key] = [acc_dict[key]]
                else:
                    data[key].append(acc_dict[key])
    
    data = {k: np.array(v, dtype=np.float32) for k, v in data.items()}

    plt.boxplot(data.values())
    plt.xticks(ticks=range(len(data)), labels=data.keys(), rotation=70, ha='left')
    plt.subplots_adjust(bottom=0.22)
#    plt.show()
    plt.savefig('report_boxplot.png')
