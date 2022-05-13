import argparse
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    n = 8                   # # of files to ensemble
    output_files = []
    ensemble_list=[]
    output_dir = "outputs/"

    for i in range(1,n+1):
        output_files.append(output_dir + '/output' + str(i) +'.csv')

    for path in output_files:
        ensemble_list.append(pd.read_csv(path)['prediction'].values)

    predictions = np.zeros_like(ensemble_list[0])
    for predict in ensemble_list:
        predictions = np.add(predictions, predict)
    predictions /= len(ensemble_list)

    with open('outputs/ensemble{}.csv'.format(n), 'w', encoding='utf8') as f:
        f.write("id,prediction\n")
        for idx, row in enumerate(predictions):
            f.write('{},{}\n'.format(idx, predictions[idx]))