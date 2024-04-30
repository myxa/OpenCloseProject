from denoising.denoise import Denoising
from denoising.atlas import Atlas
from denoising.dataset import Dataset
from denoising.helpers import *
from denoising.metrics import *
import numpy as np
import os

datapath = '/data/Projects/TestRetest_NYU/TRT_outputs'
os.mkdir(f'{datapath}/icc')
os.mkdir(f'{datapath}/icc/3ses')
os.mkdir(f'{datapath}/icc/1vs2')

for atlas in ['AAL', 'Brainnetome', 'HCPex']:
    for strategy in range(1, 7):

        r1 = fetch_ts(datapath, 
                      run=1, atlas_name=atlas, strategy=strategy)
        r2 = fetch_ts(datapath,
                      run=2, atlas_name=atlas, strategy=strategy)
        r3 = fetch_ts(datapath,
                      run=3, atlas_name=atlas, strategy=strategy)
        

        fc1 = functional_connectivity(r1, 'correlation')
        fc2 = functional_connectivity(r2, 'correlation')
        fc3 = functional_connectivity(r3, 'correlation')

        metric = ICC([fc1, fc2, fc3])
        m = metric.icc()

        np.save(f'{datapath}/icc/3ses/icc-{round(np.mean(m), 3)}_{atlas}_strategy-{strategy}.npy', m)

        metric = ICC([fc1, fc2])
        m = metric.icc()

        np.save(f'{datapath}/icc/1vs2/icc-{round(np.mean(m), 3)}_{atlas}_strategy-{strategy}.npy', m)

