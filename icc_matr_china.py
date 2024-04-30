from denoising.helpers import *
from denoising.metrics import *
import numpy as np
import os
import pandas as pd


# information about sessions
df = pd.read_csv('/arch/OpenCloseBeijin/BeijingEOEC.csv')
df = df.loc[df['SubjectID'] != 3258811]
closed_ids2 = df['SubjectID'].loc[df['Session_2'] == 'closed'].values
closed_ids3 = df['SubjectID'].loc[df['Session_3'] == 'closed'].values

datapath = '/data/Projects/OpenCloseChina/outputs_china'


os.mkdir(f'{datapath}/icc')
os.mkdir(f'{datapath}/icc/3ses')
os.mkdir(f'{datapath}/icc/1vs2')
os.mkdir(f'{datapath}/icc/1vs3')



for atlas in ['AAL', 'Brainnetome', 'HCPex']:
    for strategy in range(1, 7):

        closed12 = fetch_ts(datapath,
                        sub=closed_ids2, 
                        run=1, atlas_name=atlas, strategy=strategy)
        closed13 = fetch_ts(datapath,
                        sub=closed_ids3, 
                        run=1, atlas_name=atlas, strategy=strategy)


        closed2 = fetch_ts(datapath,
                        sub=closed_ids2, 
                        run=2, atlas_name=atlas, strategy=strategy)
        closed3 = fetch_ts(datapath,
                        sub=closed_ids3, 
                        run=3, atlas_name=atlas, strategy=strategy)
        
        s1 = fetch_ts(datapath,
                      #sub=closed_ids3, 
                      run=1, atlas_name=atlas, strategy=strategy)
        s2 = fetch_ts(datapath,
                      #sub=closed_ids3, 
                      run=2, atlas_name=atlas, strategy=strategy)
        s3 = fetch_ts(datapath,
                      #sub=closed_ids3, 
                      run=3, atlas_name=atlas, strategy=strategy)
        

        fc_cl12 = functional_connectivity(closed12, 'correlation')
        fc_cl13 = functional_connectivity(closed13, 'correlation')
        fc_cl2 = functional_connectivity(closed2, 'correlation')
        fc_cl3 = functional_connectivity(closed3, 'correlation')
        fc1 = functional_connectivity(s1, 'correlation')
        fc2 = functional_connectivity(s2, 'correlation')
        fc3 = functional_connectivity(s3, 'correlation')

        d = {'1vs2': [fc_cl12, fc_cl2],
             '1vs3': [fc_cl13, fc_cl3],
             '3ses': [fc1, fc2, fc3]}
        
        for i in d:

            metric = ICC(d[i])
            m = metric.icc()

            np.save(f'{datapath}/icc/{i}/icc-{round(np.mean(m), 3)}_{atlas}_strategy-{strategy}.npy', m)

