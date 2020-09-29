import numpy as np
import pandas as pd
import asymNdimPdf as apdf

def getMeas(alpha):
    return apdf.ndimSkewNormal(alpha=alpha).measAsymError()

alphas = np.concatenate([
    np.linspace(-18, -2, 5000, endpoint=False),
    np.linspace( -2,  0, 1000, endpoint=False),
    np.linspace(  0,  2, 1000, endpoint=False),
    np.linspace(  2, 18, 5000, endpoint=False),
])

meas = []
Ntot = len(alphas)
for i, a in enumerate(alphas):
    meas.append(getMeas(a))
    if i%500 == 0:
        print('{:.0f}% complete'.format(float(i/Ntot)*100))
meas = np.array(meas)


df = pd.DataFrame(data={
    'alpha'  : alphas, 
    'mode'   : meas[:, 0],
    'neg'    : meas[:, 1],
    'pos'    : meas[:, 2], 
    'pos/neg': meas[:, 2]/meas[:, 1],
})

df.to_csv('ParameterTableTEST.csv')
