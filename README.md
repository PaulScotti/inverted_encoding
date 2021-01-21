# Inverted Encoding

Python package for easy implementation of inverted encoding modeling as described in Scotti, Chen, & Golomb (in-prep).

Contact: scottibrain@gmail.com (Paul Scotti)

---

## Installation 

Run the following to install:

```python
pip install inverted-encoding
```

---

## Usage

```python
from inverted_encoding import IEM, permutation, circ_diff
import numpy as np

predictions, confidences, recons = IEM(trialbyvoxel,features,stim_max=180,is_circular=True)
# trialbyvoxel: your matrix of brain activations, does not necessarily have to be voxels
# features: array of your stimulus features (must be integers within range defined by stim_max)
# use "help(IEM)" for more information on required inputs

## Compute mean absolute error (MAE) by doing the following, then compare to null distribution:
if is_circular: # if your stimulus space is circular, need to compute circular differences
    mae = np.mean(np.abs(circ_diff(predictions,features,stim_max))) 
else:
    mae = np.mean(np.abs(predictions-features)) 
null_mae_distribution = permutation(features,stim_max=180,num_perm=1000,is_circular=True)
```

