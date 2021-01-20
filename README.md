# Inverted Encoding

Python package for easy implementation of inverted encoding modeling as described in Scotti, Chen, & Golomb (in-prep).

Contact: scottibrain@gmail.com (Paul Scotti)

---

## Installation 

Run the following to install:

```python
pip install inverted_encoding
```

---

## Usage

```python
from inverted_encoding import IEM, permutation

predictions, confidences, recons = IEM(trialbyvoxel,features,stim_max=180,is_circular=True)

null_mae_distribution = permutation(features,stim_max=180,num_perm=1000)
```

