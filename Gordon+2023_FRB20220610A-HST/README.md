The scripts found in this directory are for reproducing some analysis performed for Gordon et al 2023.

### Prerequisites
 - `astropy`: tested with version `5.0.4`
 - `craft-optical-followup`(https://github.com/Lachimax/craft-optical-followup); tested with:
   - Branch: `marnoch+2023`
 - `matplotlib`: tested with version `3.5.2`
 - `numpy`: tested with version `1.22.3`

### Data

To inquire after the data used by these scripts, please contact me directly.

### Scripts

#### `00-sed_sample.py`

Runs PATH (Probabilistic Association of Transients; Aggarwal et al 2021) in various configurations on the HST imaging 
data covering the field of FRB 20220610A, and generates some figures.


#### `01-p_u.py`

Performs P(U) calculations for the field of FRB 20220610A.


#### `02-path.py`

Runs PATH (Probabilistic Association of Transients; Aggarwal et al 2021) in various configurations on the HST imaging 
data covering the field of FRB 20220610A, and generates some figures.


### References
 - Gordon, A. C. et al 2023: *A Fast Radio Burst in a Compact Galaxy Group at z ∼ 1*, ApJL 963:2:L34 [https://doi.org/10.3847/2041-8213/ad2773]]
