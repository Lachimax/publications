These scripts are for reproducing the results of Marnoch et al 2026.

### Prerequisites
 - `astropy`: tested with version `5.0.4`
 - `craft-optical-followup`(https://github.com/Lachimax/craft-optical-followup); tested with:
   - Release: `v0.5-phd-thesis`
 - `matplotlib`: tested with version `3.10.6`
 - `numpy`: tested with version `2.3.4`

### Data

The data used by most of these scripts will be downloaded by `00-retrieve.py`. The scripts `15-0-galfit_all_bands.py` and `15-1-residuals.py` require a lot of data products to be in particular locations, and so are mainly for internal use; however, if you're brave enough to try and reproduce every step of this paper, contact me and we can probably work it out.

### Scripts

#### `00-retrieve.py`

Retrieve data for the following scripts.


#### `12-latex_stats.py`

Uses the derived FRB host table to generate some latex tables.


#### `14-galfit.py`

Does some analysis with the GALFIT results and writes some tables.


#### `15-0-galfit_all_bands.py`

Collates all GALFIT results from all bandpasses. This will not work unless you have all of the imaging data where it is expected, so primarily for internal use.


#### `15-1-residuals.py`

Generates residual figures showcasing the GALFIT models.  This will not work unless you have all of the imaging data where it is expected, so primarily for internal use.


#### `16-1-acsgc.py`

Performs statistical tests of CRAFT host properties against the ACS-GC (Griffith et al 2012).


#### `16-2-sdss.py`

Performs statistical tests of CRAFT host properties against SDSS DR16 (SDSS Collaboration 2020).


#### `17-correlations.py`

Performs and plots statistical tests for correlations between properties of CRAFT FRBs and their hosts.


#### `19-reproduce_bhardwaj2024.py`

Uses figures from Bhardwaj et al 2024 as templates for CRAFT FRB hosts.


#### `20-comparative_imaging.py`

Generates the imaging figures that compare our VLT imaging with the survey imaging used by B+24.



### References
 - Bhardwaj, M. et al 2024: *Selection bias obfuscates the discovery of fast radio burst sources* Nature 634:8036 [https://doi.org/10.1038/s41586-024-08065-w]
 - Griffith, R. et al 2012: *The Advanced Camera for Surveys General Catalog: Structural Parameters for Approximately Half a Million Galaxies*, ApJS 200:1 [https://doi.org/10.1088/0067-0049/200/1/9]
 - SDSS Collaboration 2020: *The 16th Data Release of the Sloan Digital Sky Surveys: First Release from the APOGEE-2 Southern Survey and Full Release of eBOSS Spectra*, ApJS 249:3 [https://doi.org/10.3847/1538-4365/ab929e]
