The scripts found in this directory are for reproducing the results in Chapter 3 of my forthcoming PhD thesis.


### Scripts

#### `00-retrieve.py`

Retrieve data for the following scripts.


#### `02-imaging.py`

Generates imaging figures. This script is for internal use only, as it depends on the fully-processed imaging data being in the correct place; but it is presented here to show how the figures were generated.


#### `11-dm-z_plots.py`

Generates DM-z figures from tables of CRAFT and other FRBs.


#### `12-latex_tables.py`

Uses the derived FRB host table to generate some latex tables.


#### `13-photometry_tables.py`
Generates the table of photometry seen in Shannon et al 2024, **Table X**.

#### `14-galfit.py`

Does some analysis with the GALFIT results and writes some tables.


#### `15-0-galfit_all_bands.py`

Collates all GALFIT results from all bandpasses. Like 02-imaging, this will not work unless you have all of the imaging data where it is expected, so primarily for internal use.


#### `15-1-residuals.py`

Generates residual figures showcasing the GALFIT models.  Like 02-imaging, this will not work unless you have all of the imaging data where it is expected, so primarily for internal use.


#### `16-acsgc.py`

Performs statistical tests of CRAFT host properties against the ACS-GC (Griffith et al 2012).


#### `17-axis_ratio_correlations.py`

Performs and plots statistical tests for correlations between CRAFT FRB host properties.


#### `18-offset_analysis.py`

Does statistical tests of CRAFT GALFIT FRB host properties against other object populations.


#### `19-reproduce_bhardwaj2024.py`

Uses figures from Bhardwaj et al 2024 as templates for CRAFT FRB hosts.


### Prerequisites
 - `astropy`: tested with version `5.0.4`
 - `craft-optical-followup`(https://github.com/Lachimax/craft-optical-followup); tested with:
   - Release: `v0.5-phd-thesis`
 - `matplotlib`: tested with version `3.5.2`
 - `numpy`: tested with version `1.22.3`
