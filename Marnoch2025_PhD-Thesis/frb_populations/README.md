The scripts found in this directory are for reproducing the results in Chapter 3 of my forthcoming PhD thesis.


### Scripts

#### `00-retrieve.py`

Currently does nothing; will retrieve data for the following scripts.


#### `01-frb_table.py`

Downloads a CSV version of a Google Sheet I've been maintaining, which tabulates all of the FRB hosts in the literature 
that I'm aware of (as of September 2024), as well as unpublished CRAFT hosts. It then adds some derived values and re-saves it in a more 
machine-readable format. This takes quite a while; it will also not work .


#### `02-imaging.py`

Generates imaging figures.


#### `11-dm-z_plots.py`

Generates DM-z figures from tables of CRAFT and other FRBs.


#### `12-latex_tables.py`

Uses the derived FRB host table to generate some latex tables.


#### `13-photometry_tables.py`
Generates the table of photometry seen in Shannon et al 2024, **Table X**.

#### `14-galfit.py`

Does some analysis with the GALFIT results and writes some tables.


#### `15-0-galfit_all_bands.py`

Collates all GALFIT results from all bandpasses.


#### `15-1-residuals.py`

Generates residual figures showcasing the GALFIT models.


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
   - Branch: `thesis-main`
 - `matplotlib`: tested with version `3.5.2`
 - `numpy`: tested with version `1.22.3`
