The scripts found in this directory are for reproducing the results in Chapter 5 of my PhD thesis.

### Prerequisites
 - `astropy`: tested with version `5.0.4`
 - `craft-optical-followup`(https://github.com/Lachimax/craft-optical-followup); tested with:
   - Branch: `marnoch+2023`
 - `matplotlib`: tested with version `3.5.2`
 - `numpy`: tested with version `1.22.3`

### Data

To inquire after the data used by these scripts, please contact me directly.

### Scripts

#### `01-smhm.py`

Generates figures showing off the stellar-to-halo-mass relationships.


#### `02-halo_table.py`

Performs fiducial halo calculations.


#### `03-0-halo_mc.py`

Does Monte Carlo modelling of foreground halos.


#### `03-1-halo_mc_collate.py`

Collates the results of MC modelling.


#### `03-2-halo_mc_analysis.py`

Analyses the collated results of the MC modelling.


#### `05-0-rmax-fhot.py`

Does the grid modelling to constrain Rmax and fhot.


#### `05-1-rmax-fhot_analysis.py`

Analyses the fhot-Rmax grid models.


#### `07-plots.py`

Generates miscellaneous figures.


#### `09-imaging.py`

Generates imaging figures of foreground galaxies.


#### `11-0-halo_sky_arrays.py`

Generates the sky grids of DM_halos.


#### `11-1-halo_sky_plots.py`

Generates the figures showing sky projections of DM_halos.


#### `13-scattering.py`

Performs the scattering analysis.


#### `14-intersection_probabilities.py`

Does some calculations of FRB probabilities of intersecting halos at mass thresholds.


#### `15-dm_cumulative.py`

Generates the cumulative DM figures.


#### `17-FGe.py`

Analysis of whether FGe is actually int he foreground.


#### `18-groups.py`

Analysis of the foreground galaxy group candidates.


#### `20-latex_tables.py`

Translates tables into Latex for thesis insertion.

