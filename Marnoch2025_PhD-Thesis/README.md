The scripts found here are for reproducing certain results described in my forthcoming PhD thesis.

### Prerequisites
 - `astropy`: tested with version `5.0.4`
 - `craft-optical-followup`(https://github.com/Lachimax/craft-optical-followup); tested with:
   - Branch: `marnoch+2023`
 - `matplotlib`: tested with version `3.5.2`
 - `numpy`: tested with version `1.22.3`

### Data

To inquire after the data used by these scripts, please contact me directly.

### Scripts

#### `observations.py`

Generates a Latex table of CRAFT VLT observations.


#### `pipeline_stages.py`

Generates figures illustrating pipeline stages.


#### `validation.py`

Generates the figures for the Validation section.


#### `path/`
The scripts found in this directory are for reproducing the results in Appendix D of my PhD thesis.
Scripts in this subdirectory have a set of prerequisites separate to those below; please see the README file there.

#### `frb_populations/`
The scripts found in this directory are for reproducing the results in Chapter 3 of my forthcoming PhD thesis.

Scripts in this subdirectory have a set of prerequisites separate to those below; please see the README file there.

#### `astrometry-tests/`
These scripts are for running the tests comparing the various astrometry solving methods for use in the
craft-optical-followup imaging pipeline, described in-depth in forthcoming thesis.

Scripts in this subdirectory have a set of prerequisites separate to those below; please see the README file there.
