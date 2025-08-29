These scripts are for running the tests comparing the various astrometry solving methods for use in the
craft-optical-followup imaging pipeline, described in-depth in forthcoming thesis.


### Scripts

#### `01-astrometry_tests.py`
Main script for performing astrometry tests.
##### Instructions:
  - This script should be run from the directory that contains it.
  - An account and API key from https://nova.astrometry.net/api_help is required; the API key must be placed in the
  `keys.json` file created by craft-optical-followup in the param directory.

#### `02-astrometry_analysis.py`

Performs analysis of the tests, generating tables and figures.


### Prerequisites
 - `astropy`: tested with version `5.0.4`
 - `craft-optical-followup`(https://github.com/Lachimax/craft-optical-followup); tested with:
   - Branch: `astrometry-tests`
 - `matplotlib`: tested with version `3.5.2`
 - `numpy`: tested with version `1.22.3`
