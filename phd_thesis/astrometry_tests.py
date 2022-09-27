# Code by Lachlan Marnoch, 2022

# This script is for running the tests for comparing the various astrometry solving methods for use in the
# craft-optical-followup imaging pipeline, to be described in-depth in forthcoming thesis.

# Instructions:
#   This script should be run from the directory that contains it.

# Prerequisites:
#   craft-optical-followup; tested on astrometry-tests branch:
#       https://github.com/Lachimax/craft-optical-followup/tree/astrometry-tests
#   To run test 5, an account and API key from https://nova.astrometry.net/api_help; the API key must be placed in the
#   'keys.json' file created by craft-optical-followup in the param directory.

import os
import sys
import traceback

import astropy.units as units

import craftutils.observation.field as field
import craftutils.params as p
import craftutils.utils as u
import craftutils.observation.image as image

keys = p.keys()


def main(
        test_dir: str,
        astrometry_net_path: bool,
        fields: str = None,
        epochs: str = None,
        exclude: str = None,
        override_status: bool = False,
        retry: bool = False,
):
    # Create directory to perform tests in.
    u.mkdir_check_nested(test_dir, remove_last=False)

    gaia_path = os.path.join(test_dir, "astrometry_tests_ngaia.yaml")
    ngaia = p.load_params(gaia_path)
    if ngaia is None:
        ngaia = {}
    # Load the status file, which records whether each test, for each epoch, has been performed.
    # If a test succeeds, it will be recorded as 'fine'; otherwise, the traceback call is recorded.
    status_path = os.path.join(test_dir, "astrometry_tests_status.yaml")
    status = p.load_params(status_path)
    if status is None:
        status = {}
    # Load the files file, which records the path of the file on which astrometry tests were performed; chiefly for
    # debug purposes.
    file_path = os.path.join(test_dir, "astrometry_tests_files.yaml")
    files = p.load_params(file_path)
    if files is None:
        files = {}
    # Load the results file, in which the epoch-specific astrometry and PSF statistics for each test are stored.
    results_path = os.path.join(test_dir, "astrometry_tests_results.yaml")
    results = p.load_params(results_path)
    if results is None:
        results = {}

    # This function determines whether to perform a given test, based on script arguments and whether
    # the experiment has been performed successfully previously (as determined from the 'status' file)
    def do_this(
            exp_name: str,
    ):
        # If passed --override_status, the script re-processes everything.
        if override_status:
            do = True
        elif retry and status[epoch_name][exp_name]["processing"] != "fine":
            do = True
        else:
            do = status[epoch_name][exp_name]["processing"] is None
        return do

    def experiment(
            epoch: field.FORS2ImagingEpoch,
            exp_name: str,
            stage_names: list,
            message: str,
            frames_name: str,
            coadded_name: str,
            stage_kwargs: dict = {},
            switch_coadded: dict = None,
    ):
        """
        The function below is just for convenience, in order to avoid copy/pasting code; for the passed Epoch object,
        :param epoch: the Epoch on which the test is being run.
        :param exp_name: the experiment name with which to label dictionary entries, paths etc.
        :param stage_names: names of reduction stages to perform for this test.
        :param message: message to print before the test.
        :param frames_name: name of frames dictionary to use for coaddition; "science", "reduced", "trimmed",
            "normalised", "registered", "astrometry" or "diagnosed"
        :param coadded_name: name of coadded dictionary to run astrometry tests on.
        :param stage_kwargs: Keyword arguments to pass to stage methods.
        :param switch_coadded: a dictionary of CoaddedImage objects to replace the Epoch's coadded dict.
        :return:
        """
        # Set up dictionaries to catch status and results for this experiment.
        if exp_name not in status[epoch_name]:
            status[epoch_name][exp_name] = {
                "processing": None,
                "analysis": None
            }
        if exp_name not in files[epoch_name]:
            files[epoch_name][exp_name] = {
                "coadded_from": None,
                "diagnostics_from": None
            }
        # Set up epoch directory
        test_dir_epoch = os.path.join(test_dir, epoch_name)
        u.mkdir_check(test_dir_epoch)
        # Override normal output path with our test directory
        test_dir_epoch_exp = os.path.join(test_dir_epoch, exp_name)
        u.mkdir_check(test_dir_epoch_exp)
        epoch.data_path = test_dir_epoch_exp
        # Build the path where the new Epoch output file will be located.
        new_output_file = os.path.join(
            epoch.data_path,
            f"{epoch.name}_outputs.yaml"
        )

        if not os.path.isfile(new_output_file):
            # Read in output file from normal reduction
            epoch.output_file = old_output_file
            epoch.load_output_file()
            # Switch path to output file
            epoch.output_file = new_output_file
            # Write copy of old output file to new directory
            epoch.update_output_file()

        # Set the Epoch output file to the new path, so that values derived from these images go here instead of to the
        # primary epoch file.
        epoch.output_file = new_output_file
        epoch.load_output_file()

        if switch_coadded is not None:
            epoch.coadded_astrometry = {}
            for fil in switch_coadded:
                epoch.add_coadded_astrometry_image(
                    img=switch_coadded[fil],
                    key=fil
                )
        ############
        # Processing
        ############
        # Do this test?
        do = do_this(exp_name=exp_name)
        if do:
            try:
                print(f"Doing {message} (processing)")
                epoch.do = stage_names
                # Set up the special parameters for this experiment
                for stage_name in epoch.do:
                    if stage_name not in epoch.param_file:
                        epoch.param_file[stage_name] = {}
                        if stage_name in stage_kwargs:
                            epoch.param_file[stage_name] = stage_kwargs[stage_name]
                    # Force the stages to be performed as required, overwriting the input param file
                    epoch.do_kwargs[stage_name] = True
                if "coadd" not in epoch.param_file:
                    epoch.param_file["coadd"] = {}
                epoch.param_file["coadd"]["frames"] = frames_name
                if "source_extraction" not in epoch.param_file:
                    epoch.param_file["source_extraction"] = {}
                epoch.param_file["source_extraction"]["image_type"] = coadded_name
                # So that we're on an even footing for all tests, set the offset tolerance for astrometry diagnostics to
                # 2 arcseconds
                epoch.param_file["source_extraction"]["offset_tolerance"] = 2 * units.arcsec
                # Run the epoch's pipeline, on the relevant stages and with the correct input parameters, as established
                # above.
                epoch.pipeline()
                epoch.update_output_file()
                # Tell the 'files' file which files offsets were actually measured from
                files[epoch_name][exp_name]["coadded_from"] = field._output_img_dict_list(
                    epoch._get_frames(frames_name))
                files[epoch_name][exp_name]["diagnostics_from"] = field._output_img_dict_single(
                    epoch._get_images(coadded_name))
                # If all of this worked, set status to 'fine'. If it didn't, the exception below will trigger.
                status[epoch_name][exp_name]["processing"] = "fine"
            except:
                # This too-broad except block exists entirely as a convenience so that failed epochs/tests get skipped
                # over gracefully without having to rerun the script.
                tb_message = []
                # Catch and save traceback for failure, in case something can be learned from it.
                for tb in traceback.format_tb(sys.exc_info()[2]):
                    tb_message.append(str(tb))
                status[epoch_name][exp_name]["processing"] = tb_message
                p.save_params(status_path, status)
                p.save_params(file_path, files)
            p.save_params(status_path, status)
            ##########
            # Analysis
            ##########
            try:
                # Pull astrometry test results out of output files.
                print(f"Doing {message} (analysis)")
                results_epoch_ast = epoch.astrometry_stats
                results_epoch_psf = epoch.psf_stats
                results_epoch = {
                    "astrometry": results_epoch_ast,
                    "psf": results_epoch_psf
                }

                if epoch_name not in results:
                    results[epoch_name] = {}
                results[epoch_name][exp_name] = results_epoch
                epoch.output_file = os.path.join(
                    epoch.data_path,
                    f"{epoch.name}_outputs.yaml")
                epoch.update_output_file()
                status[epoch_name][exp_name]["analysis"] = "fine"
            except:
                # This too-broad except block exists entirely as a convenience so that failed epochs/tests get skipped
                # over gracefully without having to rerun the script.
                # Catch and save traceback for failure, in case something can be learned from it.
                tb_message = []
                for tb in traceback.format_tb(sys.exc_info()[2]):
                    tb_message.append(str(tb))
                status[epoch_name][exp_name]["analysis"] = tb_message
            epoch.update_output_file()
            p.save_params(status_path, status)
            p.save_params(results_path, results)

    if exclude is not None:
        exclude = exclude.split(",")
    else:
        exclude = []

    # Gather FRB fields
    if fields is None:
        fields = list(filter(lambda n: n.startswith("FRB") and n not in exclude, field.list_fields()))
    else:
        fields = fields.split(",")

    print(fields)

    for fld_name in fields:
        print("Processing field", fld_name)
        if fld_name in exclude:
            continue
        # Forcing Gaia catalogs onto DR3, and checking how this affects the number of stars.
        fld = field.FRBField.from_params(fld_name)
        if fld_name not in ngaia:
            fld.retrieve_catalogue("gaia", force_update=True, data_release=2)
            gaia_dr2 = fld.load_catalogue("gaia", data_release=2)
            ngaia[fld_name] = {
                "dr2": len(gaia_dr2)
            }
            fld.retrieve_catalogue("gaia", force_update=True)
            gaia_dr3 = fld.load_catalogue("gaia")
            ngaia[fld_name]["dr3"] = len(gaia_dr3)
            p.save_params(gaia_path, ngaia)
        print(fld.gather_epochs_imaging)
        # Gather imaging epochs for this field
        epochs_imaging = fld.gather_epochs_imaging()
        print("Found epochs:")
        print(list(epochs_imaging.keys()))
        for epoch_name in epochs_imaging:
            epoch_dict = epochs_imaging
            if epoch_dict[epoch_name]["instrument"] == "vlt-fors2" and (epochs is None or epoch_name in epochs):
                print("Processing epoch", epoch_name)
                # Initialise epoch object
                if epoch_name not in status:
                    status[epoch_name] = {}
                if epoch_name not in files:
                    files[epoch_name] = {}
                ######################################
                # 1. Coaddition without any correction
                ######################################
                print("Initialising epoch...")
                epoch = field.FORS2ImagingEpoch.from_params(
                    name=epoch_name,
                    instrument="vlt-fors2",
                    field=fld
                )
                old_output_file = os.path.join(
                    epoch.data_path,
                    f"{epoch.name}_outputs.yaml"
                )
                experiment(
                    epoch=epoch,
                    exp_name="1-no_correction",
                    stage_names=[
                        "coadd",
                        "source_extraction"
                    ],
                    message="1. Coaddition without correction",
                    frames_name="normalised",
                    coadded_name="coadded"
                )
                del epoch

                ######################################
                # 2. Offline Gaia on coadded image
                ######################################
                print("Initialising epoch...")
                epoch = field.FORS2ImagingEpoch.from_params(
                    name=epoch_name,
                    instrument="vlt-fors2",
                    field=fld
                )
                experiment(
                    epoch=epoch,
                    exp_name="2-gaia_coadded",
                    stage_names=[
                        "coadd",
                        "correct_astrometry_coadded",
                        "source_extraction",
                    ],
                    message="2. Offline Gaia on coadded image",
                    frames_name="normalised",
                    coadded_name="coadded_astrometry",
                )
                del epoch

                ######################################
                # 3. Offline Gaia on individual, no epoch correction
                ######################################
                print("Initialising epoch...")
                epoch = field.FORS2ImagingEpoch.from_params(
                    name=epoch_name,
                    instrument="vlt-fors2",
                    field=fld
                )
                experiment(
                    epoch=epoch,
                    exp_name="3-gaia_individual_no_epoch_correction",
                    stage_names=[
                        "correct_astrometry_frames",
                        "coadd",
                        "source_extraction",
                    ],
                    message="3. Offline Gaia on individual, no epoch correction",
                    frames_name="astrometry",
                    coadded_name="coadded",
                    stage_kwargs={"correct_astrometry_frames": {"correct_to_epoch": False}}
                )
                del epoch

                ######################################
                # 4. Offline Gaia on individual, with epoch correction
                ######################################
                print("Initialising epoch...")
                epoch = field.FORS2ImagingEpoch.from_params(
                    name=epoch_name,
                    instrument="vlt-fors2",
                    field=fld
                )
                experiment(
                    epoch=epoch,
                    exp_name="4-gaia_individual_epoch_correction",
                    stage_names=[
                        "correct_astrometry_frames",
                        "coadd",
                        "source_extraction",
                    ],
                    message="4. Offline Gaia on individual, with epoch correction",
                    frames_name="astrometry",
                    coadded_name="coadded",
                )
                del epoch

                ######################################
                # 5. Online Astrometry.net on coadded images
                ######################################
                print("Initialising epoch...")
                epoch = field.FORS2ImagingEpoch.from_params(
                    name=epoch_name,
                    instrument="vlt-fors2",
                    field=fld
                )
                exp_name_this = "5-astrometry_upload"
                if exp_name_this not in status[epoch_name]:
                    status[epoch_name][exp_name_this] = {
                        "processing": None,
                        "analysis": None
                    }
                # This test is a special case - we need to send the file to Astrometry.net online for processing, and
                # then swap it in for analysis.
                if status[epoch_name][exp_name_this]["processing"] != "fine" or override_status:
                    test_dir_epoch = os.path.join(test_dir, epoch_name)
                    for fil in epoch.filters:
                        # Set up the various arguments to pass to the astrometry-client script
                        filename = f"{epoch.name}_{epoch.date_str()}_{fil}_coadded_mean-sigmaclip.fits"
                        path_to_uncorrected = os.path.join(test_dir_epoch, "1-no_correction", "8-coadd", fil, filename)
                        output_path = os.path.join(test_dir_epoch, exp_name_this)
                        u.mkdir_check(output_path)
                        output_file = os.path.join(output_path,
                                                   filename.replace(
                                                       "coadded_mean-sigmaclip.fits",
                                                       "nova_astrometry.fits"))
                        ra = fld.centre_coords.ra.value
                        dec = fld.centre_coords.dec.value
                        u.system_command_verbose(
                            command=f'python2 "{astrometry_net_path}" --apikey "{keys["astrometry"]}" -u "{path_to_uncorrected}" -w --newfits "{output_file}" --ra "{ra}" --dec "{dec}" --radius 0.5 --private --no_commercial',
                        )

                        img_ast = image.FORS2CoaddedImage(output_file)
                        switch = {
                            fil: img_ast
                        }
                    # If the final file is missing, then nova.astrometry.net hasn't worked.
                    if os.path.isfile(output_file):
                        experiment(
                            epoch=epoch,
                            exp_name=exp_name_this,
                            stage_names=[
                                "source_extraction",
                            ],
                            message="Online Astrometry.net on coadded images",
                            frames_name="reduced",
                            coadded_name="coadded_astrometry",
                            switch_coadded=switch
                        )
                    else:
                        status[epoch_name][exp_name_this]["processing"] = "nova failure"
                del epoch

                p.save_params(status_path, status)
                p.save_params(file_path, files)
                p.save_params(results_path, results)

        del fld


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Script for testing of astrometry correction methods."
    )
    default_path = os.path.join(
        os.path.expanduser("~"), "Data", "test", "astrometry_2022"
    )
    parser.add_argument(
        "-p",
        help="Path to output directory. Must have plenty of space.",
        type=str,
        default=f"/home/lachlan/Data/test/astrometry_2022/"
    )
    parser.add_argument(
        "--fields", "--field",
        help="Comma-separated field names to include. If not provided, all available fields will be processed.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--exclude",
        help="Comma-separated field names to exclude. Overrides --fields",
        type=str,
        default=None
    )
    parser.add_argument(
        "--epochs",
        help="Comma-separated epoch names to include. If not provided, all available epochs will be processed",
        type=str,
        default=None
    )
    parser.add_argument(
        "--override_status",
        help="Reprocess all fields, including successful ones.",
        action="store_true"
    )
    parser.add_argument(
        "--retry",
        help="Reprocess failed fields.",
        action="store_true"
    )
    default_path_client = os.path.join(
        os.path.expanduser("~"), "Projects", "publications", "utils", "astrometry-client.py"
    )
    parser.add_argument(
        "--astrometry_net_path",
        help="Path to astrometry-client.py.",
        type=str,
        default=default_path_client
    )

    args = parser.parse_args()

    main(
        test_dir=args.p,
        fields=args.fields,
        exclude=args.exclude,
        epochs=args.epochs,
        override_status=args.override_status,
        retry=args.retry,
        astrometry_net_path=args.astrometry_net_path
    )
