import os

import craftutils.utils as u
import craftutils.params as p

from craftutils.observation import field, image
from craftutils.observation import instrument

from astropy import units, table

script_dir = os.path.dirname(__file__)

p.set_param_dir(path=os.path.join(script_dir, "param"), write=False)

params = p.load_params(os.path.join(script_dir, "params.yaml"))
_, paper_name = os.path.split(script_dir)

default_data_dir = os.path.join(
    os.path.expanduser("~"),
    "Data",
    "publications",
    paper_name
)

default_output_path = os.path.join(
    default_data_dir, "output"
)
output_path = default_output_path
u.mkdir_check_nested(default_output_path, False)
repo_data = os.path.join(script_dir, "data")


def set_output_path(path):
    global output_path
    output_path = path
    u.mkdir_check_nested(output_path, False)


default_input_path = os.path.join(
    default_data_dir, "input"
)
input_path = default_input_path


def set_input_path(path):
    global input_path
    input_path = path


fld = field.Field.from_params("FRB20210407A")
fld.gather_epochs_imaging()
frb210407 = fld.frb
frb210407.read_p_z_dm(path=os.path.join(repo_data, "james.c.w-p-z-dm", "210407_pzgdm.npz"))

deimos = instrument.Instrument.from_params("keck-deimos")
mmt = instrument.Instrument.from_params("sdss")

z_deimos = deimos.filters["z"]
z_deimos.lambda_eff = 9085 * units.Angstrom
z_trans_tbl = table.QTable.read(
    os.path.join(repo_data, "deimos", "z_deimos_throughput.txt"),
    format="ascii"
)
z_trans_tbl["Wavelength"] *= units.Angstrom
tbl_fil = z_trans_tbl["Wavelength", "Transmission_filter"]
tbl_fil["Transmission"] = tbl_fil["Transmission_filter"]
tbl_fil.remove_column("Transmission_filter")
z_deimos.transmission_tables["filter"] = tbl_fil

tbl_fil = z_trans_tbl["Wavelength", "Transmission_filter+ccd"]
tbl_fil["Transmission"] = tbl_fil["Transmission_filter+ccd"]
tbl_fil.remove_column("Transmission_filter+ccd")
z_deimos.transmission_tables["filter+instrument"] = tbl_fil
z_deimos.write_transmission_tables()
z_deimos.update_output_file()

i_mmt = mmt.filters["i"]

img_z = None
img_i = None


def load_images():
    global img_z
    img_z = image.CoaddedImage(
        os.path.join(input_path, "FRB20210407_2021-09-09_Keck-DEIMOS_z_wcs.fits")
    )
    img_z.load_output_file()
    img_z.filter = z_deimos
    img_z.instrument = deimos

    global img_i
    img_i = image.CoaddedImage(
        os.path.join(input_path, "FRB210407_i_offset_i_1_11_right_wcs.fits")
    )
    img_i.load_output_file()
    img_i.filter = i_mmt
    img_i.instrument = i_mmt

    return img_z, img_i
