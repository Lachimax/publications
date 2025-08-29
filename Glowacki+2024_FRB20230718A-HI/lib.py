import os

import craftutils.utils as u
import craftutils.params as p
from craftutils.observation import objects, field, instrument, image

script_dir = os.path.dirname(__file__)

params = p.load_params(os.path.join(script_dir, "params.yaml"))
_, paper_name = os.path.split(script_dir)

top_data_dir = p.data_dir
if top_data_dir is None:
    top_data_dir = os.path.join(
        os.path.expanduser("~"),
        "Data",
    )

default_data_dir = os.path.join(
    top_data_dir,
    "publications",
    paper_name
)

default_output_path = os.path.join(
    default_data_dir, "output"
)
output_path = default_output_path
u.mkdir_check_nested(default_output_path, False)


def set_output_path(path):
    global output_path
    output_path = path
    u.mkdir_check_nested(output_path, False)


default_input_path = os.path.join(
    default_data_dir, "input"
)
input_path = default_input_path
data_path = os.path.join(script_dir, "data")
param_path = os.path.join(script_dir, "param")


def set_input_path(path):
    global input_path
    input_path = path
    u.mkdir_check_nested(input_path, False)


fld = field.Field.from_params("FRB20230718A")
fld.frb.get_host()

decam = instrument.Instrument.from_params("decam")

g_decam = decam.filters["g"]
r_decam = decam.filters["r"]
z_decam = decam.filters["z"]


def cat_path():
    return os.path.join(input_path, "DECaPS_cat.ecsv")


cutout_name = "DECaPS_cutout_{}.fits"


def cutout_path(band="r"):
    return os.path.join(input_path, cutout_name.format(band))
