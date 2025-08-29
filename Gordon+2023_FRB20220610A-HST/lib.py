import os

from astropy import units

import craftutils.utils as u
import craftutils.params as p

from craftutils.observation import field, sed, image

script_dir = os.path.dirname(__file__)

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

param_dir = os.path.join(script_dir, "param")
fld = field.Field.from_params("FRB20220610A")
frb20220610a = fld.frb

data_path = os.path.join(script_dir, "data")
pzdm_path = os.path.join(data_path, "FRB20220610A_pzgdm.npz")

sample = sed.SEDSample.from_params(
    "Gordon+2023"
)
sample.set_output_dir(os.path.join(output_path, "sed_sample"))
img_ir = None
img_uvis = None
imgs = ()

def load_images():
    global img_ir
    img_ir = image.HubbleImage(os.path.join(input_path, "FRB20220610A_F160W_60mas_full_drz_sci.fits"))
    img_ir.load_data()
    img_ir.load_output_file()
    if not img_ir.zeropoint_best:
        img_ir.zeropoint()
        img_ir.update_output_file()
    img_ir.extract_pixel_scale()
    global img_uvis
    img_uvis = image.HubbleImage(os.path.join(input_path, "FRB20220610A_F606W_30mas_drc_sci.fits"))
    img_uvis.load_data()
    img_uvis.load_output_file()
    if not img_uvis.zeropoint_best:
        img_uvis.zeropoint()
        img_uvis.update_output_file()
    img_uvis.extract_pixel_scale()

    # Insert some numbers from Gordon+2023
    img_ir.headers[0]["PSF_FWHM"] = (3.699 * units.pixel).to("arcsec", img_ir.pixel_scale_y).value
    img_ir.headers[0]["ASTM_RMS"] = 0.054
    img_uvis.headers[0]["PSF_FWHM"] = (3.033 * units.pixel).to("arcsec", img_ir.pixel_scale_y).value
    img_uvis.headers[0]["ASTM_RMS"] = 0.024

    global imgs
    imgs = (img_ir, img_uvis)
    return imgs


