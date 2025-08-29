import os

import craftutils.params as p
from craftutils.observation import field
import craftutils.plotting as pl

import os

pl.latex_setup()

model_dir = os.path.join(p.param_dir, "sed_samples/Gordon+2023")
model_files = list(
    filter(
        lambda f: os.path.isfile(os.path.join(model_dir, f)) and f.endswith(".txt"), os.listdir(model_dir)
    )
)
events = list(set(map(lambda f: f[:6], model_files)))
events.sort()

frb_list = field.list_fields()

model_dict = {
    "FRB20180301A": {"z": 0.3304},
    "FRB20180916B": {"z": 0.0337},
    "FRB20190520B": {"z": 0.241},
    "FRB20201124A": {"z": 0.0979},
    "FRB20210410D": {"z": 0.1415}
}

do_not_load = ["FRB20220610"]

for event in events:
    frb_name = f"FRB20{event}"
    # print(frb_name)
    if frb_name in do_not_load:
        continue
    z = None
    found = False
    for frb_this in frb_list:
        if frb_this.startswith(frb_name):
            frb_name = frb_this
            fld_this = field.Field.from_params(frb_name)
            if fld_this.frb.tns_name is not None:
                frb_name = fld_this.frb.tns_name
            z = fld_this.frb.host_galaxy.z
            found = True
    if not found:
        for frb_this in model_dict:
            if frb_this.startswith(frb_name):
                frb_name = frb_this
                z = model_dict[frb_name]["z"]
                found = True
    if not found:
        raise ValueError(f"A redshift could not be found for {frb_name}.")
    if frb_name not in model_dict:
        model_dict[frb_name] = {}

    model_flux_path = os.path.join(model_dir, f"{event}_model_spectrum_FM07.txt")
    if not os.path.isfile(model_flux_path):
        model_flux_path = os.path.join(model_dir, f"{event}_model_spectrum.txt")

    model_wavelength_path = os.path.join(model_dir, f"{event}_model_wavelengths_FM07.txt")
    if not os.path.isfile(model_wavelength_path):
        model_wavelength_path = os.path.join(model_dir, f"{event}_model_wavelengths.txt")

    observed_flux_path = os.path.join(model_dir, f"{event}_observed_spec_FM07.txt")
    if not os.path.isfile(observed_flux_path):
        observed_flux_path = os.path.join(model_dir, f"{event}_observed_spectrum_FM07.txt")

    observed_wavelength_path = os.path.join(model_dir, f"{event}_observed_wave_FM07.txt")

    print(frb_name, z)

    model_dict[frb_name]["type"] = "GordonProspectorModel"
    model_dict[frb_name]["name"] = f"{frb_name}_GordonProspector"
    model_dict[frb_name]["z"] = z
    model_dict[frb_name]["model_flux_path"] = model_flux_path.replace(p.param_dir, "")
    model_dict[frb_name]["model_wavelength_path"] = model_wavelength_path.replace(p.param_dir, "")
    model_dict[frb_name]["observed_flux_path"] = observed_flux_path.replace(p.param_dir, "")
    model_dict[frb_name]["observed_wavelength_path"] = observed_wavelength_path.replace(p.param_dir, "")

model_list = list(model_dict.values())
p.save_params(os.path.join(p.param_dir, "sed_samples", "Gordon+2023", "Gordon+2023.yaml"), model_list)
