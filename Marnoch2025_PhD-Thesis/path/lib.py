import os

import craftutils.utils as u
import craftutils.params as p

from craftutils.observation import objects

import craftutils.observation.output as outp

script_dir = os.path.dirname(__file__)

params = p.load_params(os.path.join(script_dir, "params.yaml"))
paper_dir, subdir_name = os.path.split(script_dir)
_, paper_name = os.path.split(paper_dir)

default_data_dir = os.path.join(
    os.path.expanduser("~"),
    "Data",
    "publications",
    paper_name,
    subdir_name
)

default_output_path = os.path.join(
    default_data_dir, "output"
)
output_path = default_output_path
u.mkdir_check_nested(default_output_path, False)
dropbox_path = "/home/lachlan/Dropbox/Apps/Overleaf/PhD Thesis/06_path"
tex_path = os.path.join(output_path, "tex")
os.makedirs(tex_path, exist_ok=True)
table_path = os.path.join(output_path, "tables")
os.makedirs(table_path, exist_ok=True)


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

dropbox_path = "/home/lachlan/Dropbox/Apps/Overleaf/PhD Thesis/06_path"
dropbox_figs = os.path.join(dropbox_path, "figures")

def savefig(fig, filename, subdir=None, tight=True):
    output_this = output_path
    db_this = dropbox_figs
    if subdir is not None:
        output_this = os.path.join(output_this, subdir)
        db_this = os.path.join(db_this, subdir)
    os.makedirs(output_this, exist_ok=True)
    os.makedirs(db_this, exist_ok=True)
    output = os.path.join(output_this, filename)
    print("Saving figure to ", output + ".pdf")
    if tight:
        bb = "tight"
    else:
        bb = None
    fig.savefig(output + ".pdf", bbox_inches=bb)
    fig.savefig(output + ".png", bbox_inches=bb, dpi=200)
    fig.savefig(os.path.join(db_this, filename + ".pdf"), bbox_inches=bb)

zdm_dir = os.path.join(script_dir, "zdm")


def generate_zdm_command(
        frb: objects.FRB
):
    name = f"{frb.name}_pzgdm"
    print(name, frb.dm, frb.snr, frb.width_total, frb.host_galaxy.z)
    if not frb.dm:
        return None
    survey = frb.survey
    if not survey:
        return None
    command = frb._p_z_dm_command()

    return command + "\n"

optical_cat = None

def load_photometry_table(force: bool = False):
    global optical_cat
    if optical_cat is None or force:
        optical_cat = outp.OpticalCatalogue("optical")
    optical_cat.load_table(force=force)
    host_photometry = optical_cat.to_astropy()
    return host_photometry


