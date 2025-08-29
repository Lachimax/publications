import os

import craftutils.utils as u
import craftutils.params as p

script_dir = os.path.dirname(__file__)

params = p.load_params(os.path.join(script_dir, "params.yaml"))
paper_dir, subdir_name = os.path.split(script_dir)
_, paper_name = os.path.split(paper_dir)

default_data_dir = os.path.join(
    p.config["publications_output_dir"],
    paper_name,
    "astrometry_tests"
)

default_output_path = os.path.join(
    str(default_data_dir), "output"
)
output_path = default_output_path
os.makedirs(default_output_path, exist_ok=True)


def set_output_path(path):
    global output_path
    output_path = path
    os.makedirs(output_path, exist_ok=True)


default_input_path = os.path.join(
    default_data_dir, "input"
)
input_path = default_input_path


def set_input_path(path):
    global input_path
    input_path = path

dropbox_path = "/home/lachlan/Dropbox/Apps/Overleaf/PhD Thesis/02_pipeline"

exclude = [
            "FRB20240203",
            "FRB20240310",
            "FRB20240525A"
        ]