import os

import craftutils.utils as u
import craftutils.params as p

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
os.makedirs(default_output_path, exist_ok=True)
repo_data = os.path.join(script_dir, "data")

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
    os.makedirs(input_path, exist_ok=True)

dropbox_path = None # "/home/lachlan/Dropbox/Apps/Overleaf/PhD Thesis/02_pipeline"
dropbox_figs = None
