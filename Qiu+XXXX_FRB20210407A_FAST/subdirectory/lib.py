import os

import craftutils.utils as u
import craftutils.params as p

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
