import os

import craftutils.utils as u
import craftutils.params as p

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
default_input_path = os.path.join(
    default_data_dir, "input"
)
u.mkdir_check_nested(default_output_path, False)
