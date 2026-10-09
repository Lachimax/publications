#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os
import shutil

import matplotlib.pyplot as plt
import cmasher as cmr

from astropy import table

import craftutils.utils as u
import craftutils.observation.field as field
from craftutils.params import load_params
import craftutils.plotting as pl

from craftutils.wrap import galfit

import lib

description = """
Generates residual figures showcasing the GALFIT models.  This will not work unless you have all of the imaging data where it is expected, so primarily for internal use.
"""


def main(
        output_dir: str,
        input_dir: str,
        test_plot: bool
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    cmap = "cmr.wildfire"

    exclude = ["FRB20230731A", "FRB20250106", "FRB20250201"]

    plt.rcParams['axes.linewidth'] = 1.

    tex_path = os.path.join(lib.tex_path, "residual_figures.tex")

    if test_plot:
        field_list = ["FRB20210807D"]
    else:
        field_list = list(filter(lambda f: f.startswith("FRB"), field.list_fields()))
        field_list.sort()
    output_path_galfit = os.path.join(output_dir, "galfit_residuals")
    # shutil.rmtree(output_path_galfit)
    os.makedirs(output_path_galfit, exist_ok=True)

    fpath = os.path.join(lib.output_path, "frames.yaml")
    frames = load_params(fpath)

    first_success = False

    file_lines = []

    for field_name in field_list:
        print(field_name)
        fld = field.Field.from_params(field_name)
        frb = fld.frb
        host = frb.host_galaxy
        if field_name in exclude:
            print(f"Skipping {field_name} (in exclude list)")
        elif host is not None:
            host.load_output_file()
            print("\t Checking for GALFIT model...")
            if host.galfit_models is not None:
                if "best" in host.galfit_models:
                    model_dict = host.galfit_models["best"]
                    sersic = model_dict["COMP_2"].copy()
                    sersic["image"] = model_dict["image"]
                    sersic["field_name"] = field_name
                    sersic["object_name"] = host.name
                    if host.z is not None and host.z > -990.:
                        sersic["z"] = host.z
                    frame_model = sersic["frame"]
                    galfit_dir = os.path.join(
                        host.data_path,
                        "GALFIT",
                        os.path.basename(model_dict["image"]).replace(".fits", "")
                    )
                    imgblock_path = os.path.join(galfit_dir, f"{host.name}_galfit-out_{int(frame_model)}.fits")
                    print("Using file", imgblock_path)
                    frame_sersic = sersic["r_eff"].value * 2.5
                    print("\tFrame from GALFIT model:", frame_sersic)
                    if field_name in frames:
                        frame_file = frames[field_name].value
                        print("\tFrame from frames.yaml:", frame_file)
                    else:
                        frame_file = 0.
                        print("\tNo frame in frames.yaml:", frame_file)

                    frame_min = 15

                    if frame_file <= frame_sersic:
                        if frame_sersic <= frame_min:
                            frame_fig = frame_min
                            print("\tUsing min frame:", frame_min)
                        else:
                            # frame_fig = frame_model * 0.5
                            frame_fig = frame_sersic
                            print("\tUsing frame from GALFIT model:", frame_fig)
                    else:
                        frame_fig = frame_file
                        print("\tUsing frame from frames.yaml:", frame_file)

                    fig = plt.figure(figsize=(pl.textwidths["PASA"], pl.textwidths["PASA"] / 4))
                    fig, margins = galfit.imgblock_plot(
                        imgblock_path,
                        # output=os.path.join(output_path_galfit, f"residuals_{field_name}"),
                        frame=frame_fig,
                        fig=fig,
                        cmap=cmap
                    )
                    fig.tight_layout(w_pad=-2)

                    if fld.frb.tns_name is None:
                        title = field_name
                    else:
                        title = fld.frb.tns_name
                    title_nice = title.replace("FRB", "FRB\,")
                    # fig.suptitle(title_nice, fontsize=8, ha="center")
                    lib.savefig(
                        fig,
                        filename=f"residuals_{title}",
                        subdir="galfit_residuals",
                        tight=True
                    )
                    if not first_success:
                        first_label = title

                    caption = rf"\galfit{{}} model subtraction for {title_nice}. "
                    if not first_success:
                        caption += "The white ellipse demonstrates the fitted \Reff{} and axis ratio, with the long axis equal to \Reff{}. The black cross marks the model centroid."
                    else:
                        caption += rf" Markings are explained in the caption of \autoref{{fig:residuals:{first_label}}}."
                    caption += "\n" + r"\figscript{" + u.latex_sanitise(os.path.basename(__file__)) + "}"

                    first_success = True

                    fig_text = r"""
\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{03_host_properties/figures/galfit_residuals/residuals_""" + title + r""".pdf}
    \caption[\galfit{} subtraction of """ + title_nice + r"""]{""" + caption + r"""}
    \label{fig:residuals:""" + title + r"""}
\end{figure}
                    """
                    file_lines.append(fig_text)

        if test_plot and first_success:
            break

        print()

    if not test_plot and lib.dropbox_path is not None:
        with open(tex_path, "w") as f:
            f.writelines(file_lines)
        shutil.copy(tex_path, os.path.join(lib.dropbox_path, "figures", "galfit_residuals"))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=description
    )

    parser.add_argument(
        "-o",
        help="Path to output directory.",
        type=str,
        default=lib.default_output_path
    )
    parser.add_argument(
        "-i",
        help="Path to directory containing input files.",
        type=str,
        default=lib.default_input_path
    )
    parser.add_argument(
        "--test_plot",
        help="Generates only a couple of plots, for testing purposes. Latex file generation will be skipped.",
        action="store_true"
    )

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path,
        test_plot=args.test_plot
    )
