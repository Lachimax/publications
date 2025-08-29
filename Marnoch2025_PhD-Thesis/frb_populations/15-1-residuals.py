#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os
import shutil

import matplotlib.pyplot as plt

from astropy import table

import craftutils.utils as u
import craftutils.observation.field as field
from craftutils.params import load_params
import craftutils.plotting as pl

from craftutils.wrap import galfit

import lib

description = """
Generates residual figures showcasing the GALFIT models.
"""


def main(
        output_dir: str,
        input_dir: str,
        test_plot: bool
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    exclude = ["FRB20230731A"]

    tex_path = os.path.join(lib.tex_path, "residual_figures.tex")


    field_list = list(filter(lambda f: f.startswith("FRB"), field.list_fields()))

    output_path_galfit = os.path.join(output_dir, "galfit_residuals")
    # shutil.rmtree(output_path_galfit)
    os.makedirs(output_path_galfit, exist_ok=True)

    frames = load_params(os.path.join(lib.output_path, "frames.yaml"))

    first_success = False

    file_lines = []

    for field_name in field_list:
        print(field_name)
        fld = field.Field.from_params(field_name)
        frb = fld.frb
        host = frb.host_galaxy
        if host is not None and field_name not in exclude:
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
                    frame = sersic["frame"]
                    galfit_dir = os.path.join(
                        host.data_path,
                        "GALFIT",
                        os.path.basename(model_dict["image"]).replace(".fits", "")
                    )
                    imgblock_path = os.path.join(galfit_dir, f"{host.name}_galfit-out_{int(frame)}.fits")
                    if field_name in frames:
                        frame_fig = frames[field_name].value
                        print("Frame found:", frame_fig)
                    else:
                        frame_fig = 0.
                        print("Frame not found:", frame_fig)

                    if frame_fig >= frame:
                        frame_fig = frame

                    fig = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] / 4))
                    fig, margins = galfit.imgblock_plot(
                        imgblock_path,
                        # output=os.path.join(output_path_galfit, f"residuals_{field_name}"),
                        frame=frame_fig,
                        fig=fig
                    )
                    fig.tight_layout(w_pad=0)

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

    if not test_plot:
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
