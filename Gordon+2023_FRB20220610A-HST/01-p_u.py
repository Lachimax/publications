#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as units
import astropy.table as table
import astropy.coordinates as coordinates

import craftutils.utils as u
import craftutils.params as p
import craftutils.plotting as pl
import craftutils.observation.image as image
import craftutils.observation.instrument as instrument
import craftutils.observation.sed as sed

import lib

description = """
Performs P(U) calculations for the field of FRB 20220610A.
"""


def main(
        output_dir: str,
        input_dir: str
):

    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    pl.latex_setup()

    frb220610 = lib.fld
    # frb220610.retrieve_catalogues()

    frb220610_zdm = lib.frb20220610a.read_p_z_dm(lib.pzdm_path)

    # Set up instruments and bands
    hst_ir = instrument.Instrument.from_params("hst-wfc3_ir")
    f160w = hst_ir.filters["F160W"]
    hst_uv = instrument.Instrument.from_params("hst-wfc3_uvis2")
    f606w = hst_uv.filters["F606W"]

    # fors2 = instrument.Instrument.from_params("vlt-fors2")
    # g_fors2 = fors2.filters["g_HIGH"]
    # I_fors2 = fors2.filters["I_BESS"]

    # bands = [f160w, f606w]

    # Load HST image from disk
    lib.load_images()

    # Load SED model sample
    sample = lib.sample
    sample.load_output_file()
    sample.read_z_mag_tbls()

    # Generate a table of the group objects
    hst_cands = {"id": [], "position": []}
    for obj in frb220610.objects:
        if obj.name.startswith("Gordon"):
            hst_cands["id"].append(obj.name[-1])
            hst_cands["position"].append(obj.position)
    hst_cand_tbl = table.QTable(hst_cands)
    hst_cand_tbl["ra"] = hst_cand_tbl["position"].ra
    hst_cand_tbl["dec"] = hst_cand_tbl["position"].dec

    # Specify an empty coordinate for depth testing
    test_coord = coordinates.SkyCoord("23h24m19.2894s -33d30m00.554s")

    # Set up the plot
    centre = coordinates.SkyCoord(np.mean(hst_cand_tbl["position"].ra), np.mean(hst_cand_tbl["position"].dec))
    fig = plt.figure(figsize=(pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] * 2 / 3))
    for i, img in enumerate(lib.imgs):
        results = {}
        name = img.instrument_name
        output_this = os.path.join(lib.output_path, name)
        u.mkdir_check(output_this)

        print(f"\n Processing {name}, {img.filter.name}, {img.filter.cmap} ({img.name})")

        # Estimate depth with a 2 * PSF_FWHM aperture
        depth_tbl = img.test_limit_location(
            test_coord,
            ap_radius=2 * img.extract_psf_fwhm()
        )
        depth_tbl.write(os.path.join(output_this, f"depth_{name}.ecsv"), overwrite=True)
        depth_5sigma = depth_tbl[4]["mag"]
        results["depth"] = depth_5sigma
        print("5-sigma depth:", depth_5sigma)
        vals, tbl, z_lost = sample.host_probability_unseen(
            band=img.filter,
            limit=depth_5sigma,
            obj=frb220610_zdm,
            show=True,
            plot=True,
            output=output_this
        )
        results.update(vals)
        results["z_lost"] = z_lost

        # Plot Gordon+2023 identifiers on the image
        fig, ax, _ = frb220610.plot_host(
            img,
            fig=fig,
            n_x=2,
            n_y=1,
            n=i + 1,
            centre=centre,
            frame=3 * units.arcsec,
            imshow_kwargs={"cmap": img.filter.cmap},
            include_img_err=True
        )
        ax.set_title(f"{img.instrument.formatted_name}, {img.filter.name}")
        hst_cand_tbl[f"x_{name}"], hst_cand_tbl[f"y_{name}"] = img.world_to_pixel(hst_cand_tbl["position"])
        offset = img.pixel(0.1 * units.arcsec).value
        for row in hst_cand_tbl:
            ax.text(
                row[f"x_{name}"] + offset,
                row[f"y_{name}"] + offset,
                f"$\mathbf{{{row['id']}}}$",
                c="white",
                fontsize=pl.tick_fontsize
            )
        ra, dec = ax.coords
        dec.set_axislabel(" ")
        ra.set_axislabel(" ")
        if i == 1:
            dec.set_ticklabel_visible(False)
        else:
            dec.set_ticklabel(fontsize=pl.tick_fontsize)
        ra.set_ticks(
                values=units.Quantity([
                    coordinates.Angle("23h24m17.9s"),
                    coordinates.Angle("23h24m17.74s"),
                    coordinates.Angle("23h24m17.58s"),
                    coordinates.Angle("23h24m17.42s"),
                    # Angle("23h24m17.45s")
                ]).to("deg"),
            )
        ra.set_ticklabel(
            # rotation=45,
            fontsize=pl.tick_fontsize,
            # pad=40,
            horizontalalignment="right"
        )
        model = results["p(z|DM,U)"]["normal"]
        results["p(z|DM,U)"]["normal"] = dict(zip(model.param_names, model.parameters))
        for key, value in results.items():
            print(f"{key}:\n\t", value)
        p.save_params(os.path.join(output_this, f"{name}_results.yaml"), results)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_frame_on(False)
    ax.tick_params(left=False, right=False, top=False, bottom=False)
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks([])
    # ax.set_ticks([])
    # ax.set_aspect("equal")
    ax.set_xlabel(
        "Right Ascension (J2000)",
        # labelpad=30,
        fontsize=pl.axis_fontsize
    )
    ax.set_ylabel(
        "Declination (J2000)",
        rotation=-90,
        labelpad=30,
        fontsize=pl.axis_fontsize
    )
    ax.yaxis.set_label_position("right")

    fig.savefig(os.path.join(lib.output_path, f"FRB20220610A_group_labelled.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(lib.output_path, f"FRB20220610A_group_labelled.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    hst_cand_tbl.write(os.path.join(lib.output_path, "candidates.ecsv"), overwrite=True)
    hst_cand_tbl.write(os.path.join(lib.output_path, "candidates.csv"), overwrite=True)



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

    args = parser.parse_args()
    output_path = args.o
    input_path = args.i
    main(
        output_dir=output_path,
        input_dir=input_path
    )
