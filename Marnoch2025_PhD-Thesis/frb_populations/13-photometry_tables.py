#!/usr/bin/env python
# Code by Lachlan Marnoch, 20XX

import os

import numpy as np

import astropy.table as table
import astropy.units as units

import craftutils.utils as u
from astropy.coordinates import SkyCoord

import lib

description = "Generates the table of photometry seen in Shannon et al 2024, **Table X**."


def main(
        output_dir: str,
):
    db_path = os.path.join(lib.dropbox_path, "tables")

    # Load the object properties table.
    bands = ("u-HIGH", "g-HIGH", "R-SPECIAL", "I-BESS", "z-GUNN", "J", "H", "Ks")
    all_objects = lib.load_photometry_table()

    # These are the table columns we care about for the purposes of this table.

    def get_inst(band_name: str):
        if band_name in ("J", "H", "Ks"):
            inst = "vlt-hawki"
        else:
            inst = "vlt-fors2"
        return inst

    colnames = [
        "object_name",
        "field_name",
        f"e_b-v",
    ]

    phot_colnames = []
    ext_colnames = []
    colour_colnames = []

    for band in bands:
        instrument = get_inst(band)
        phot_colnames.append(f"mag_best_{instrument}_{band}")
        phot_colnames.append(f"mag_best_{instrument}_{band}_err")
        ext_colnames.append(f"ext_gal_{instrument}_{band}")
        colour_colnames.append(f"transient_position_surface_brightness_{instrument}_{band}")
        colour_colnames.append(f"transient_position_surface_brightness_{instrument}_{band}_err")

    colnames += phot_colnames + ext_colnames

    # Limit the table to FRB host galaxies, which in this table all start with 'HG'.
    hosts = list(map(lambda r: r["object_name"].startswith("HG"), all_objects))
    hosts = all_objects[hosts]
    hosts["field_name"] = list(map(lambda f: f"{f[3:]}", hosts["field_name"]))
    # Make the field names a bit Latex-fancy (insert '\,' between FRB and the number)


    # HOST PHOTOMETRY, UNCORRECTED
    # ============================

    col_replace = {
        'field_name': "FRB",
        'e_b-v': r"$\ebvmw$",
        'mag_u-HIGH': r"\uhigh",
        'mag_g-HIGH': r"\ghigh",
        'mag_R-SPECIAL': r"\Rspec",
        'mag_I-BESS': r"\Ibess",
        'mag_z-GUNN': r"\zgunn",
        'mag_J': "$J$",
        'mag_H': "$H$",
        'mag_Ks': r"\Ks",
        "colour_g-I": r"$g-I$",
        "colour_R-K": r"$R-K$",
        "colour_local_g-I": r"$(g-I)_\mathrm{local}$",
        "colour_local_R-K": r"$(R-\Ks)_\mathrm{local}$",
    }

    host_photometry = hosts[["field_name"] + phot_colnames]

    # Specify the number of acceptable significant figures in the uncertainties.
    n_digits_err = 1
    # Cycle through each filter name to collect magnitudes
    for band in bands:
        instrument = get_inst(band)

        color_str = []
        for row in host_photometry:
            mag = row[f"mag_best_{instrument}_{band}"]
            mag_err = row[f"mag_best_{instrument}_{band}_err"]
            if mag == 0.:
                this_str = "--"
            else:
                # Put the value and uncertainty in a nice format, as a string.
                this_str, value, uncertainty = u.uncertainty_string(
                    value=mag,
                    uncertainty=mag_err,
                    n_digits_err=n_digits_err,
                    n_digits_lim=3,
                    limit_type="lower"
                )
            color_str.append(this_str)
        # Generate  the column and remove the old ones.
        host_photometry[f"mag_{band}"] = color_str
        host_photometry.remove_column(f"mag_best_{instrument}_{band}")
        host_photometry.remove_column(f"mag_best_{instrument}_{band}_err")

    tbl_path = os.path.join(lib.tex_path, "craft_photometry.tex")
    # We spit this out as a latex table, ready to go.
    print("Writing table to", tbl_path)
    # host_photometry.write(tbl_path, format="ascii.latex", overwrite=True)

    u.latexise_table(
        tbl=host_photometry,
        column_dict=col_replace,
        output_path=tbl_path,
        short_caption="CRAFT host VLT photometry",
        caption=r"Integrated photometry for CRAFT host galaxies, without correcting for Galactic extinction."
                r"\tabscript{" + u.latex_sanitise(os.path.basename(__file__)) + "}",
        label="tab:craft-photometry",
        longtable=True,
        landscape=True,
        coltypes="cccc|ccccc|ccc",
        multicolumn=[(4, "c|", ""), (5, "c|", "FORS2"), (3, "c", "HAWK-I")],
        second_path=os.path.join(db_path, "craft_photometry.tex"),
    )

    # HOST EXTINCTION
    # ===============

    host_ext = hosts[["field_name", "e_b-v", "path_pox"] + phot_colnames + ext_colnames]
    for band in bands:
        instrument = get_inst(band)

        mag_str = []
        for row in host_ext:
            mag = row[f"mag_best_{instrument}_{band}"]
            ext = row[f"ext_gal_{instrument}_{band}"]
            if mag > 0:
                mag -= ext
            mag_err = row[f"mag_best_{instrument}_{band}_err"]
            # Put the value and uncertainty in a nice format, as a string.
            this_str, value, uncertainty = u.uncertainty_string(
                value=mag,
                uncertainty=mag_err,
                n_digits_err=n_digits_err,
                n_digits_lim=3,
                limit_type="lower"
            )
            mag_str.append(this_str)
        # Generate  the column and remove the old ones.
        host_ext[f"mag_{band}"] = mag_str
        host_ext.remove_column(f"ext_gal_{instrument}_{band}")
        host_ext.remove_column(f"mag_best_{instrument}_{band}")
        host_ext.remove_column(f"mag_best_{instrument}_{band}_err")

    tbl_path = os.path.join(lib.tex_path, "craft_photometry_ext.tex")
    # We spit this out as a latex table, ready to go.
    print("Writing table to", tbl_path)
    # host_ext.write(tbl_path, format="ascii.latex", overwrite=True)

    u.latexise_table(
        tbl=host_ext,
        column_dict=col_replace,
        output_path=tbl_path,
        short_caption="CRAFT host VLT photometry",
        caption=r"Integrated photometry for CRAFT host galaxies, corrected for Galactic extinction."
                r"\tabscript{" + u.latex_sanitise(os.path.basename(__file__)) + "}",
        label="tab:craft-photometry",
        longtable=True,
        landscape=True,
        coltypes="cccc|ccccc|ccc",
        multicolumn=[(4, "c|", ""), (5, "c|", "FORS2"), (3, "c", "HAWK-I")],
        second_path=os.path.join(db_path, "craft_photometry_ext.tex"),
    )

    # BOTH
    # ======

    host_both = []
    host_photometry["ebv"] = ["-" * 50] * len(host_photometry)
    host_ext["ebv"] = ["-" * 50] * len(host_ext)
    host_ext["Fig."] = [r"\ref{fig:imaging:FRB" + row["field_name"] + "}" for row in host_ext]
    host_photometry["Fig."] = [r"\ref{fig:imaging:FRB" + row["field_name"] + "}" for row in host_photometry]
    host_photometry["\POx"] = ["-" * 50] * len(host_photometry)
    host_ext["\POx"] = ["-" * 50] * len(host_ext)

    for i, row_ext in enumerate(host_ext):
        row_phot = host_photometry[i]
        row_phot["ebv"] = str(row_ext["e_b-v"].value)
        row_phot["\POx"] = str(row_ext["path_pox"].round(4))

        host_both.append(row_phot)
        row_ext["ebv"] = ""
        row_ext["Fig."] = ""
        row_ext["field_name"] = ""
        row_ext["\POx"] = ""
        for col in host_ext.colnames:
            if isinstance(row_ext[col], str) and "(" in row_ext[col]:
                row_ext[col] = row_ext[col][:-4] + "$"
        host_both.append(row_ext)
    host_both = table.vstack(host_both)
    host_both["e_b-v"] = host_both["ebv"]
    host_both.remove_column("ebv")
    host_both.remove_column("path_pox")
    colnames = host_both.colnames
    colnames.pop(-1)
    colnames.insert(1, "\POx")
    colnames.pop(-1)
    colnames.insert(1, "e_b-v")
    colnames.pop(-1)
    colnames.insert(1, "Fig.")
    host_both = host_both[colnames]

    host_both.remove_columns(["e_b-v", "\POx", "Fig."])

    tbl_path = os.path.join(lib.tex_path, "craft_photometry_both.tex")
    # We spit this out as a latex table, ready to go.
    print("Writing table to", tbl_path)
    # print(host_both.colnames)
    # host_both.write(tbl_path, format="ascii.latex", overwrite=True)

    u.latexise_table(
        tbl=host_both,
        column_dict=col_replace,
        output_path=tbl_path,
        short_caption="CRAFT host VLT photometry",
        caption=r"Integrated photometry for CRAFT host galaxies. In each row, the lower values are corrected for "
                r"Galactic extinction, while the uncertainties for the uncorrected magnitudes are given in brackets. "
                r"All quantities are AB magnitudes, aside from \POx; this gives the PATH association posterior probability "
                r"(with $P(U)$ set to 0.1; see \autoref{chapter:path} for further information). "
                # r"\textbf{Include FURBY target Y/N? Criteria passed Y/N}? "
                r"\tabscript{" + u.latex_sanitise(os.path.basename(__file__)) + "}",
        label="tab:craft-photometry",
        longtable=True,
        landscape=True,
        coltypes="cccc|ccccc|ccc",
        multicolumn=[(4, "c|", ""), (5, "c|", "FORS2"), (3, "c", "HAWK-I")],
        second_path=os.path.join(db_path, "craft_photometry_both.tex"),
    )

    # HOST COLOURS

    host_colour = hosts[["field_name", "e_b-v", "path_pox"] + phot_colnames + ext_colnames + colour_colnames]

    for band in ("g-HIGH", "I-BESS", "R-SPECIAL", "Ks"):
        instrument = get_inst(band)

        sb = []
        sb_err = []
        mag_val = []
        mag_err_val = []
        for row in host_colour:
            mag = row[f"mag_best_{instrument}_{band}"]
            ext = row[f"ext_gal_{instrument}_{band}"]
            if mag > 0:
                mag -= ext

            mag_err = row[f"mag_best_{instrument}_{band}_err"]
            sb_ = row[f"transient_position_surface_brightness_{instrument}_{band}"]
            sb_err_ = row[f"transient_position_surface_brightness_{instrument}_{band}_err"]
            mag_val.append(mag)
            mag_err_val.append(mag_err)
            sb.append(sb_)
            sb_err.append(sb_err_)
            # Put the value and uncertainty in a nice format, as a string.
            # this_str, value, uncertainty = u.uncertainty_string(
            #     value=mag,
            #     uncertainty=mag_err,
            #     n_digits_err=n_digits_err,
            #     n_digits_lim=3,
            #     limit_type="lower"
            # )
            # color_str.append(this_str)
        # Generate  the column and remove the old ones.
        host_colour[f"mag_{band}"] = mag_val
        host_colour[f"mag_{band}_err"] = mag_err_val
        host_colour[f"sb_{band}"] = sb
        host_colour[f"sb_{band}_err"] = sb_err

    for band in bands:
        instrument = get_inst(band)
        host_colour.remove_column(f"ext_gal_{instrument}_{band}")
        host_colour.remove_column(f"mag_best_{instrument}_{band}")
        host_colour.remove_column(f"mag_best_{instrument}_{band}_err")
        host_colour.remove_column(f"transient_position_surface_brightness_{instrument}_{band}")
        host_colour.remove_column(f"transient_position_surface_brightness_{instrument}_{band}_err")


    for band_pair in (("g-HIGH", "I-BESS"), ("R-SPECIAL", "Ks")):
        band_short, band_long = band_pair
        colour = []
        colour_err = []
        colour_str = []
        local = []
        local_err = []
        local_str = []
        for row in host_colour:
            mag_short = row[f"mag_{band_short}"]
            mag_short_err = row[f"mag_{band_short}_err"]
            mag_long = row[f"mag_{band_long}"]
            mag_long_err = row[f"mag_{band_long}_err"]
            if mag_short > 0 * units.mag and mag_long > 0 * units.mag:
                c = mag_short - mag_long
                if mag_short_err > 0 * units.mag and mag_long_err > 0 * units.mag:
                    c_err = np.sqrt(mag_short_err ** 2 + mag_long_err ** 2)
                else:
                    c_err = -999 * units.mag
            else:
                c = -999 * units.mag
                c_err = -999 * units.mag

            this_str, value, uncertainty = u.uncertainty_string(
                value=c,
                uncertainty=c_err,
                n_digits_err=n_digits_err,
                n_digits_lim=3,
                limit_type="lower"
            )
            colour_str.append(this_str)

            colour.append(c)
            colour_err.append(c_err)

            sb_short = row[f"sb_{band_short}"]
            sb_short_err = row[f"sb_{band_short}_err"]
            sb_long = row[f"sb_{band_long}"]
            sb_long_err = row[f"sb_{band_long}_err"]
            print(row["field_name"], sb_short, sb_short_err, sb_long, sb_long_err)
            if sb_short > 0 and sb_long > 0 and np.isfinite(sb_short) and np.isfinite(sb_long):
                c_local = sb_short - sb_long
                if sb_short_err > 0 and sb_long_err > 0:
                    c_local_err = np.sqrt(sb_short_err ** 2 + sb_long_err ** 2)
                else:
                    c_local_err = -999 * units.mag
            else:
                c_local = -999 * units.mag
                c_local_err = -999 * units.mag

            local.append(c_local)
            local_err.append(c_local_err)

            this_str, value, uncertainty = u.uncertainty_string(
                value=c_local,
                uncertainty=c_local_err,
                n_digits_err=n_digits_err,
                n_digits_lim=3,
                limit_type="lower"
            )
            local_str.append(this_str)



        host_colour[f"colour_{band_short[0]}-{band_long[0]}"] = colour_str
        # host_colour[f"colour_{band_short[0]}-{band_long[0]}_err"] = colour_err
        host_colour[f"colour_local_{band_short[0]}-{band_long[0]}"] = local_str
        # host_colour[f"colour_local_{band_short[0]}-{band_long[0]}_err"] = local_err
        host_colour.remove_column(f"mag_{band_short}")
        host_colour.remove_column(f"mag_{band_short}_err")
        host_colour.remove_column(f"mag_{band_long}")
        host_colour.remove_column(f"mag_{band_long}_err")
        host_colour.remove_column(f"sb_{band_short}")
        host_colour.remove_column(f"sb_{band_short}_err")
        host_colour.remove_column(f"sb_{band_long}")
        host_colour.remove_column(f"sb_{band_long}_err")

    tbl_path = os.path.join(lib.tex_path, "craft_colours.tex")

    host_colour["ebv"] = ["-" * 50] * len(host_colour)
    host_colour["Fig."] = [r"\ref{fig:imaging:FRB" + row["field_name"] + "}" for row in host_colour]
    host_colour["\POx"] = ["-" * 50] * len(host_colour)

    for i, row in enumerate(host_colour):
        row["ebv"] = str(row["e_b-v"].value)
        row["\POx"] = str(row["path_pox"].round(4))
        # host_both.append(row_ext)

    host_colour["e_b-v"] = host_colour["ebv"]
    host_colour.remove_column("path_pox")
    host_colour.remove_column("ebv")

    colnames = host_colour.colnames
    colnames.pop(-1)
    colnames.insert(1, "\POx")
    colnames.pop(-1)
    colnames.insert(1, "Fig.")
    host_colour = host_colour[colnames]

    u.latexise_table(
        tbl=host_colour,
        column_dict=col_replace,
        output_path=tbl_path,
        short_caption="CRAFT host VLT colours",
        caption=r"Colours for CRAFT host galaxies. "
                # r"In each row, the lower values are corrected for "
                # r"Galactic extinction, while the uncertainties for the uncorrected magnitudes are given in brackets. "
                # r"All quantities are AB magnitudes, aside from \POx; this gives the PATH association posterior probability "
                # r"(with $P(U)$ set to 0.1; see \autoref{chapter:path} for further information). "
                # r"\textbf{Include FURBY target Y/N? Criteria passed Y/N}? "
                r"\tabscript{" + u.latex_sanitise(os.path.basename(__file__)) + "}",
        label="tab:craft-photometry",
        longtable=True,
        # landscape=True,
        coltypes="cccccccc",
        # multicolumn=[(4, "c|", ""), (2, "c|", "FORS2"), (3, "c", "HAWK-I")],
        second_path=os.path.join(db_path, "craft_colours.tex"),
    )


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
    )
