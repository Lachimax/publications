import os
import urllib
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import odr

from ltsfit.ltsfit import ltsfit

import astropy.coordinates as coord
import astropy.table as table
import astropy.units as units
from astropy.cosmology import Planck18
from astropy.modeling import models, fitting

import craftutils.params as p
import craftutils.utils as u
from craftutils.observation import field
from craftutils.observation.objects.frb import FRB, dm_units
import craftutils.observation.output as outp
import craftutils.plotting as pl
from craftutils.utils import dequantify

pl.latex_setup()

script_dir = os.path.dirname(__file__)

params = p.load_params(os.path.join(script_dir, "params.yaml"))
paper_dir, subdir_name = os.path.split(script_dir)
_, paper_name = os.path.split(paper_dir)

default_data_dir = os.path.join(
    p.config["publications_output_dir"],
    paper_name,
    subdir_name
)

default_output_path = os.path.join(
    default_data_dir, "output"
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


frb_table_gs = table.QTable()
# frb_nohost_table_gs = table.QTable()
frb_table = table.QTable()

table_path = os.path.join(output_path, "tables")
os.makedirs(table_path, exist_ok=True)

host_csv_path = os.path.join(table_path, "frb_hosts_gs.csv")
undetected_csv_path = os.path.join(table_path, "frb_no_hosts_gs.csv")
frb_path = os.path.join(table_path, "frb_hosts_derived.ecsv")
bib_path = os.path.join(table_path, "frb_hosts_bib.csv")

def craft_galfit_path(q_0=0.2):
    return os.path.join(table_path, f"craft_galfit_{q_0}.ecsv")

# craft_galfit_path = os.path.join(table_path, "craft_galfit.ecsv")

tex_path = os.path.join(output_path, "tex")
os.makedirs(tex_path, exist_ok=True)


dropbox_path = None
dropbox_figs = None #os.path.join(dropbox_path, "figures")


def write_google_sheet(
        doc_id: str,
        sheet_name: str,
        output_file: str,
):
    sheet_url = f"https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    sheet_str = str(urllib.request.urlopen(sheet_url).read(), 'utf-8')
    with open(output_file, "w") as file:
        file.write(sheet_str)
    return table.QTable.read(output_file)


def update_csv_from_sheet():
    global frb_table_gs
    global frb_nohost_table_gs
    sheet_id = "1Pib9RBHuI8i4Wd6B611xXqUbFx7hOzqfn9g6_y8qZNM"
    frb_table_gs = write_google_sheet(
        doc_id=sheet_id,
        sheet_name="Data",
        output_file=host_csv_path
    )
    frb_table_gs["host_identified"] = [True] * len(frb_table_gs)

    frb_nohost_table_gs = write_google_sheet(
        doc_id=sheet_id,
        sheet_name="UndetectedHosts",
        output_file=undetected_csv_path
    )
    frb_nohost_table_gs["host_identified"] = [False] * len(frb_nohost_table_gs)

    frb_table_gs = table.vstack([frb_table_gs, frb_nohost_table_gs])

    write_google_sheet(
        doc_id=sheet_id,
        sheet_name="Bibliography",
        output_file=bib_path
    )
    load_bib_dict()
    return frb_table_gs


def isreal(value):
    return not np.ma.is_masked(value) and value != 0


def split_err(err: str):
    if not isreal(err):
        return 0., 0.
    if isinstance(err, str) and "+" in err:
        err_1, err_2 = err.split(" ")
        if "+" in err_1:
            err_plus = err_1
            if "-" in err_2:
                err_minus = err_2
            else:
                raise ValueError(f"Uncertainty '{err}' not understood; no minus sign found.")
        elif "-" in err_1:
            err_minus = err_1
            if "+" in err_2:
                err_plus = err_2
            else:
                raise ValueError(f"Uncertainty '{err}' not understood; no plus sign found.")
        else:
            raise ValueError(f"Uncertainty '{err}' not understood; no sign found in first component {err_1}")

        err_minus = float(err_minus.replace("-", ""))
        err_plus = float(err_plus.replace("+", ""))
        return err_plus, err_minus
    else:
        return float(err), float(err)


def split_err_column(column):
    plus = []
    minus = []
    for r in column:
        err_plus, err_minus = split_err(r)
        plus.append(err_plus)
        minus.append(err_minus)
    return plus, minus


def generate_table(flds: list = None, quick: bool = False):
    if flds is None:
        field_list = field.list_fields()
    else:
        field_list = flds

    if not frb_table_gs:
        update_csv_from_sheet()
    global frb_table
    frb_table = table.QTable()
    frb_table["name"] = frb_table_gs["FRB"]
    frb_table["repeater"] = frb_table_gs["Repeater?"]
    frb_table["dm_named"] = frb_table_gs[
                                "Dispersion Measure (named burst for repeaters) S/N-maximised DM pc cm-3"
                            ] * dm_units
    dm_err_plus, dm_err_minus = split_err_column(frb_table_gs["DM uncertainty pc cm-3"])
    frb_table["dm_named_err+"] = dm_err_plus * dm_units
    frb_table["dm_named_err-"] = dm_err_minus * dm_units
    frb_table["ref_dm_named"] = frb_table_gs["DM reference"]

    frb_table["dm_struct"] = frb_table_gs["Structure-maximised DM_struct pc cm-3"] * dm_units
    frb_table["dm_struct_err"] = frb_table_gs["DM_struct uncertainty pc cm-3"] * dm_units
    frb_table["ref_dm_struct"] = frb_table_gs["DM_struct reference"]

    frb_table["dm_repeater"] = frb_table_gs[
                                   "Repeater Dispersion Measure DM_repeater pc cm-3"] * dm_units
    frb_table["dm_repeater_err"] = frb_table_gs["DM_repeater uncertainty pc cm-3"] * dm_units
    frb_table["ref_dm_repeater"] = frb_table_gs["DM_repeater reference"]

    frb_table["snr_incoherent"] = frb_table_gs["Burst Signal-to-noise (S/N)  Incoherent detection S/N incoherent"]
    frb_table["reference_snr_incoherent"] = frb_table_gs['S/N incoherent reference']
    frb_table["snr_coherent"] = frb_table_gs["Coherent detection S/N coherent"]
    frb_table["reference_snr_coherent"] = frb_table_gs['S/N coherent reference']

    frb_table["tau"] = frb_table_gs["Scattering time τ ms"] * units.ms
    frb_table["tau_err"] = frb_table_gs["τ uncertainty ms"] * units.ms
    frb_table["sigma_tau"] = frb_table_gs[
        "FRB Temporal properties (named burst for repeaters) Intrinsic width (first pulse if multiple components) σ ms"]
    frb_table["sigma_tau_err"] = frb_table_gs["σ uncertainty ms"]
    frb_table["nu_tau"] = frb_table_gs["Measurement frequency ν_τ GHz"] * units.GHz
    frb_table["ref_tau"] = frb_table_gs["τ reference"]
    frb_table["width"] = frb_table_gs["Total width w ms"]
    frb_table["width_err"] = frb_table_gs['w uncertainty ms']
    frb_table["ref_width"] = frb_table_gs["w reference"]

    frb_table["z"] = frb_table_gs["Source distance Host redshift z"]
    # frb_table["z"][np.isnan(frb_table["z"])] = -999.
    frb_table["z_err"] = frb_table_gs["z uncertainty"]
    frb_table["ref_z"] = frb_table_gs["z reference"]

    frb_table["distance"] = frb_table_gs["Non-cosmological distance D kpc"] * units.kpc
    frb_table["distance_err"] = frb_table_gs["D uncertainty kpc"] * units.kpc
    frb_table["ref_distance"] = frb_table_gs["D reference"]

    frb_table["ra"] = frb_table_gs[
        "FRB source localisation (most precise for repeating sources) Right Ascension RA J2000"]
    frb_table["ra_err"] = frb_table_gs['RA reported uncertainty " (or s if noted)'] * units.arcsec
    frb_table["ra_err_stat"] = frb_table_gs['RA statistical uncertainty "']
    frb_table["ra_err_sys"] = frb_table_gs['RA systematic uncertainty "']

    frb_table["dec"] = frb_table_gs["Declination Dec J2000"]
    frb_table["dec_err"] = frb_table_gs['Dec reported uncertainty "'] * units.arcsec
    frb_table["dec_err_stat"] = frb_table_gs['Dec statistical uncertainty "']
    frb_table["dec_err_sys"] = frb_table_gs['Dec systematic uncertainty "']

    frb_table["a"] = frb_table_gs['Uncertainty ellipse a "'] * units.arcsec
    frb_table["b"] = frb_table_gs['b "'] * units.arcsec
    frb_table["theta"] = frb_table_gs['θ deg E of N'] * units.deg

    frb_table["ref_position"] = frb_table_gs["Localisation reference"]
    frb_table["team"] = frb_table_gs["Localisation team"]
    frb_table["telescope"] = frb_table_gs["FRB detection telescope (named burst) Detection telescope"]
    frb_table["telescope_localisation"] = frb_table_gs["Localisation telescope"]
    frb_table["survey"] = frb_table_gs["Detection Survey"]
    frb_table["ref_telescope"] = frb_table_gs["Detection telescope reference"]
    frb_table["host_identified"] = frb_table_gs["host_identified"]
    frb_table["in_flimflam_dr1"] = frb_table_gs["Khrykin+2024 sample"]
    frb_table["vlt_imaged"] = frb_table_gs["VLT imaging (CRAFT only)"]
    frb_table["host_published"] = frb_table_gs["Host first published (peer-reviewed) Paper"]

    frb_table["mass_stellar"] = frb_table_gs["Host mass Stellar mass M* M☉"]
    frb_table["mass_stellar_err"] = frb_table_gs["M* uncertainty M☉"]
    frb_table["log_mass_stellar"] = frb_table_gs["log10(M*/M☉)"]
    frb_table["log_mass_stellar_err"] = frb_table_gs["log10(M*/M☉) uncertainty"]

    frb_table.sort("name")

    coords = []
    frbs = []
    dms_best = []
    dms_best_err = []
    dms_best_ref = []
    dms_best_type = []
    vlt_imaged = []
    host_published = []
    repeater = []
    flimflam = []
    ra_err_proj = []
    dec_err_proj = []
    a_proj = []
    b_proj = []
    lookback = []
    age = []

    n_noloc = 0
    n_loc = 0

    for i, row in enumerate(frb_table):
        imagd = row["vlt_imaged"]
        if imagd == "Y":
            vlt_imaged.append(True)
        else:
            vlt_imaged.append(False)

        published = row["host_published"]
        if published in ("Unpublished", "N/A"):
            host_published.append(False)
        else:
            host_published.append(True)

        repeats = row["repeater"]
        if repeats == "Y":
            repeater.append(True)
        else:
            repeater.append(False)

        flimflamd = row["in_flimflam_dr1"]
        if flimflamd == "Y":
            flimflam.append(True)
        else:
            flimflam.append(False)

        name = row["name"]
        fld_name = "FRB" + str(name)

        # if flds and fld_name not in flds:
        #     continue

        ra = row["ra"]
        dec = row["dec"]
        z = -999.
        # print(row["name"], isreal(row["z"]), row["z"])
        if isreal(row["z"]):
            z = row["z"]
        else:
            row["z"] = -999
        z_err = -999.
        if isreal(row["z_err"]):
            z_err = row["z_err"]

        # print("\t", z)
        # print("\t", ra, dec)

        if not isreal(ra) or not isreal(dec):
            coord_this = coord.SkyCoord(0, 0, unit="deg")
            n_noloc += 1
        else:
            coord_this = coord.SkyCoord(ra, dec)
            n_loc += 1
        coords.append(coord_this)

        # Find the best total FRB DM
        dm_named = row["dm_named"]
        dm_named_err = np.max((row["dm_named_err+"].value, row["dm_named_err-"].value)) * dm_units
        dm_struct = row["dm_struct"]
        dm_repeater = row["dm_repeater"]
        if isreal(dm_repeater):
            dm = dm_repeater
            dm_err = row["dm_repeater_err"]
            dm_ref = row["ref_dm_repeater"]
            dm_best_type = "dm_repeater"
        elif isreal(dm_struct) and row["dm_struct_err"] < dm_named_err:
            dm = dm_struct
            dm_err = row["dm_struct_err"]
            dm_ref = row["ref_dm_struct"]
            dm_best_type = "dm_struct"
        else:
            dm = dm_named
            dm_err = dm_named_err
            dm_ref = row["ref_dm_named"]
            dm_best_type = "dm_named"
        dms_best.append(dm)
        dms_best_err.append(dm_err)
        dms_best_ref.append(dm_ref)
        dms_best_type.append(dm_best_type)

        tau = 0 * units.ms
        nu_tau = 0 * units.GHz
        if isreal(row["tau"]) and isreal(row["nu_tau"]):
            tau = row["tau"]
            nu_tau = row["nu_tau"]

        # Set up FRB object
        frb = FRB(
            dm=dm,
            name=name,
            position=coord_this,
            # z=z,
            # z_err=z_err,
            tau=tau,
            nu_scattering=nu_tau
        )
        frb.get_host()

        frbs.append(frb)

        if isreal(row["distance"]):
            frb.host_galaxy.D_A = row["distance"]
            frb.host_galaxy.D_comoving = row["distance"]
            frb.host_galaxy.D_L = row["distance"]
        else:
            frb.host_galaxy.set_z(z)

        # Project positional uncertainties
        if z > 0:
            ra_err_proj.append(frb.host_galaxy.projected_size(row["ra_err"]))
            dec_err_proj.append(frb.host_galaxy.projected_size(row["dec_err"]))
            a_proj.append(frb.host_galaxy.projected_size(row["a"]))
            b_proj.append(frb.host_galaxy.projected_size(row["b"]))
            lookback.append(Planck18.lookback_time(z))
            age.append(Planck18.age(z))
        else:
            ra_err_proj.append(-999 * units.kpc)
            dec_err_proj.append(-999 * units.kpc)
            a_proj.append(-999 * units.kpc)
            b_proj.append(-999 * units.kpc)
            lookback.append(-999 * units.Gyr)
            age.append(-999 * units.Gyr)

        # Update our field param files from the table
        print(f"{fld_name} in field_list:", fld_name in field_list)
        do_field = fld_name in field_list
        if not do_field and fld_name[:-1] in field_list:
            fld_name = fld_name[:-1]
            do_field = True

        if do_field:
            print("\tUpdating param files:")
            field_yaml_path = str(os.path.join(p.param_dir, "fields", fld_name, f"{fld_name}.yaml"))
            print(f"\t\t{field_yaml_path}")
            field_yaml = p.load_params(field_yaml_path)
            frb_yaml_path = str(os.path.join(p.param_dir, "fields", fld_name, "objects", f"{fld_name}.yaml"))
            print(f"\t\t{frb_yaml_path}")
            if os.path.isfile(frb_yaml_path):
                frb_yaml = p.load_params(frb_yaml_path)

                if isreal(ra):
                    if "ra" in frb_yaml["position"]:
                        ra_old = frb_yaml["position"].pop("ra")["hms"]
                    else:
                        ra_old = frb_yaml["position"]["alpha"]["hms"]
                    print(f"\t\tUpdating Right Ascension from {ra_old} to {ra}")
                    frb_yaml["position"]["alpha"] = {
                        "hms": str(ra),
                        "decimal": coord_this.ra.value
                    }
                if isreal(dec):
                    if "dec" in frb_yaml["position"]:
                        dec_old = frb_yaml["position"].pop("dec")["dms"]
                    else:
                        dec_old = frb_yaml["position"]["delta"]["dms"]
                    print(f"\t\tUpdating Declination from {dec_old} to {dec}")
                    frb_yaml["position"]["delta"] = {
                        "dms": str(dec),
                        "decimal": coord_this.dec.value
                    }

                yaml_pos_err = frb_yaml["position_err"]
                if "ra" in yaml_pos_err:
                    yaml_pos_err["alpha"] = yaml_pos_err.pop("ra")
                if "alpha" not in yaml_pos_err:
                    yaml_pos_err["alpha"] = {}
                if isreal(row["ra_err"]):
                    print(f"\t\tUpdating RA error to {row['ra_err']}")
                    yaml_pos_err["alpha"]["total"] = row["ra_err"]
                if isreal(row["ra_err_stat"]):
                    print(f"\t\tUpdating RA error stat to {row['ra_err_stat']}")
                    yaml_pos_err["alpha"]["stat"] = row["ra_err_stat"]
                if isreal(row["ra_err_sys"]):
                    print(f"\t\tUpdating RA error sys to {row['ra_err_sys']}")
                    yaml_pos_err["alpha"]["sys"] = row["ra_err_sys"]

                if "dec" in yaml_pos_err:
                    yaml_pos_err["delta"] = yaml_pos_err.pop("dec")
                if "delta" not in yaml_pos_err:
                    yaml_pos_err["delta"] = {}
                if isreal(row["dec_err"]):
                    print(f"\t\tUpdating Dec error to {row['dec_err']}")
                    yaml_pos_err["delta"]["total"] = row["dec_err"]
                if isreal(row["dec_err_stat"]):
                    print(f"\t\tUpdating Dec error stat to {row['dec_err_stat']}")
                    yaml_pos_err["delta"]["stat"] = row["dec_err_stat"]
                if isreal(row["dec_err_sys"]):
                    print(f"\t\tUpdating Dec error sys to {row['dec_err_sys']}")
                    yaml_pos_err["delta"]["sys"] = row["dec_err_sys"]

                if isreal(row["a"]):
                    print(f"\t\tUpdating uncertainty a to {row['a']}")
                    if "a" not in yaml_pos_err:
                        yaml_pos_err["a"] = {}
                    yaml_pos_err["a"]["total"] = row["a"]
                if isreal(row["b"]):
                    print(f"\t\tUpdating uncertainty b to {row['b']}")
                    if "b" not in yaml_pos_err:
                        yaml_pos_err["b"] = {}
                    yaml_pos_err["b"]["total"] = row["b"]
                if isreal(row["theta"]):
                    print(f"\t\tUpdating uncertainty theta to {row['theta']}")
                    yaml_pos_err["theta"] = row["theta"]

                if row["host_identified"]:
                    host_name = frb_yaml["host_galaxy"]
                    host_yaml_path = str(os.path.join(p.param_dir, "fields", fld_name, "objects", f"{host_name}.yaml"))
                    host_yaml = p.load_params(host_yaml_path)
                    if not host_yaml:
                        host_yaml = FRB.default_host_params(
                            frb_name=fld_name,
                            position=frb.position
                        )
                    if isreal(z) and z > -990:
                        print(f"\t\tUpdating z to {z}")
                        host_yaml["z"] = z
                    if isreal(z_err):
                        print(f"\t\tUpdating z_err to {z_err}")
                        host_yaml["z_err"] = z_err
                    p.save_params(host_yaml_path, host_yaml)
                if isreal(dm):
                    print(f"\t\tUpdating dm from to {dm}")
                    frb_yaml["dm"] = dm
                if isreal(dm_err):
                    print(f"\t\tUpdating dm_err from to {dm_err}")
                    frb_yaml["dm_err"] = dm_err
                if isreal(tau):
                    print(f"\t\tUpdating tau from to {tau}")
                    frb_yaml["tau"] = tau
                if isreal(row["tau_err"]):
                    print(f"\t\tUpdating tau_err from to {row['tau_err']}")
                    frb_yaml["tau_err"] = row["tau_err"]
                if isreal(nu_tau):
                    print(f"\t\tUpdating nu_tau to {nu_tau}")
                    frb_yaml["nu_scattering"] = nu_tau
                if isreal(row["sigma_tau"]):
                    print(f"\t\tUpdating sigma_tau to {row['sigma_tau']}")
                    frb_yaml["width_int"] = row["sigma_tau"]
                if isreal(row["sigma_tau_err"]):
                    print(f"\t\tUpdating sigma_tau_err to {row['sigma_tau_err']}")
                    frb_yaml["width_int_err"] = row["sigma_tau_err"]
                if isreal(row["width"]):
                    print(f"\t\tUpdating width_total to {row['width']}")
                    frb_yaml["width_total"] = row["width"]
                if isreal(row["width_err"]):
                    print(f"\t\tUpdating width_total_err to {row['width_err']}")
                    frb_yaml["width_total_err"] = row["width_err"]
                if isreal(row["snr_incoherent"]):
                    print(f"\t\tUpdating snr to {row['snr_incoherent']}")
                    frb_yaml["snr"] = row["snr_incoherent"]
                if isreal(row["snr_coherent"]):
                    print(f"\t\tUpdating snr_coherent to {row['snr_coherent']}")
                    frb_yaml["snr_coherent"] = row["snr_coherent"]
                if isreal(row["telescope"]):
                    print(f"\t\tUpdating telescope to {row['telescope']}")
                    frb_yaml["instrument"] = str(row["telescope"])
                if isreal(row["survey"]):
                    print(f"\t\tUpdating survey to {row['survey']}")
                    frb_yaml["survey"] = str(row["survey"])
                p.save_params(frb_yaml_path, frb_yaml)
                p.save_params(field_yaml_path, field_yaml)

    frb_table["repeater"] = repeater
    frb_table["dm"] = dms_best
    frb_table["dm_err"] = dms_best_err
    frb_table["ref_dm"] = dms_best_ref
    frb_table["type_dm"] = dms_best_type
    frb_table["vlt_imaged"] = vlt_imaged
    frb_table["host_published"] = host_published
    frb_table["in_flimflam_dr1"] = flimflam

    frb_table["ra_err_proj"] = ra_err_proj
    frb_table["dec_err_proj"] = dec_err_proj
    frb_table["a_proj"] = a_proj
    frb_table["b_proj"] = b_proj

    # SCATTERING TIMESCALES
    # ==============================================================
    # MW scattering contribution, according to Cordes et al 2022
    frb_table["tau_ism_cordes"], frb_table["tau_ism_cordes_err"] = zip(*[f.tau_mw() for f in frbs])
    # NE2001, from PyGEDM
    frb_table["tau_ism_ne2001"] = [f._tau_mw_ism_ne2001 for f in frbs]
    # YMW16, from PyGEDM
    frb_table["tau_ism_ymw16"] = [f._tau_mw_ism_ymw16 for f in frbs]
    # COORDINATES
    # ==============================================================
    frb_table["coord"] = coords
    # Galactic coordinates
    frb_table["galactic"] = frb_table["coord"].galactic
    frb_table["galactic_l"] = frb_table["galactic"].l
    frb_table["galactic_b"] = frb_table["galactic"].b

    # DISTANCES
    # ==============================================================
    frb_table["distance_comoving"] = list(map(lambda f: f.host_galaxy.D_comoving, frbs))
    frb_table["distance_ang"] = list(map(lambda f: f.host_galaxy.D_A, frbs))
    frb_table["distance_lum"] = list(map(lambda f: f.host_galaxy.D_L, frbs))

    frb_table["lookback_time"] = lookback
    frb_table["universe_age"] = age

    frb_table["frb_object"] = frbs

    def cosmic(f):
        z_ = f.host_galaxy.z
        if z_ is not None and z_ > 0:
            return f.dm_cosmic()
        else:
            return -999 * dm_units

    def excess(r):
        if r["z"] > 0:
            return r["dm_exgal"] - r["dm_cosmic_avg"]
        else:
            return -999 * dm_units

    def rest(r, col):
        if r["z"] > 0:
            return r[col] * (1 + r["z"])
        else:
            return -999 * dm_units

    if not quick:

        # MASSES and HALOS
        # ==============================================================

        mass_stellar = []
        mass_stellar_err = []
        mass_stellar_plus = []
        mass_stellar_minus = []
        log_mass_stellar = []
        log_mass_stellar_err = []
        log_mass_stellar_plus = []
        log_mass_stellar_minus = []
        mass_halo = []
        log_mass_halo = []
        host_objects = []
        dm_halo = []

        print("Calculating DM_host_halo")
        for row in frb_table:
            host = row["frb_object"].host_galaxy
            host_objects.append(host)
            if isreal(row["mass_stellar"]):
                m_star = float(row["mass_stellar"]) * units.solMass
                m_star_plus, m_star_minus = split_err(row["mass_stellar_err"])
                m_star_plus *= units.solMass
                m_star_minus *= units.solMass

                log_m_star = np.log10(m_star.value)
                print(log_m_star, m_star)
                print(u.uncertainty_log10(arg=m_star, uncertainty_arg=m_star_minus))
                log_m_star_minus = float(u.uncertainty_log10(arg=m_star, uncertainty_arg=m_star_minus))
                log_m_star_plus = float(u.uncertainty_log10(arg=m_star, uncertainty_arg=m_star_plus))

            elif isreal(row["log_mass_stellar"]):
                log_m_star = float(row["log_mass_stellar"])
                log_m_star_plus, log_m_star_minus = split_err(row["log_mass_stellar_err"])

                m_star = units.solMass * 10 ** log_m_star
                m_star_plus = u.uncertainty_power(x=10., power=log_m_star, sigma_x=log_m_star_plus) * units.solMass
                m_star_minus = u.uncertainty_power(x=10., power=log_m_star, sigma_x=log_m_star_minus) * units.solMass

            else:
                m_star = -999. * units.solMass
                m_star_plus = -999. * units.solMass
                m_star_minus = -999. * units.solMass

                log_m_star = -999.
                log_m_star_minus = -999.
                log_m_star_plus = -999.

            m_star_err = max([m_star_plus, m_star_minus])
            log_m_star_err = max([log_m_star_plus, log_m_star_minus])

            mass_stellar.append(m_star)
            mass_stellar_plus.append(m_star_plus)
            mass_stellar_minus.append(m_star_minus)
            mass_stellar_err.append(m_star_err)

            log_mass_stellar.append(log_m_star)
            log_mass_stellar_minus.append(log_m_star_minus)
            log_mass_stellar_plus.append(log_m_star_plus)
            log_mass_stellar_err.append(log_m_star_err)

            if isreal(log_m_star) and log_m_star > 0.:
                host.log_mass_stellar = log_m_star
                # print(row)
                # print(row["name"], row["z"], row["log_mass_stellar"], row["log_mass_stellar_err"], row["mass_stellar"], row["mass_stellar_err"])
                # print("\t", host.name, host.mass_stellar, log_m_star, host.z)
                m_halo, log_m_halo = host.halo_mass()
                # print("\t", host.log_mass_halo, host.halo_concentration_parameter())
                mnfw = host.halo_model_mnfw()
                dm_h = mnfw.Ne_Rperp(
                    Rperp=1. * units.kpc,
                    rmax=1.,
                    step_size=0.1 * units.kpc
                ) / (2 * (1 + host.z))
            else:
                log_m_halo = -999.
                m_halo = -999. * units.solMass
                dm_h = -999. * dm_units

            log_mass_halo.append(log_m_halo)
            mass_halo.append(m_halo)
            dm_halo.append(dm_h)

        frb_table["mass_stellar"] = mass_stellar
        frb_table["mass_stellar_err"] = mass_stellar_err
        frb_table["mass_stellar_err+"] = mass_stellar_plus
        frb_table["mass_stellar_err-"] = mass_stellar_minus
        frb_table["log_mass_stellar"] = log_mass_stellar
        frb_table["log_mass_stellar_err"] = log_mass_stellar_err
        frb_table["log_mass_stellar_err+"] = log_mass_stellar_plus
        frb_table["log_mass_stellar_err-"] = log_mass_stellar_minus
        frb_table["mass_halo"] = mass_halo
        frb_table["log_mass_halo"] = log_mass_halo
        frb_table["dm_host_halo"] = dm_halo
        frb_table["host_object"] = host_objects

        # MILKY WAY ISM DMs
        # ==============================================================
        # NE2001, from the Bar-Or+Prochaska implementation
        print("Calculating DM_ISM_NE2001 (Bar-Or+Prochaska)")
        frb_table["dm_ism_ne2001_baror"] = list(
            map(lambda f: f.dm_mw_ism_ne2001_baror(distance=f.host_galaxy.D_comoving), frbs))
        # NE2001, from PyGEDM
        print("Calculating DM_ISM_NE2001 (PyGEDM)")
        frb_table["dm_ism_ne2001"] = list(map(lambda f: f.dm_mw_ism_ne2001(distance=f.host_galaxy.D_comoving), frbs))
        print("Calculating DM_ISM_YMW16 (PyGEDM)")
        # YMW16, from PyGEDM
        frb_table["dm_ism_ymw16"] = list(map(lambda f: f.dm_mw_ism_ymw16(distance=f.host_galaxy.D_comoving), frbs))
        # Difference
        frb_table["dm_ism_delta"] = np.abs(frb_table["dm_ism_ne2001"] - frb_table["dm_ism_ymw16"])
        # MILKY WAY HALO DMs
        # ==============================================================
        # Prochaska & Zheng 2019 mNFW model
        print("Calculating DM_MW_HALO (PZ19)")
        frb_table["dm_mw_halo_pz19"] = list(map(lambda f: f.dm_mw_halo("pz19"), frbs))
        # Y. Faerman et al 2017 model
        # print("Calculating DM_MW_HALO (YF17)")
        # frb_table["dm_mw_halo_yf17"] = list(map(lambda f: f.dm_mw_halo("yf17"), frbs))
        # Miller & Bregman 2015 model
        # print("Calculating DM_MW_HALO (MB15)")
        # frb_table["dm_mw_halo_mb15"] = list(map(lambda f: f.dm_mw_halo("mb15"), frbs))
        # Total Milky Way DM
        frb_table["dm_mw"] = 40 * dm_units + frb_table["dm_ism_ne2001"]
        # Some attempt at uncertainty
        frb_table["dm_mw_err"] = frb_table["dm_ism_delta"]
        # HOST DM ESTIMATES
        # ==============================================================
        frb_table["dm_host_nominal"] = 50 * dm_units / (1 + frb_table["z"])
        # Host DM contribution from scattering, according to Cordes et al 2022 (AFG=1)
        frb_table["dm_host_tau_rest"] = list(map(lambda f: f.dm_host_from_tau(afg=0.1), frbs))
        # Ditto, in observer frame
        frb_table["dm_host_tau"] = frb_table["dm_host_tau_rest"] / (1 + frb_table["z"])
        # EXTRAGALACTIC AND COSMIC DM
        # ==============================================================
        # Extragalactic DM
        frb_table["dm_exgal"] = frb_table["dm"] - frb_table["dm_mw"]
        frb_table["dm_exgal_err"] = np.sqrt(frb_table["dm_err"] ** 2 + frb_table["dm_mw_err"] ** 2)
        # Expectation for cosmic DM contribution
        print("Calculating DM_cosmic_avg")
        frb_table["dm_cosmic_avg"] = [cosmic(f) for f in frbs]
        # Cosmic DM after subtracting a nominal host DM
        frb_table["dm_cosmic_nominal"] = frb_table["dm_exgal"] - frb_table["dm_host_nominal"]
        # Cosmic DM after subtracting the host DM modelled from scattering
        frb_table["dm_cosmic_tau"] = frb_table["dm_exgal"] - frb_table["dm_host_tau"]
        # Expectation for DM from halo intersections
        print("Calculating DM_halos_avg")
        frb_table["dm_halos_avg"] = list(map(lambda f: f.dm_halos_avg(), frbs))
        # Expectation for IGM DM (cosmic DM with halos DM subtracted)
        frb_table["dm_igm_avg"] = frb_table["dm_cosmic_avg"] - frb_table["dm_halos_avg"]
        # Excess DM, after subtracting modelled Milky Way and cosmic components
        frb_table["dm_excess"] = [excess(row) for row in frb_table]
        frb_table["dm_excess_err"] = frb_table["dm_exgal_err"]
        # Excess DM in the host galaxy rest frame
        frb_table["dm_excess_rest"] = [rest(row, "dm_excess") for row in frb_table]
        frb_table["dm_excess_rest_err"] = frb_table["dm_excess_err"] * (1 + frb_table["z"])
        # Residual DM after subtracting all modelled / estimated components
        frb_table["dm_residual"] = frb_table["dm_excess"] - frb_table["dm_host_nominal"]

        # DM RATIOS
        # ==============================================================
        frb_table["dm/distance"] = frb_table["dm"] / frb_table["distance_comoving"]
        frb_table["dm_exgal/distance"] = frb_table["dm_exgal"] / frb_table["distance_comoving"]
        frb_table["dm/z"] = frb_table["dm"] / frb_table["z"]
        frb_table["dm_exgal/z"] = frb_table["dm_exgal"] / frb_table["z"]
        frb_table["dm_cosmic_nominal/z"] = frb_table["dm_cosmic_nominal"] / frb_table["z"]
        # FRB object
        frb_table["frb_object"] = frbs

    if quick:
        frb_table.write(frb_path.replace(".ecsv", "_quick.ecsv"), overwrite=True)
        frb_table.write(frb_path.replace(".ecsv", "_quick.csv"), overwrite=True)
    else:
        frb_table_write = frb_table.copy()
        frb_table_write.remove_column("frb_object")
        frb_table_write.remove_column("host_object")

        u.detect_problem_column(frb_table_write)

        frb_table_write.write(frb_path, overwrite=True)
        frb_table_write.write(frb_path.replace(".ecsv", ".csv"), overwrite=True)

        # Placeholder Latex table
        frb_table_write.write(os.path.join(table_path, "frb_hosts.tex"), overwrite=True, format="ascii.latex")

    return frb_table


bib_dict = {}


def load_bib_dict(force: bool = False):
    if not bib_dict or force:
        bib_table = table.QTable.read(bib_path)
        for row in bib_table:
            bib_dict[row["Reference"]] = row["Mendeley Key"]
    return bib_dict


def get_ref(key: str):
    load_bib_dict()
    if isreal(key) and "+" in key:
        return bib_dict[key]
    else:
        return None


def build_ref_list(tbl: table.Table, key: str, new_key: str = None, replace: bool = False):
    row_list = []
    for row in tbl:
        ref = get_ref(row[key])
        if ref is not None and not np.ma.is_masked(ref):
            row_list.append(r"(\citenum{" + ref + "})")
        else:
            row_list.append(" ")
    if new_key is not None:
        tbl[new_key] = row_list
    if replace:
        tbl[key] = row_list
    return row_list


def check_refs():
    missing_refs = []
    load_bib_dict()
    load_frb_table()
    for colname in frb_table.colnames:
        if colname.startswith("ref_"):
            for entry in frb_table[colname]:
                if isreal(entry) and entry not in bib_dict:
                    missing_refs.append(entry)
    return set(missing_refs)


def load_frb_table(hosts_only=False, craft_only=False):
    global frb_table
    frb_table = table.QTable.read(frb_path)
    frb_table.sort("name")
    frbs = []
    for row in frb_table:
        z = None
        z_err = None
        if isreal(row["z"]) and row["z"] > -990:
            z = row["z"]
            z_err = row["z_err"]
        frb = FRB(
            dm=row["dm"],
            name=row["name"],
            position=row["coord"],
            z=z,
            z_err=z_err,
            tau=row["tau"],
            nu_scattering=row["nu_tau"]
        )
        frbs.append(frb)
    frb_table["frb_object"] = frbs

    frb_table_ = frb_table.copy()
    if hosts_only:
        frb_table_ = frb_table_[frb_table_["host_identified"]]
    if craft_only:
        frb_table_ = frb_table_[frb_table_["team"] == "CRAFT"]
    return frb_table_


optical_cat = None


def load_photometry_table(force: bool = False):
    global optical_cat
    if optical_cat is None or force:
        optical_cat = outp.OpticalCatalogue("optical")
    optical_cat.load_table(force=force)
    host_photometry = optical_cat.to_astropy()
    return host_photometry


def add_q_0(tbl, colname, default=0.2):
    tbl["q_0"] = default
    for row in tbl:
        if row[colname] < default:
            # row["q_0"] = 0.1
            # if row[colname] < 0.1:
            row["q_0"] = row[colname]


def odr_fit(x, y, x_err=None, y_err=None, f=None):
    if f is None:
        def line(B, x):
            '''Linear function y = m*x + b'''
            # B is a vector of the parameters.
            # x is an array of the current x values.
            # x is in the same format as the x passed to Data or RealData.
            #
            # Return an array in the same format as y passed to Data or RealData.
            return B[0] * x + B[1]

        f = line
    #     guess_slope = (y[-1] - y[0]) / (x[-1] - x[0])
    #     guess_intercept = (y[0] - guess_slope * x[0])
    linear = odr.Model(f)
    mydata = odr.RealData(x, y, sx=x_err, sy=y_err)
    myodr = odr.ODR(mydata, linear, beta0=[0., 0.])

    odr_out = myodr.run()

    N = len(x)
    # From https://stackoverflow.com/questions/21395328/how-to-estimate-goodness-of-fit-using-scipy-odr
    df = N - 2  # equivalently, df = odr_out.iwork[10]
    beta_0 = 0  # test if slope is significantly different from zero
    t_stat = (odr_out.beta[0] - beta_0) / odr_out.sd_beta[0]  # t statistic for the slope parameter
    p_val = stats.t.sf(np.abs(t_stat), df) * 2
    # print(
    #     'Recovered equation: y={:3.5f}x + {:3.2f}, t={:3.2f}, p={:.2e}'.format(odr_out.beta[0], odr_out.beta[1], t_stat,
    #                                                                            p_val))
    print("Solved parameters:", odr_out.beta)
    line_sp = f(odr_out.beta, x)
    return odr_out, line_sp


def sanitise_column(tbl, col_name):
    # tbl = tbl[[isreal(row[col_name]) for row in tbl]]
    tbl = tbl[tbl[col_name].value != -999]
    tbl = tbl[np.isfinite(tbl[col_name])]
    return tbl


def scatter_prop(
        col_x, col_y, tbl,
        ax=None, fig=None,
        fit=True, func=None,
        name_col="object_name",
        tbl_name: str = None,
        filename: Union[bool, str] = None,
        **kwargs
):
    tbl = tbl.copy()
    out_dict = {}
    for col in col_x, col_y:
        tbl = sanitise_column(tbl=tbl, col_name=col)
        out_dict[col] = print_stats(col, tbl)
    if ax is None or fig is None:
        fig, ax = plt.subplots()

    x = tbl[col_x].value
    y = tbl[col_y].value

    corr = stats.pearsonr(x, y)

    print("Correlation coefficient:", corr.statistic)
    print("Correlation p-value:", corr.pvalue)

    err_col_x = col_x + "_err"
    has_x_err = err_col_x in tbl.colnames
    if has_x_err:
        x_err = tbl[err_col_x]
        out_dict[err_col_x] = print_stats(err_col_x, tbl)
    else:
        x_err = None

    err_col_y = col_y + "_err"
    has_y_err = err_col_y in tbl.colnames
    if has_y_err:
        y_err = tbl[err_col_y]
        out_dict[err_col_y] = print_stats(err_col_y, tbl, name_col=name_col)
    else:
        y_err = None

    if fit:
        fit, line_sp = odr_fit(x=x, y=y, x_err=x_err, y_err=y_err, f=func)
        ax.plot(x, line_sp, c="red")

    ax.errorbar(
        tbl[col_x], tbl[col_y],
        xerr=x_err, yerr=y_err,
        linestyle="none",
        marker="x",
        color="black",
        **kwargs
    )

    ax.set_xlabel(nice_axis_label(colname=col_x, tbl=tbl))
    ax.set_ylabel(nice_axis_label(colname=col_y, tbl=tbl))

    fig.legend()

    if filename is not False:
        if filename is None:
            filename = f"odr_{col_y}-v-{col_x}"
        if tbl_name is not None:
            filename += f"_{tbl_name}"
        fig = plt.gcf()
        savefig(fig, filename=filename, subdir="correlations")
        out_dict.update(
            {
                "x": col_x,
                "y": col_y,
                "pearson_coeff": corr.statistic,
                "pearson_p": corr.pvalue,
                "fit_slope": fit.beta[0],
                "fit_intercept": fit.beta[1],
            }
        )
        p.save_params(
            os.path.join(output_path, "correlations", filename + ".yaml"),
            out_dict,
        )

    return fig, ax, fit


def lts_prop(
        col_x,
        col_y,
        tbl,
        col_x_err=None,
        col_y_err=None,
        tbl_name: str = None,
        name_col="object_name",
        filename: Union[bool, str] = None,
        clip: float = None,
        x_legend: float = 1.05,
        fontsize=13,
        command_lines: list = None,
        **kwargs
):
    fig = plt.gcf()
    plt.close(fig)
    plt.clf()
    plt.cla()
    tbl = tbl.copy()
    out_dict = {}
    if command_lines is None:
        command_lines = []
    if col_y_err is None:
        col_y_err = col_y + "_err"
    if not isinstance(col_x, list):
        col_x = [col_x]
    if not isinstance(col_x_err, list):
        if col_x_err is not None:
            col_x_err = [col_x_err]
        else:
            col_x_err = [None] * len(col_x)

    cols = col_x + [col_y]
    cols_err = col_x_err + [col_y_err]
    cols_err_ = []
    for i, col in enumerate(cols):
        err_col = cols_err[i]
        if err_col is None:
            err_col = col + "_err"
        cols_err_.append(err_col)
        tbl = sanitise_column(tbl=tbl, col_name=col)
        out_dict[col] = print_stats(col, tbl, name_col)
        out_dict[err_col] = print_stats(err_col, tbl, name_col)
    col_x_err = cols_err_[:-1]

    y = tbl[col_y].value
    x_cols = []
    sigx_cols = []
    fx = ""
    for i, col_ in enumerate(col_x):
        fx += col_ + "+"
        col_err_ = col_x_err[i]
        sigx_col = tbl[col_err_].value
        sigx_cols.append(sigx_col)
        x_col = tbl[col_].value
        x_cols.append(x_col)
    fx = fx[:-1]
    x = np.column_stack(x_cols)
    sigx = np.column_stack(sigx_cols)
    sigy = tbl[col_y_err].value
    pivot = np.round(np.median(x, 0), 1)
    print("Pivot:", np.median(x, 0), pivot)
    # fig, ax = plt.subplots()
    fitted = None
    if clip is None:
        clip = 2.6
    while fitted is None and clip > 0.:
        try:
            fitted = ltsfit(
                x0=x,
                y=y,
                sigx=sigx,
                sigy=sigy,
                pivot=pivot,
                legend=False,
                text=False,
                corr=False,
                **kwargs
            )
        except (ValueError, TypeError):
            clip -= 0.1
    if fitted is None:
        return None, None, None, None, command_lines
    ax = plt.gca()
    ax.legend(loc=(x_legend, 0), fontsize=fontsize)

    _, equstrings = y_from_lts(
        x_col=col_x,
        y_data_col=col_y,
        x_err_col=col_x_err,
        y_err_col=col_y_err,
        tbl=tbl,
        f=fitted
    )
    equstring = equstrings["equation_latex"]

    statstring = rf"""
    $N$ = {len(tbl)}
    Pearson:
    - $r = {np.round(fitted.pearsonr.statistic, 2)}$
    - $p = {np.round(fitted.pearsonr.pvalue, 3)}$
    Spearman:
    - $r = {np.round(fitted.spearmanr.correlation, 2)}$
    - $p = {np.round(fitted.spearmanr.pvalue, 3)}$
    """

    f"""Linear fit: 
    {equstring}"""

    print(equstrings["x_for_2D"])
    if len(col_x) == 1:
        ax.set_xlabel(nice_axis_label(col_x[0], tbl))
    else:
        ax.set_xlabel(equstrings["x_for_2D"])

    ax.text(
        x_legend, 1,
        statstring,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=fontsize,
    )
    ax.set_ylabel(nice_axis_label(col_y, tbl))
    fig = plt.gcf()
    fig.suptitle(f"{equstring}", fontsize=fontsize)

    if tbl_name is None:
        suffix = ""
    else:
        suffix = "_" + tbl_name
    cols_str = ""
    for i, col in enumerate(col_x):
        cols_str += col + "+"
    cols_str = cols_str[:-1]

    stats = {
        "x": col_x,
        "y": col_y,
        "pearson": {
            "r": float(fitted.pearsonr.statistic),
            "p": float(fitted.pearsonr.pvalue),
            "command_r": u.latex_command(
                command="Pearson" + f"{cols_str}{col_y}{suffix}",
                value=fitted.pearsonr.statistic.round(3)
            ),
            "command_p": u.latex_command(
                command="PPearson" + f"{cols_str}{col_y}{suffix}",
                value=fitted.pearsonr.pvalue.round(3)
            )
        },
        "spearman": {
            "r": float(fitted.spearmanr.correlation),
            "p": float(fitted.spearmanr.pvalue),
            "command_r": u.latex_command(
                command="Spearman" + f"{cols_str}{col_y}{suffix}",
                value=fitted.spearmanr.correlation.round(3)
            ),
            "command_p": u.latex_command(
                command="PSpearman" + f"{cols_str}{col_y}{suffix}",
                value=fitted.spearmanr.pvalue.round(3)
            ),
        },
        "p": pivot,
        "fit_slope": fitted.coef[1],
        "fit_intercept": fitted.coef[0],
        "table_name": tbl_name,
        "clip": clip,
        "equation": equstring
    }

    print(tbl_name, cols_str, col_y)

    print("Adding Latex command:", stats["pearson"]["command_r"])
    command_lines.append(stats["pearson"]["command_r"])
    print("Adding Latex command:", stats["pearson"]["command_p"])
    command_lines.append(stats["pearson"]["command_p"])
    print("Adding Latex command:", stats["spearman"]["command_r"])
    command_lines.append(stats["spearman"]["command_r"])
    print("Adding Latex command:", stats["spearman"]["command_p"])
    command_lines.append(stats["spearman"]["command_p"])

    out_dict["stats"] = stats

    if filename is not False:
        if filename is None:
            filename = f"lts_{col_y}-v-{fx}"
            if tbl_name is not None:
                filename += f"_{tbl_name}"
        savefig(fig, filename=filename, subdir="correlations")
        p.save_params(
            os.path.join(output_path, "correlations", filename + ".yaml"),
            out_dict,
        )
        if dropbox_figs is not None:
            p.save_params(
                os.path.join(dropbox_figs, "correlations", filename + ".yaml"),
                out_dict,
            )
    return fig, ax, fitted, stats, command_lines


def print_stats(col, tbl, name_col="object_name"):
    print("=" * 20)
    print(col)
    stat_dict = {
        "min": units.Quantity(np.nanmin(tbl[col])),
        "min_obj": str(tbl[name_col][np.argmin(tbl[col])]),
        "max": units.Quantity(np.nanmax(tbl[col])),
        "max_obj": str(tbl[name_col][np.argmax(tbl[col])]),
        "mean": units.Quantity(np.nanmean(tbl[col])),
        "median": units.Quantity(np.nanmedian(tbl[col])),
        "n": len(tbl)
    }
    print("N:", stat_dict["n"])
    print(f"Min {col}:", stat_dict["min"].round(3), "\t", stat_dict["min_obj"])
    print(f"Median {col}:", stat_dict["median"].round(3))
    print(f"Mean {col}:", stat_dict["median"].round(3))
    print(f"Max {col}:", stat_dict["max"].round(3), "\t", stat_dict["max_obj"])
    print("=" * 20)
    print()
    return stat_dict


def hist_prop(
        col, tbl,
        ax=None, fig=None,
        name_col="name",
        bins="auto",
        label=None,
        filename: Union[str, bool] = None,
        **kwargs
):
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    tbl = sanitise_column(tbl=tbl.copy(), col_name=col)
    ax.hist(tbl[col], bins=bins, label=label, **kwargs)
    # fig.suptitle(col)
    ax.set_xlabel(nice_axis_label(col, tbl))
    ax.set_ylabel("N")
    out_dict = print_stats(col, tbl, name_col=name_col)

    if filename is not False:
        if filename is None:
            filename = f"hist_{col}_{label}"

        savefig(fig, filename=filename, subdir="population_comparisons")

    return fig, ax, out_dict


def cdf_prop(
        col,
        tbl,
        ax=None, fig=None,
        log=True,
        name_col="object_name",
        label=None,
        label_legend=None,
        step: bool = True,
        clean_min=None,
        clean_max=None,
        filename: Union[str, bool] = None,
        mc_bound_samples: int = None,
        **kwargs
):
    if clean_min is not None:
        print(f"Removing {col} < {clean_min}")
        tbl = tbl[tbl[col] >= clean_min]
    if clean_max is not None:
        print(f"Removing {col} > {clean_max}")
        tbl = tbl[tbl[col] <= clean_max]

    if ax is None or fig is None:
        fig, ax = plt.subplots()
    if label is None:
        label = col
    if label_legend is None:
        label_legend = label + f": $N = {len(tbl)}$"
    tbl = tbl.copy()

    tbl = sanitise_column(tbl=tbl, col_name=col)
    data = tbl[col]
    n = len(data)
    # sort the data in ascending order
    x = np.sort(data)
    y = list(np.arange(n) / float(n))
    if isinstance(x, units.Quantity):
        x = x.value
    x = list(x) + [np.max(x)]
    if isinstance(y, units.Quantity):
        y = y.value
    y += [1.]
    if step:
        ax.step(x, y, label=label_legend, **kwargs)
        # ax.scatter(x[0], y[0], marker="x", zorder=-1)
        # ax.scatter(x[-2], y[-2], marker="x", zorder=-1)
    else:
        ax.plot(x, y, label=label_legend, **kwargs)
    if log:
        ax.set_xscale('log')

    if mc_bound_samples is not None:
        n_realisations = 10000
        for i in range(n_realisations):
            samples = np.random.randint(low=0, high=len(tbl), size=mc_bound_samples)
            # print(len(data), samples.shape)
            data_ = data[samples]
            n = mc_bound_samples
            # sort the data in ascending order
            x = np.sort(data_)
            y = list(np.arange(n) / float(n))
            if isinstance(x, units.Quantity):
                x = x.value
            x = list(x) + [np.max(x)]
            if isinstance(y, units.Quantity):
                y = y.value
            y += [1.]
            ax.step(x, y, label=label, **kwargs, alpha=0.01, zorder=-1)

    ax.set_ylim(0, 1.)
    ax.set_xlabel(nice_axis_label(col, tbl))
    ax.set_ylabel("Cumulative fraction")
    out_dict = print_stats(col, tbl, name_col=name_col)

    if filename is not False:
        if filename is None:
            filename = f"cdf_{col}_{label}"

        savefig(fig, filename=filename, subdir="population_comparisons")

    return fig, ax, tbl, out_dict


def nice_axis_label(colname, tbl):
    label = nice_var(colname)
    unit = tbl[colname].unit
    if unit not in (None, ""):
        label += r" [" + nice_unit(unit) + "]"
    return label


def nice_unit(unit):
    if isinstance(unit, units.Quantity):
        unit = unit.unit
    u_dict = {
        dm_units: r"$\mathrm{pc\ cm^{-3}}$",
        units.ms: r"$\mathrm{ms}$"
    }
    if unit in u_dict:
        unit = u_dict[unit]
    else:
        unit = unit.to_string(format='latex')
    return unit


nice_var_dict = {
    "dm": r"$\mathrm{DM_{FRB}}$",
    "dm_cosmic_avg": r"$\tavg{\DMCosmic}$",
    "dm_inclination_corrected": r"$\mathrm{DM^{corrected}_{exgal}}$",
    "dm_ism_delta": r"$\Delta\mathrm{DM_{MW,ISM}}$",
    "dm_ism_ne2001": r"$\mathrm{DM_{NE2001}}$",
    "dm_ism_ymw16": r"$\mathrm{DM_{YMW16}}$",
    "dm_exgal": r"$\mathrm{DM_{exgal}}$",
    "dm_excess": r"$\mathrm{DM_{excess}}$",
    "dm_excess_rest": r"$\mathrm{DM^\prime_{excess}}$",
    "dm_excess_halo_rest": r"$\mathrm{DM^\prime_{excess} - DM^{halo\prime}_{host}}$",
    "dm_excess_rest_ymw16": r"$\mathrm{DM^{YMW16\prime}_{excess}}$",
    "dm_residual_flimflam": r"$\mathrm{DM^{FLIMFLAM}_{excess}}$",
    "dm_residual_flimflam_rest": r"$\mathrm{DM^{FLIMFLAM\prime}_{excess}}$",
    "g-I": r"$g-I$",
    "g-I_local": r"$g_\mathrm{local}-I_\mathrm{local}$",
    "galfit_axis_ratio": r"$b/a$",
    "galfit_band": "Band",
    "galfit_inclination": "$i$",
    "galfit_cos_inclination": r"$\cos(i)$",
    "galfit_1-cos_inclination": r"$\frac{1}{\cos(i)}$",
    "galfit_mag": r"$m_\mathrm{galfit}$",
    "galfit_n": "$n$",
    "galfit_offset": r"$\rho$",
    "galfit_offset_norm": r"$\rho / R_\mathrm{eff}$",
    "galfit_offset_proj": r"$\rho_\mathrm{proj}$",
    "galfit_offset_norm_proj": r"$\rho_\mathrm{proj} / R_\mathrm{eff}$",
    "galfit_offset_deproj": r"$\rho_\mathrm{deproj}$",
    "galfit_offset_norm_deproj": r"$\rho_\mathrm{deproj} / R_\mathrm{eff}$",
    "galfit_offset_disk": r"$\rho_\mathrm{disk}$",
    "galfit_r_eff": r"$R_\mathrm{eff}$",
    "galfit_r_eff_proj": r"$R^\mathrm{proj}_\mathrm{eff}$",
    "galfit_theta": r"$\theta_\mathrm{PA}$",
    "galfit_ra": r"$\alpha_\mathrm{host}$",
    "galfit_dec": r"$\delta_\mathrm{host}$",
    "name": "FRB",
    "name_legend": "FRB",
    "legend": "\phantom{S}",
    "tau": r"$\tau_\mathrm{FRB}$",
    "z": "$z$",
    "OFFSET_KPC": r"$\rho_\mathrm{proj}$",
    "OFFSET_KPC_PLANCK18": r"$\rho_\mathrm{proj}$",
    "Offset": r"$\rho / R_\mathrm{eff}$",
}


def nice_var(colname):
    if colname in nice_var_dict:
        return nice_var_dict[colname]
    else:
        return colname


def compare_cat(
        col_1, col_2,
        tbl_1, tbl_2,
        name_col_1="object_name", name_col_2="OBJNO",
        label_1=None, label_2=None,
        log: bool = False,
        log_and_linear: bool = False,
        clean_max=None,
        clean_min=None,
        alpha: float = 0.05,
        filename: Union[str, bool] = None,
        hist_kwargs: dict = {},
        cdf_kwargs: dict = {},
        legend: bool = True,
        fig: plt.Figure = None,
        ax_cdf: plt.Axes = None,
        ax_2: plt.Axes = None,
        c_1: str = "purple", c_2: str = "green",
        n_cols: int = None,
        n_rows: int = 1,
        n: int = 1,
        plot_1: bool = True,
        mc_bounds: bool = False,
        do_hist: bool = True,
        p_on_plot: bool = True,
        lw_1: float = None,
        lw_2: float = None,
        ls_1: str = None,
        ls_2: str = None
):
    print()
    print("=" * 50, "\n", "+" * 50)

    if log_and_linear:
        do_hist = False
        n_cols = 2
        log = True

    if fig is None:
        fig = plt.figure(figsize=[pl.textwidths["mqthesis"], pl.textwidths["mqthesis"] / 2])
    if n_cols is None:
        if do_hist:
            n_cols = 2
            n_cdf = n * 2
        else:
            n_cols = 2
            n_cdf = n * 2 - 1
    else:
        n_cdf = n_cols


    tbl_1 = tbl_1.copy()
    tbl_2 = tbl_2.copy()

    if clean_min is not None:
        print(f"Removing {col_1} < {clean_min}")
        tbl_1 = tbl_1[tbl_1[col_1] >= clean_min]
    if clean_min is not None:
        print(f"Removing {col_2} < {clean_min}")
        tbl_2 = tbl_2[tbl_2[col_2] >= clean_min]
    if clean_max is not None:
        print(f"Removing {col_1} > {clean_max}")
        tbl_1 = tbl_1[tbl_1[col_1] <= clean_max]
    if clean_max is not None:
        print(f"Removing {col_2} > {clean_max}")
        tbl_2 = tbl_2[tbl_2[col_2] <= clean_max]

    ks_result = stats.kstest(rvs=tbl_1[col_1].value, cdf=tbl_2[col_2].value)
    # n = len(tbl_1)
    # m = len(tbl_2)
    print("\nK-S Test:")
    print("\tstatistic:", ks_result.statistic)
    print("\tp:", ks_result.pvalue)
    # condition = np.sqrt(-np.log(alpha / 2) * (1 + (m / n)) / (2 * m))
    # print("Condition:", condition)
    if ks_result.pvalue < alpha:
        print(f"\tThe null hypothesis was rejected to p < {alpha} ({np.round(ks_result.pvalue, 3)})")
    else:
        print(f"\tThe null hypothesis could not be rejected to p < {alpha} ({np.round(ks_result.pvalue, 3)})")

    ad_result = stats.anderson_ksamp(samples=(tbl_1[col_1].value, tbl_2[col_2].value))
    print("\nA-D Test:")
    print("\tstatistic:", ad_result.statistic)
    print("\tp:", ad_result.significance_level)
    if ad_result.significance_level <= 0.001:
        p_str_ad = f"$p_\\textsc{{ad}} \leq 0.001$"
        print(f"\tThe null hypothesis was rejected to p < {alpha} (< 0.001)")
    elif ad_result.significance_level < alpha:
        p_str_ad = f"$p_\\textsc{{ad}}={np.round(ad_result.significance_level, 3)}$"
        print(f"\tThe null hypothesis was rejected to p < {alpha} ({np.round(ad_result.significance_level, 3)})")
    elif ad_result.significance_level < 0.25:
        p_str_ad = f"$p_\\textsc{{ad}}={np.round(ad_result.significance_level, 3)}$"
        print(
            f"\tThe null hypothesis could not be rejected to p < {alpha} ({np.round(ad_result.significance_level, 3)})")
    else:
        p_str_ad = f"$p_\\textsc{{ad}} \geq 0.25$"
        print(f"\tThe null hypothesis could not be rejected to p < {alpha} (> 0.25)")

    label_1_leg = label_1 + f": $N = {len(tbl_1)}$"
    label_2_leg = label_2 + f": {p_str_ad}; $p_\\textsc{{ks}}={ks_result.pvalue.round(3)}$; $N = {len(tbl_2)}$"

    if ax_cdf is None:
        ax_cdf = fig.add_subplot(n_rows, n_cols, n_cdf)
    if ax_2 is None and (do_hist or log_and_linear):
        ax_2 = fig.add_subplot(n_rows, n_cols, n * 2 - 1)

    mc_n = None
    if mc_bounds:
        mc_n = len(tbl_1)

    if plot_1:
        fig, ax_cdf, tbl_1, stats_1 = cdf_prop(
            col=col_1, tbl=tbl_1, name_col=name_col_1,
            ax=ax_cdf, fig=fig,
            log=log, label=label_1_leg,
            # clean_max=clean_max, clean_min=clean_min,
            filename=False,
            color=c_1,
            zorder=10,
            lw=lw_1,
            **cdf_kwargs
        )
        if do_hist:
            fig, ax_2, _ = hist_prop(
                col=col_1, tbl=tbl_1, name_col=name_col_1,
                ax=ax_2, fig=fig, alpha=0.5, label=label_1_leg,
                density=True,
                filename=False,
                color=c_1,
                **hist_kwargs
            )
        elif log_and_linear:
            fig, ax_2, tbl_1, stats_1 = cdf_prop(
                col=col_1, tbl=tbl_1, name_col=name_col_1,
                ax=ax_2, fig=fig,
                log=False, label=label_1_leg,
                # clean_max=clean_max, clean_min=clean_min,
                filename=False,
                color=c_1,
                zorder=10,
                lw=lw_1,
                **cdf_kwargs
            )
    else:
        stats_1 = print_stats(col_1, tbl_1, name_col_1)

    fig, ax_cdf, tbl_2, stats_2 = cdf_prop(
        col=col_2, tbl=tbl_2, name_col=name_col_2,
        ax=ax_cdf, fig=fig, log=log, label=label_2,
        # clean_max=clean_max, clean_min=clean_min,
        filename=False,
        color=c_2,
        mc_bound_samples=mc_n,
        label_legend=label_2_leg,
        lw=lw_2,
        ls=ls_2
    )
    if do_hist:
        fig, ax_2, _ = hist_prop(
            col=col_2, tbl=tbl_2, name_col=name_col_2,
            ax=ax_2, fig=fig, alpha=0.5, label=label_2, density=True,
            filename=False,
            color=c_2
        )
        ax_2.set_ylabel("Density")
        # ax_hist.yaxis.set_label_position("right")
        ax_2.set_xlabel(nice_axis_label(col_1, tbl_1))
    elif log_and_linear:
        fig, ax_2, tbl_2, stats_2 = cdf_prop(
            col=col_2, tbl=tbl_2, name_col=name_col_2,
            ax=ax_2, fig=fig,
            log=False, label=label_2,
            # clean_max=clean_max, clean_min=clean_min,
            filename=False,
            color=c_2,
            mc_bound_samples=mc_n,
            label_legend=label_2_leg,
            lw=lw_2,
            ls=ls_2
        )
        ax_2.set_ylabel("Cum. fraction")
        ax_2.set_xlabel(nice_axis_label(col_1, tbl_1))

    if do_hist:
        ax_cdf.set_ylabel("Cum. fraction")
        ax_cdf.yaxis.set_label_position("right")
        ax_cdf.yaxis.set_tick_params(
            labelright=True, right=True,
            labelleft=False, left=False
        )
    elif log_and_linear:
        ax_cdf.set_ylabel(" ")
        ax_cdf.yaxis.set_tick_params(
            labelright=False, right=False,
            labelleft=False, left=False
        )
    ax_cdf.set_xlabel(nice_axis_label(col_1, tbl_1))

    if legend:
        if do_hist:
            ax_2.legend(loc=(0, 1.1), fontsize=10)
        ax_cdf.legend(loc=(0, 1.1), fontsize=10)

    if p_on_plot:
        ax_cdf.text(
            0.95, 0.25, f"$p_\\textsc{{ks}}={ks_result.pvalue.round(3)}$",
            transform=ax_cdf.transAxes, horizontalalignment="right",
            verticalalignment="bottom",
        )
        ax_cdf.text(
            0.95, 0.05, p_str_ad,
            verticalalignment="bottom",
            transform=ax_cdf.transAxes, horizontalalignment="right",
        )

    if filename is None:
        filename = sanitise_filename(f"compare_{label_1}_{col_1}-v-{label_2}_{col_2}")

    out = {
        "col_1": col_1,
        "col_2": col_2,
        "label_1": label_1,
        "label_2": label_2,
        "K-S": {
            "p": ks_result.pvalue,
            "statistic": ks_result.statistic,
            "command": u.latex_command(
                command="KS" + f"{col_1}-{label_1}-v-{label_2}",
                value=ks_result.pvalue.round(3)
            )
        },
        "A-D": {
            "p": ad_result.significance_level,
            "statistic": ad_result.statistic,
            "command": u.latex_command(
                command="AD" + f"{col_1}-{label_1}-v-{label_2}",
                value=np.round(ad_result.significance_level, 3).round(3)
            )
        },
        "n_1": len(tbl_1),
        "n_1_command": u.latex_command(
            command="N" + f"{col_1}-{label_1}-v-{label_2}",
            value=len(tbl_1)
        ),
        "n_2": len(tbl_2),
        "n_2_command": u.latex_command(
            command="N" + f"{col_2}-{label_1}-v-{label_2}",
            value=len(tbl_2)
        ),
        col_1: stats_1,
        col_2: stats_2,
    }
    if filename is not False:
        fig.tight_layout()
        savefig(fig, filename=filename, subdir="population_comparisons")

        p.save_params(
            os.path.join(output_path, "population_comparisons", filename + ".yaml"),
            out,
        )
        if dropbox_figs is not None:
            p.save_params(
                os.path.join(dropbox_figs, "population_comparisons", filename + ".yaml"),
                out,
            )

    return out, fig, ax_2, ax_cdf


def sanitise_filename(filename):
    return filename.replace(
        r' \& ', '+'
    ).replace(
        '\&', '+'
    ).replace(
        ' ', '-'
    ).replace(
        '/', '-'
    ).replace(
        "$", ""
    ).replace(
        ",", ""
    ).replace(
        "<",
        ""
    ).replace(
        ">",
        ""
    )


from typing import List, AnyStr


def y_from_lts(
        x_col: Union[AnyStr, List[AnyStr]],
        y_data_col,
        tbl, f,
        x_err_col=None, y_err_col=None,
        n_grid: int = 1000
):
    if y_err_col is None:
        y_err_col = y_data_col + "_err"

    if not isinstance(x_col, list):
        x_col = [x_col]

    if x_err_col is None:
        x_err_col = [c + "_err" for c in x_col]

    from inspect import currentframe, getframeinfo

    y_data = tbl[y_data_col]
    y_data_err = tbl[y_err_col]

    if y_data.unit is None:
        y_un = 1.
        y_un_tex = ""
    else:
        y_un = y_data.unit
        y_un_tex = nice_unit(y_un).replace("$", "")

    y_data = u.dequantify(y_data)
    y_data_err = u.dequantify(y_data_err)

    a = f.coef[0]  # * y_un
    a_err = f.coef_err[0]  # * y_un

    bigstring = f"{a.round(2)}"
    y_tex = nice_var(y_data_col)
    y = a
    yy = np.zeros(n_grid) + a
    var = a_err ** 2
    xxs = []
    latex = ""
    latex_ = ""
    for i, col_ in enumerate(x_col):
        col_err = x_err_col[i]
        x = tbl[col_]
        x_err = tbl[col_err]
        if x.unit is None:
            x_un = 1.
            x_un_tex = ""
        else:
            x_un = x.unit
            x_un_tex = nice_unit(x_un).replace("$", "")

        x = dequantify(x)
        x_err = dequantify(x_err)

        x_tex = nice_var(col_).replace("$", "")

        xx = np.linspace(x.min(), x.max(), n_grid)
        xxs.append(xx)
        p_0 = np.round(np.median(x, 0), 1)

        b = f.coef[1 + i]  # * y_un / x_un
        y += (x - p_0) * b
        # yy += (xx - p_0) * b

        b_err = f.coef_err[1]  # * y_un / x_un
        xb_var = x ** 2 * b ** 2 * (x_err / x) ** 2 + (b_err / b) ** 2
        var += xb_var

        b_un = y_un / x_un
        b_un_tex = nice_unit(b_un).replace("$", "")

        if b < 0:
            sign = "-"

        else:
            sign = "+"

        b_tex = f"({b.round(1)}\pm{b_err.round(1)})"
        b_tex_ = f"{np.abs(b.round(1))}"

        string_x = f"{sign}{np.abs(b.round(2))} * ({col_} - {p_0})"

        latex_x = fr"+{b_tex}\ {b_un_tex}\times({x_tex}-{p_0})"
        latex_x_ = fr"{sign}{b_tex_}\ {b_un_tex}\times({x_tex}-{p_0})"

        bigstring += string_x
        latex += latex_x
        latex_ += latex_x_

    # bigstring = bigstring
    latex = fr"{y_tex}$= {latex[1:]} + ({a.round(1)}\pm{a_err.round(1)})\ {y_un_tex}$"
    latex_ = fr"${latex_} + {a.round(1)}\ {y_un_tex}$"

    y_err = np.sqrt(var)

    print(f)
    print(f"{y_data_col} = {bigstring}")
    print(latex)

    y *= y_un
    y_err *= y_un
    residuals = y_data * y_un - y
    residuals_err = np.sqrt((y_data_err * y_un) ** 2 + y_err ** 2)

    tbl[y_data_col + "_model"] = y
    tbl[y_data_col + "_model_err"] = y_err
    tbl[y_data_col + "_model_res"] = residuals
    tbl[y_data_col + "_model_res_err"] = residuals_err

    return yy, {
        "equation_latex": latex,
        "x_for_2D": latex_,
        "equation": bigstring
    }


def savefig(fig, filename, subdir=None, tight=True):
    output_this = output_path
    db_this = dropbox_figs
    if subdir is not None and db_this is not None:
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
    # fig.savefig(os.path.join(db_this, filename + ".pdf"), bbox_inches=bb)


def cut_to_band(tbl, fil_name, instrument="vlt-fors2"):
    tbl = tbl.copy()
    fil_name = fil_name.replace("_", "-")
    keep = has_band(tbl, fil_name, instrument)
    tbl = tbl[keep]
    return tbl

def has_band(tbl, fil_name, instrument="vlt-fors2"):
    fil_name = fil_name.replace("_", "-")
    has_band_ = [(row[f"mag_best_{instrument}_{fil_name}"] > 0 * units.mag) & (row[f"mag_best_{instrument}_{fil_name}_err"] > 0 * units.mag) for row in tbl]
    return has_band_


