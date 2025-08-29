#!/usr/bin/env python
# Code by Lachlan Marnoch, 2024

import os

import numpy as np
import matplotlib.pyplot as plt

import craftutils.utils as u
from craftutils.observation.objects import galaxy
import craftutils.plotting as pl

from frb.halos.models import halomass_from_stellarmass, stellarmass_from_halomass

import lib

description = """
Generates figures showing off the stellar-to-halo-mass relationships.
"""


def main(
        output_dir: str,
        input_dir: str
):
    lib.set_input_path(input_dir)
    lib.set_output_path(output_dir)

    pl.latex_setup()

    log_mhalos = np.linspace(9, 15.5, 100)
    fig = plt.figure(figsize=(0.9*lib.figwidth, 0.9*lib.figwidth))
    ax = fig.add_subplot(2, 2, 1)
    # for log_mhalo in log_mstars:


    log_mstars_b13 = galaxy.stellarmass_from_halomass_b13(log_mhalos)
    log_mstars_k18 = galaxy.stellarmass_from_halomass_b13(log_mhalos, **galaxy.params_k18)
    log_mstars_m13 = stellarmass_from_halomass(log_mhalos)
    ax.plot(log_mhalos, log_mstars_m13, color='red', label="Moster et al, 2013"),
    ax.plot(log_mhalos, log_mstars_b13, color='blue', label="Behroozi et al, 2013"),
    ax.plot(log_mhalos, log_mstars_k18, color='purple', label="Kravtsov et al, 2018"),
    # ax.legend(loc="upper left")

    ax.set_xlim(9, 15.5)
    ax.set_ylim(7, 13)
    ax.set_ylabel('$\log_{10}(M_\star / \mathrm{M}_{\odot})$', fontsize=14)
    ax.set_xlabel('$\log_{10}(M_\mathrm{halo} / \mathrm{M}_{\odot})$', fontsize=14)
    ax.tick_params(
        axis='both',
        labelsize=pl.tick_fontsize
    )

    # lib.savefig(fig, "halomass_to_stellarmass", subdir="smhm")

    log_mstars = np.linspace(7, 13, 100)
    # fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))

    ax = fig.add_subplot(2, 2, 2)

    # for log_mhalo in log_mhalos:
    log_mhalos_m13 = halomass_from_stellarmass(log_mstars)
    log_mhalos_b13 = galaxy.halomass_from_stellarmass_b13(log_mstars)
    log_mhalos_k18 = galaxy.halomass_from_stellarmass_b13(log_mstars, **galaxy.params_k18)
    ax.plot(log_mstars, log_mhalos_m13, color='red', label="Moster et al, 2013"),
    ax.plot(log_mstars, log_mhalos_b13, color='blue', label="Behroozi et al, 2013"),
    ax.plot(log_mstars, log_mhalos_k18, color='purple', label="Kravtsov et al, 2018"),
    # ax.legend(loc="upper left")

    ax.set_ylim(9, 15.5)
    ax.set_xlim(7, 13)
    ax.set_xlabel('$\log_{10}(M_\star / \mathrm{M}_{\odot})$', fontsize=14)
    ax.set_ylabel('$\log_{10}(M_\mathrm{halo} / \mathrm{M}_{\odot})$', fontsize=14)
    ax.tick_params(
        axis='both',
        labelsize=pl.tick_fontsize
    )

    # lib.savefig(fig, "stellarmass_to_halomass", subdir="smhm")




    log_mstar = 10.

    zs = np.linspace(0, 3, 100)
    # fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
    ax = fig.add_subplot(2, 2, 4)

    log_mhalos_m13 = []
    log_mhalos_b13 = []
    log_mhalos_k18 = []

    xi_0 = galaxy.params_b13["xi_0"]
    xi_a = galaxy.params_b13["xi_a"]

    for z in zs:
        log_mhalo_m13 = halomass_from_stellarmass(log_mstar, z=z)
        log_mhalo_b13 = galaxy.halomass_from_stellarmass_b13(log_mstar, z=z)
        log_mhalo_k18 = galaxy.halomass_from_stellarmass_b13(log_mstar, z=z, **galaxy.params_k18)

        log_mhalos_m13.append(log_mhalo_m13)
        log_mhalos_b13.append(log_mhalo_b13)
        log_mhalos_k18.append(log_mhalo_k18)

    ax.plot(zs, log_mhalos_m13, color='red', label="Moster et al, 2013")
    ax.plot(zs, log_mhalos_b13, color='blue', label="Behroozi et al, 2013")
    ax.plot(zs, log_mhalos_k18, color='purple', label="Kravtsov et al, 2018")

    # ax.legend(loc="upper left")

    # ax.legend(loc="upper left")

    # ax.set_ylim(9, 15.5)
    ax.set_xlim(0, np.max(zs))
    ax.set_xlabel('$z$', fontsize=14)
    ax.set_ylabel('$\log_{10}(M_\mathrm{halo} / \mathrm{M}_{\odot})$', fontsize=14)
    ax.tick_params(
        axis='both',
        labelsize=pl.tick_fontsize
    )

    # lib.savefig(fig, "stellarmass_to_halomass_z", subdir="smhm")




    log_mhalo = 12.

    # fig, ax = plt.subplots(figsize=(lib.figwidth, lib.figwidth))
    ax = fig.add_subplot(2, 2, 3)

    log_mstars_m13 = []
    log_mstars_b13 = []
    log_mstars_k18 = []

    scatters_k18 = []

    for z in zs:
        log_mstar_m13 = stellarmass_from_halomass(log_mhalo, z=z)
        log_mstar_b13 = galaxy.stellarmass_from_halomass_b13(log_mhalo, z=z)
        log_mstar_k18 = galaxy.stellarmass_from_halomass_b13(log_mhalo, z=z, **galaxy.params_k18)

        log_mstars_m13.append(log_mstar_m13)
        log_mstars_b13.append(log_mstar_b13)
        log_mstars_k18.append(log_mstar_k18)

        a = 1 / (1 + z)

        scatters_k18.append(xi_0 + xi_a * (a - 1))

    log_mstars_k18 = np.array(log_mstars_k18)
    scatters_k18 = np.array(scatters_k18)

    ax.plot(zs, log_mstars_m13, color='red', label="Moster et al, 2013"),
    ax.plot(zs, log_mstars_b13, color='blue', label="Behroozi et al, 2013"),
    ax.plot(zs, log_mstars_k18, color='purple', label="Kravtsov et al, 2018"),
    ax.plot(zs, log_mstars_k18 + scatters_k18, color='purple', label="Scatter in K+18", ls=":")
    ax.plot(zs, log_mstars_k18 - scatters_k18, color='purple', ls=":")
    # ax.legend(loc="upper left")

    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # ax.set_ylim(9, 15.5)
    ax.set_xlim(0, np.max(zs))
    ax.set_xlabel('$z$', fontsize=14)
    ax.set_ylabel('$\log_{10}(M_\star / \mathrm{M}_{\odot})$', fontsize=14)
    ax.tick_params(
        axis='both',
        labelsize=pl.tick_fontsize
    )

    fig.subplots_adjust(wspace=0.4, hspace=0.3)

    lib.savefig(fig, "shmr_panels", subdir="smhm", tight=True)

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
