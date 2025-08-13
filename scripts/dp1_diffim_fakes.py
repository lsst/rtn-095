# This file is part of rtn-095
#
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Script for generating plots that show the astrometry metrics.
# Original notebook written by Bruno Sanchez Waters. Converted to a python
# script by James Mullaney.

# Contact Authors: Bruno Sanchez Waters, James Mullaney

CAMPAIGN="DP1diffimWithFakes"
RUNVERSION="4full"
repo = '/repo/main'
instrument = 'LSSTComCam'
collection = f"u/bos/v29.0.1/LSSTComCam/DRP_DP1diffimWithFakes_v{RUNVERSION}"

import os
import numpy as np
import pandas as pd
import tqdm
from astropy import units as u
from astropy import coordinates as coords
from astropy.table import vstack

import lsst.daf.butler as dafButler

from astropy.stats import sigma_clipped_stats
from scipy import stats

from lsst.utils.plotting import (
    get_multiband_plot_colors,
    get_multiband_plot_symbols,
    get_multiband_plot_linestyles,
)
from lsst.utils.plotting import stars_cmap
from lsst.utils.plotting import publication_plots
publication_plots.set_rubin_plotstyle()

clrs = get_multiband_plot_colors()
bands_dict = publication_plots.get_band_dicts()

from matplotlib import colors
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import PercentFormatter
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pathEffects

from lsst.utils.plotting import (galaxies_cmap,
                                 galaxies_color,
                                 make_figure,
                                 stars_cmap,
                                 stars_color,
                                 set_rubin_plotstyle,
                                 divergent_cmap,
                                 accent_color,)
set_rubin_plotstyle()

from scipy.stats import binned_statistic, binned_statistic_2d
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from pathlib import Path
figures_filepath = Path("../figures")

# All input data read from ../data
data_filepath = Path("../data")
fake_matches_file = data_filepath / "dp1_diffim_fakes_matches.pqt"
fake_visits_list = data_filepath / "diffim_fakes_visit_list.txt"

if os.path.isfile(fake_matches_file):
    matches = pd.read_parquet(fake_matches_file)
    found = matches.found
    filter_flags = matches.filter_flags
else:
    butler = dafButler.Butler(repo, collections=collection, instrument=instrument)
    visits = np.loadtxt(fake_visits_list, dtype=np.int64)
    drefs = butler.query_datasets(
        "fakes_dia_source_matched",
        instrument=instrument,
        collections=collection,
        where=f"visit IN ({','.join(visits.astype(str))}) AND band IN ('g', 'r', 'i', 'z', 'y')"
    )

    catalogs = []
    for acat in tqdm.tqdm(drefs):
        matchcat = butler.get(acat)
        matchcat["visit"] = acat.dataId["visit"]
        matchcat["detector"] = acat.dataId["detector"]
        matchcat["band"] = acat.dataId["band"]
        matchcat["run"] = acat.run
        matchcat["day_obs"] = acat.dataId["day_obs"]
        catalogs.append(matchcat)

    matches = pd.concat(catalogs)
    del(catalogs)

    found =  matches.diaSourceId > 0
    matches["found"] = found

    matches["dist_host"] = (
        np.sqrt(matches["delta_ra"] ** 2 + matches["delta_dec"] ** 2) * 3600
    )

    matches["delta_mag"] = matches["mag"] - matches["host_magnitude"]

    matches_flags_when_false = [
        "forced_base_PixelFlags_flag_bad",
        "forced_base_LocalBackground_flag",
        "forced_base_PixelFlags_flag_interpolated",
        "forced_base_PixelFlags_flag_edgeCenter",
    ]
    filter_flags = np.ones(len(matches), dtype=bool)
    print(filter_flags.sum())
    for aflag in matches_flags_when_false:
        filter_flags &= ~matches[aflag].values

    matches['filter_flags'] = filter_flags

    matches["psfFlux_mag"] = (
        (u.nanojansky * matches["psfFlux"].values).to(u.ABmag).value
    )
    matches["psfFlux_magErr"] = (
        (u.nanojansky * (matches["psfFlux"].values  - matches["psfFluxErr"].values)).to(u.ABmag).value - \
        (u.nanojansky * (matches["psfFlux"].values  + matches["psfFluxErr"].values)).to(u.ABmag).value
        )/2
    matches["apFlux_mag"] = (
        (u.nanojansky * matches["apFlux"].values).to(u.ABmag).value
    )
    matches["apFlux_magErr"] = (
        (u.nanojansky * (matches["apFlux"].values  - matches["apFluxErr"].values)).to(u.ABmag).value - \
        (u.nanojansky * (matches["apFlux"].values  + matches["apFluxErr"].values)).to(u.ABmag).value
        )/2

    matches["pulls_psf"] = (
        matches["psfFlux"]
        - (matches["mag"].values * u.ABmag).to(u.nanojansky).value
    ) / matches["psfFluxErr"]
    matches["pulls_ap"] = (
        matches["apFlux"]
        - (matches["mag"].values * u.ABmag).to(u.nanojansky).value
    ) / matches["apFluxErr"]

    merged_fluxes = matches[(matches.forced_base_PsfFlux_instFlux_SNR > 5) & filter_flags]
    matches = matches[
        [
            "found", "ra_ssi", "dec_ssi", "ra_diaSrc", "dec_diaSrc",
            "forced_base_PsfFlux_instFlux_SNR", "isAssocDiaSource", "host_id",
            "band", "diaSourceId", "psfFlux", "mag", "psfFluxErr",
            "psfFlux_mag", "psfFlux_magErr", "visit", "detector", "filter_flags",
        ]
    ]

    matches.to_parquet(fake_matches_file)

merged_fluxes = matches[(matches.forced_base_PsfFlux_instFlux_SNR > 5) & filter_flags]
flux_offset = merged_fluxes['psfFlux'] - (merged_fluxes['mag'].values * u.ABmag).to(u.nanojansky).value
pulls = flux_offset / merged_fluxes['psfFluxErr']
merged_fluxes['psf_pulls'] = pulls

snrcut = 20
hi_snr = matches[matches.found & (matches["forced_base_PsfFlux_instFlux_SNR"]>snrcut)]
hi_snr_dra = 3600 * (hi_snr.ra_ssi - hi_snr.ra_diaSrc)
hi_snr_ddec = 3600 * (hi_snr.dec_ssi - hi_snr.dec_diaSrc)
hi_snr_x = hi_snr_dra.values * np.cos(hi_snr.dec_ssi * np.pi / 180)
hi_snr_y = hi_snr_ddec.values

dra = 3600 * (matches[matches.found].ra_ssi - matches[matches.found].ra_diaSrc)
ddec = 3600 * (matches[matches.found].dec_ssi - matches[matches.found].dec_diaSrc)

x = dra.values * np.cos(matches[found].dec_ssi * np.pi / 180)
y = ddec.values
# Set up figure with GridSpec
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(16, 16, hspace=0.01, wspace=0.01)

# Define axes
ax_main = fig.add_subplot(gs[4:, 0:12])
ax_xhist = fig.add_subplot(gs[0:4, 0:12], sharex=ax_main)
ax_yhist = fig.add_subplot(gs[4:, 12:15], sharey=ax_main)
cax = fig.add_subplot(gs[4:, 15])  # colorbar axis

# Hexbin on the main axis
hb = ax_main.hexbin(x, y, gridsize=72,
                    cmap=stars_cmap(single_color=True),
                    bins='log', mincnt=2, linewidths=0, edgecolors=None)

# Circle overlay (e.g., 1 arcsec radius)
circle = plt.Circle((0, 0), 0.5, edgecolor='gray', facecolor='none', linestyle=':', linewidth=1)
ax_main.add_patch(circle)

# Marginal histograms
bin_width = 0.005
bins = np.arange(-0.5, 0.5, bin_width)
ax_xhist.hist(x, bins=bins, color=stars_color(), histtype='step', label="All")
ax_xhist.hist(hi_snr_x, bins=bins, color=accent_color(), histtype='step', label=f"S/N >{snrcut}")

ax_yhist.hist(y, bins=bins, orientation='horizontal', color=stars_color(), histtype='step')
ax_yhist.hist(hi_snr_y, bins=bins, orientation='horizontal', color=accent_color(), histtype='step')

# get stats normal function with zero mean and estimated std from robust stats
mean_x, median_x, std_x = sigma_clipped_stats(hi_snr_x)
mean_y, median_y, std_y = sigma_clipped_stats(hi_snr_y)
mean_xy, median_xy, std_xy = sigma_clipped_stats(np.array([hi_snr_x, hi_snr_y]).flatten())

# overlay a gaussian
ax_xhist.plot(bins, stats.norm.pdf(bins, loc=0, scale=std_xy)*hi_snr_x.size*bin_width,
    color='k', lw=1, label='$\mathcal{N}(0,\sigma$'f'={std_xy:.2f})')
ax_yhist.plot(stats.norm.pdf(bins, loc=0, scale=std_xy)*hi_snr_y.size*bin_width,
    bins, color='k', lw=1)


ax_xhist.legend(ncols=1, loc="upper left")

ax_xhist.axvline(0, lw=0.5, color='k', alpha=0.5)
ax_yhist.axhline(0, lw=0.5, color='k', alpha=0.5)
# Colorbar
cb = fig.colorbar(hb, cax=cax)
# cb.set_label()
label = "Points Per Bin"
text = cax.text(0.5, 0.5, label, color="k",
                rotation="vertical",
                transform=cax.transAxes,
                ha="center",
                va="center",
                fontsize=12)
text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])

# Axis labels
ax_main.set_xlabel("Residual RA (arcsec)")
ax_main.set_ylabel("Residual Dec (arcsec)")
ax_xhist.set_ylabel("Count")
ax_yhist.set_xlabel("Count")
ax_main.set_xlim(-0.55, 0.55)
ax_main.set_ylim(-0.55, 0.55)

# Hide duplicated tick labels
ax_xhist.tick_params(labelbottom=False)
ax_yhist.tick_params(labelleft=False)

plt.savefig(figures_filepath / "coordinate_offsets_hexbin.pdf")

from scipy.optimize import curve_fit

def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))

# efficiency as function of any other parameter
def get_efficiency(
    df,
    xcol="forced_base_PsfFlux_instFlux_SNR",
    foundcol="found",
    bins=None,
    wid=0.5,
    min_snr=5,
):
    if bins is None:
        bins = np.arange(np.nanmin(df[xcol]), np.nanmax(df[xcol]), wid)

    redges = bins[:-1]
    center = bins[:-1] + wid / 2.0

    counts, mbins = np.histogram(df[xcol].values, bins=bins)
    mcount, mbins = np.histogram(df[df[foundcol] == True][xcol].values, bins=bins)

    eff = mcount / counts
    eff[np.where(counts == 0.0)] = 0.0
    err = 1.96 * np.sqrt(eff * (1 - eff) / counts)
    err[np.where(counts == 0.0)] = 0.02
    err[np.where(err == 0)] = 0.005

    hi_err = err.copy()
    hi_err[np.where((hi_err + eff) >= 1)] = (1 - eff)[np.where((hi_err + eff) >= 1)]
    yerr = [err, hi_err]
    try:
        popt, pcov = curve_fit(fsigmoid, center, eff, sigma=err, method="dogbox")
    except RuntimeError:
        popt = [0, 0, 0]
    eff_12 = np.round(popt[1], 2)
    return {
        "center": center,
        "redges": redges,
        "eff": eff,
        "counts": counts,
        "mcounts": mcount,
        "bins": bins,
        "yerr": yerr,
        "hi_err": hi_err,
        "err": err,
        "mbins": mbins,
        "eff_12": eff_12,
        "popt": popt,
    }

def plot_general_eff(
    df,
    xcol,
    bins,
    wid,
    foundcol="found",
    # candidateCol="isCandidate",
    associatedCol="isAssocDiaSource",
    ax1=None
):
    results = get_efficiency(df, xcol=xcol, foundcol=foundcol, wid=wid, bins=bins)

    resultsAssoc = get_efficiency(
        df, xcol=xcol, foundcol=associatedCol, wid=wid, bins=bins
    )

    if ax1 is None:
        fig, ax12 = plt.subplots(1, 1)

    # add a left vertical axis to plot scatters
    ax1 = ax12.twinx()

    ax12.set_ylabel(r"Completeness $\kappa$")
    ax12.set_ylim(0, 1.05)

    counts_x, binsx, _ = ax1.hist(
        df[xcol], bins=results["bins"], histtype="step",
        label="All Fakes", color=stars_color(),
        log=False, lw=2,
    )
    assoc_x, binsx, _ = ax1.hist(
        df[df[associatedCol]][xcol],
        bins=resultsAssoc["bins"],
        histtype="stepfilled",
        label="Detected",
        color=stars_color(),
        alpha=0.5,
        log=False,
    )

    ax12.plot(
        binsx[:-1] + wid / 2,
        resultsAssoc["eff"],
        c="k",
        label="Completeness",
        linestyle="-",
    )

    ax1.set_ylabel("Counts")
    ax1.tick_params(axis="y", labelcolor=stars_color())
    plt.tight_layout()

    return ax1, ax12, resultsAssoc

xcol = "forced_base_PsfFlux_instFlux_SNR"
wid = 0.25
minx = 0
maxx = 150
bins = np.arange(minx, maxx, wid)
match_table = matches[filter_flags & matches.band.isin(['g', 'r', 'i', 'z'])]
ax1, ax12, results = plot_general_eff(match_table, xcol, bins, wid)
ax12.set_xlabel('S/N Truth')
ax12.set_ylabel(r"Completeness")

ax12.plot([0, results['eff_12']], [0.5, 0.5],  color='k', linestyle=':', alpha=0.5,)
ax12.plot(2*[results['eff_12']], [0, 0.5],  color='k', linestyle=':', alpha=0.5,)
offset = 6
pct = 0.5
ax1.text(
    results['eff_12'] + offset,
    2750,
    "50%: " + str(results['eff_12']),
    ha="right",
    va="top",
    fontsize=11,
    bbox=dict(edgecolor='none', facecolor="white", alpha=0.5, boxstyle="round")
)
ax1.set_ylim(0, 6000)
ax1.set_xlim(-0.2, 30)
ax12.axhline(1, lw=0.95, color='k', alpha=0.5)
lines_right, labels_right = ax12.get_legend_handles_labels()
lines_left, labels_left = ax1.get_legend_handles_labels()
fig = plt.gcf()
fig.legend(lines_left + lines_right, labels_left + labels_right, loc=("upper center"), ncol=3)
fig.subplots_adjust(bottom=0.2, top=0.9)
plt.savefig(figures_filepath / "efficiency_snr_griz.pdf")

xcol = "mag"
foundcol = "found"
wid = 0.1
minx = 18
maxx = 25.9
bins = np.arange(minx, maxx, wid)
match_table = matches[filter_flags & matches.band.isin(['g', 'r', 'i', 'z'])]
ax1, ax12, results = plot_general_eff(match_table, xcol, bins, wid)

ax12.axhline(1, color='k', linestyle='--', label='', alpha=0.5, lw=1)
ax12.axhline(0.5, color='k', linestyle='--', label='Eff=50%', alpha=0.5, lw=1)
ax12.axhline(0.9, color='k', linestyle='-.', label='Eff=90%', alpha=0.5, lw=1)
ax1.axvline(24.35, color='k', linestyle='-', label='mag=24.4', alpha=0.5, lw=1)
ax1.axvline(23.35, color='k', linestyle=':', label='mag=23.4', alpha=0.5, lw=1)

lines_right, labels_right = ax1.get_legend_handles_labels()
lines_left, labels_left = ax12.get_legend_handles_labels()
ax1.legend(lines_left + lines_right, labels_left + labels_right,
    loc=(0.05, 0.1), ncol=2, fontsize=8)

ax1.set_xlim(17.5, 27)
ax1.set_xlabel('True mag fakes')
ax12.set_xlabel('True mag fakes')
ax12.title.set_text('$griz$ bandpass')

ax1.set_ylabel('N fakes')
plt.savefig(figures_filepath / "efficiency_mag_allbands.pdf")

xcol = "mag"
foundcol = "found"
wid = 0.1
minx = 18
maxx = 24.8
bins = np.arange(minx, maxx, wid)
df = matches[filter_flags & matches.band.isin(['i'])]
ax1, ax12, results = plot_general_eff(df, xcol, bins, wid)

ax12.axhline(1, color='k', linestyle='--', label='', alpha=0.5, lw=1)
ax12.axhline(0.5, color='k', linestyle='--', label='Eff=50%', alpha=0.5, lw=1)
ax12.axhline(0.9, color='k', linestyle='-.', label='Eff=90%', alpha=0.5, lw=1)
ax1.axvline(23.83, color='k', linestyle='-', label='mag=24.4', alpha=0.5, lw=1)
ax1.axvline(23.37, color='k', linestyle=':', label='mag=23.4', alpha=0.5, lw=1)

lines_right, labels_right = ax1.get_legend_handles_labels()
lines_left, labels_left = ax12.get_legend_handles_labels()
ax1.legend(lines_left + lines_right, labels_left + labels_right,
    loc=(0.05, 0.1), ncol=2, fontsize=8)

ax1.set_xlim(17.5, 27)
ax12.title.set_text('$i$ bandpass')
ax12.set_xlabel('True mag fakes')
ax1.set_ylabel('N fakes')

# xbins = np.arange(17, 24, 0.1)
xbins = np.logspace(np.log10(17.8), np.log10(24.5), num=30)
ybins = np.arange(-0.3, 0.3, 0.01)

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(16, 16, hspace=0.01, wspace=0.01)

# Define axes
ax_main = fig.add_subplot(gs[4:, 0:12])
ax_xhist = fig.add_subplot(gs[0:4, 0:12], sharex=ax_main)
ax_xcompl = ax_xhist.twinx()
ax_yhist = fig.add_subplot(gs[4:, 12:15], sharey=ax_main)
cax = fig.add_subplot(gs[4:, 15])

selection = merged_fluxes['band'].isin(['i']) & \
    (merged_fluxes['forced_base_PsfFlux_instFlux_SNR'] > 5)
host_ids = merged_fluxes[selection]['host_id'].values
x = np.asarray(merged_fluxes[selection]["mag"].values)
values = np.asarray(merged_fluxes[selection]["psfFlux_mag"].values - merged_fluxes[selection]["mag"].values)
indices = np.isnan(x) | np.isnan(values)

x = x[~indices]
values = values[~indices]
host_ids = host_ids[~indices]
df = matches[filter_flags & matches.band.isin(['i'])]
flux_df = merged_fluxes[selection][~indices]

bns = binned_statistic(
    x,
    values,
    statistic="median",
    bins=xbins,
    range=(18, 24),
)
xbincenters = (bns.bin_edges[1:]+bns.bin_edges[:-1])/2

means = bns.statistic
ax_main.plot(xbincenters, means, color="k", lw=2, label="Running Median")

bns = binned_statistic(
    x,
    values,
    statistic=stats.median_abs_deviation,
    bins=xbins,
    range=(18, 24),
)
stds = bns.statistic
hb = ax_main.hexbin(
    x,
    values,
    cmap=stars_cmap(single_color=True),
    #bins='log',
    linewidths=0,
    gridsize=80,
    mincnt=1,
    extent=(17.5, 25, -0.25, 0.25),
    edgecolors=None,
)
ax_main.axhline(0, linestyle=":", color="k", zorder=-1)

ax_main.plot(
    xbincenters,
    means+stds,
    color="k",
    label=r"Median $\pm\sigma_{MAD}$",
    linestyle="--",
)
ax_main.plot(
    xbincenters,
    means-stds,
    color="k",
    linestyle="--",
)

ax_xhist.hist(
    x,
    bins=xbins,
    histtype="step",
    lw=2,
    color=stars_color(),
    label="All",
    orientation="vertical",
)
ax_xhist.hist(
    x[host_ids>1],
    bins=xbins,
    histtype="step",
    lw=2,
    color=accent_color(),
    label='Hosted',
    orientation="vertical",
)
ax_xhist.hist(
    x[host_ids<=1],
    bins=xbins,
    histtype="step",
    lw=2, linestyle='--',
    color="k",
    label='Hostless',
    orientation="vertical",
)
ax_yhist.hist(
    values[host_ids>1],
    bins=ybins,
    histtype="step",
    lw=2,
    color=accent_color(),
    density=True,
    label='Hosted',
    orientation="horizontal",
)
ax_yhist.hist(
    values[host_ids<=1],
    bins=ybins,
    histtype="step",
    lw=2, linestyle='--',
    color="k",
    density=True,
    label='Hostless',
    orientation="horizontal",
)

widcomp = 0.1
minxcomp = 18
maxxcomp = 24.8
binscomp = np.arange(minxcomp, maxxcomp, widcomp)
associatedCol="isAssocDiaSource"
resultsAssoc = get_efficiency(
    df, xcol=xcol, foundcol=associatedCol, wid=widcomp, bins=binscomp
)
ax_xcompl.plot(
    resultsAssoc["center"],
    resultsAssoc["eff"],
    c="k",
    label="Completeness",
    linestyle="-",
    alpha=0.5,
)
ax_xcompl.set_ylabel(r"Completeness", fontsize=12)
ax_xcompl.set_ylim(0.05, 1.05)

ax_main.axvline(24., color='k', linestyle='-', label='Compl. 50% mag=24.4', alpha=0.5, lw=1)
ax_xcompl.axvline(24., color='k', linestyle='-', alpha=0.5, lw=1)
ax_main.axvline(23.5, color='k', linestyle='--', label='Compl. 90% mag=23.4', alpha=0.5, lw=1)
ax_xcompl.axvline(23.5, color='k', linestyle='--', alpha=0.5, lw=1)


lines_right, labels_right = ax_xcompl.get_legend_handles_labels()
lines_left, labels_left = ax_xhist.get_legend_handles_labels()
ax_xhist.legend(lines_left + lines_right, labels_left + labels_right, loc=("lower left"), ncol=4, fontsize=8)

ax_xhist.set_yscale('log')

# Colorbar
cb = fig.colorbar(hb, cax=cax)
label = "Points Per Bin"
text = cax.text(0.5, 0.5, label, color="k",
                rotation="vertical",
                transform=cax.transAxes,
                ha="center",
                va="center",
                fontsize=12)
text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])
ax_main.set_xlim(17.5, 24.7)
ax_main.set_ylim(-0.25, 0.25)

# Hide duplicated tick labels
ax_xhist.tick_params(labelbottom=False)
ax_yhist.tick_params(labelleft=False)

ax_main.legend(loc='lower left', ncols=1)
ax_main.set_ylabel("PSF Mag - True Mag (mag)")
ax_main.set_xlabel("True Mag")
ax_xhist.set_ylabel("Count")
ax_yhist.set_xlabel("Normalized\nCount", fontsize=12)
plt.savefig(figures_filepath / "hexbin_psf_mag.pdf")
plt.show()

# xbins = np.arange(17, 24, 0.1)
xbins = np.logspace(np.log10(17.8), np.log10(24.5), num=30)
ybins = np.arange(-5, 5, 0.1)

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(16, 16, hspace=0.01, wspace=0.01)

# Define axes
ax_main = fig.add_subplot(gs[4:, 0:12])
ax_xhist = fig.add_subplot(gs[0:4, 0:12], sharex=ax_main)
ax_xcompl = ax_xhist.twinx()
ax_yhist = fig.add_subplot(gs[4:, 12:15], sharey=ax_main)
cax = fig.add_subplot(gs[4:, 15])

selection = merged_fluxes['band'].isin(['i']) & \
    (merged_fluxes['forced_base_PsfFlux_instFlux_SNR'] > 5)
host_ids = merged_fluxes[selection]['host_id'].values
x = np.asarray(merged_fluxes[selection]["mag"].values)
values = merged_fluxes[selection]["psf_pulls"].values
indices = np.isnan(x) | np.isnan(values)

x = x[~indices]
values = values[~indices]
host_ids = host_ids[~indices]
df = matches[filter_flags & matches.band.isin(['i'])]
flux_df = merged_fluxes[selection][~indices]

bns = binned_statistic(
    x,
    values,
    statistic="median",
    bins=xbins,
    range=(18, 24),
)
xbincenters = (bns.bin_edges[1:]+bns.bin_edges[:-1])/2

means = bns.statistic
ax_main.plot(xbincenters, means, color="k", lw=2, label="Running Median")

bns = binned_statistic(
    x,
    values,
    statistic=stats.median_abs_deviation,
    bins=xbins,
    range=(18, 24),
)
stds = bns.statistic
hb = ax_main.hexbin(
    x,
    values,
    cmap=stars_cmap(single_color=True),
    linewidths=0,
    gridsize=80,
    mincnt=1,
    extent=(17.5, 25, -3.5, 3.5),
    edgecolors=None,
)
ax_main.axhline(0, linestyle=":", color="k", zorder=-1)

ax_main.plot(
    xbincenters,
    means+stds,
    color="k",
    label=r"Median $\pm\sigma_{MAD}$",
    linestyle="--",
)
ax_main.plot(
    xbincenters,
    means-stds,
    color="k",
    linestyle="--",
)

ax_xhist.hist(
    x,
    bins=xbins,
    histtype="step",
    lw=2,
    color=stars_color(),
    label="All",
    orientation="vertical",
)
ax_xhist.hist(
    x[host_ids>1],
    bins=xbins,
    histtype="step",
    lw=2,
    color=accent_color(),
    label='Hosted',
    orientation="vertical",
)
ax_xhist.hist(
    x[host_ids<=1],
    bins=xbins,
    histtype="step",
    lw=2, linestyle='--',
    color="k",
    label='Hostless',
    orientation="vertical",
)
ax_yhist.hist(
    values[host_ids>1],
    bins=ybins,
    histtype="step",
    lw=2,
    color=accent_color(),
    density=True,
    label='Hosted',
    orientation="horizontal",
)
ax_yhist.hist(
    values[host_ids<=1],
    bins=ybins,
    histtype="step",
    lw=2, linestyle='--',
    color="k",
    density=True,
    label='Hostless',
    orientation="horizontal",
)

widcomp = 0.1
minxcomp = 18
maxxcomp = 24.8
binscomp = np.arange(minxcomp, maxxcomp, widcomp)
associatedCol="isAssocDiaSource"
resultsAssoc = get_efficiency(
    df, xcol=xcol, foundcol=associatedCol, wid=widcomp, bins=binscomp
)
ax_xcompl.plot(
    resultsAssoc["center"],
    resultsAssoc["eff"],
    c="k",
    label="Completeness",
    linestyle="-",
    alpha=0.5,
)
ax_xcompl.set_ylabel(r"Completeness", fontsize=12)
ax_xcompl.set_ylim(0.05, 1.05)

ax_main.axvline(24., color='k', linestyle='-', label='Compl. 50% mag=24.4', alpha=0.5, lw=1)
ax_xcompl.axvline(24., color='k', linestyle='-', alpha=0.5, lw=1)
ax_main.axvline(23.5, color='k', linestyle='--', label='Compl. 90% mag=23.4', alpha=0.5, lw=1)
ax_xcompl.axvline(23.5, color='k', linestyle='--', alpha=0.5, lw=1)
ax_main.axhline(1, lw=0.5, color='k', alpha=0.5)
ax_main.axhline(-1, lw=0.5, color='k', alpha=0.5)

lines_right, labels_right = ax_xcompl.get_legend_handles_labels()
lines_left, labels_left = ax_xhist.get_legend_handles_labels()
ax_xhist.legend(lines_left + lines_right, labels_left + labels_right, loc=("lower left"), ncol=4, fontsize=8)

ax_xhist.set_yscale('log')

# Colorbar
cb = fig.colorbar(hb, cax=cax)
label = "Points Per Bin"
text = cax.text(0.5, 0.5, label, color="k",
                rotation="vertical",
                transform=cax.transAxes,
                ha="center",
                va="center",
                fontsize=12)
text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])
ax_main.set_xlim(17.5, 24.7)
ax_main.set_ylim(-5, 5)

# Hide duplicated tick labels
ax_xhist.tick_params(labelbottom=False)
ax_yhist.tick_params(labelleft=False)

ax_main.legend(loc='lower left', ncols=1)
# ax_main.set_ylabel("(PSF Flux - True Flux)/Psf Flux Err")
ax_main.set_ylabel("$(f_{PSF} - f_{True})/\sigma_{f_{PSF}}$")
ax_main.set_xlabel("True Mag")
ax_xhist.set_ylabel("Count")
ax_yhist.set_xlabel("Normalized\nCount", fontsize=12)
plt.savefig(figures_filepath / "hexbin_psf_pull.pdf")
plt.show()

# xbins = np.arange(17, 24, 0.1)
xbins = np.logspace(np.log10(17.8), np.log10(24.5), num=30)
ybins = np.arange(-0.3, 0.3, 0.01)

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(16, 16, hspace=0.01, wspace=0.01)

# Enlarge the figure and grid to accommodate another plot below
fig = plt.figure(figsize=(8, 12))  # double the height
gs = gridspec.GridSpec(24, 16, hspace=0.01, wspace=0.01)  # double the rows

# Define axes for the first plot (top, mag)
ax_main_mag = fig.add_subplot(gs[4:14, 0:12])
ax_yhist_mag = fig.add_subplot(gs[4:14, 12:15], sharey=ax_main_mag)
cax = fig.add_subplot(gs[4:14, 15])  # colorbar axis
ax_xhist = fig.add_subplot(gs[0:4, 0:12], sharex=ax_main_mag)
ax_xcompl = ax_xhist.twinx()

# Define axes for the second plot (bottom, pull)
ax_main_pull = fig.add_subplot(gs[14:24, 0:12])
ax_yhist_pull = fig.add_subplot(gs[14:24, 12:15], sharey=ax_main_pull)
cax2 = fig.add_subplot(gs[14:24, 15])  # colorbar axis for second plot

selection = merged_fluxes['band'].isin(['i']) & \
    (merged_fluxes['forced_base_PsfFlux_instFlux_SNR'] > 5)
host_ids = merged_fluxes[selection]['host_id'].values
x = np.asarray(merged_fluxes[selection]["mag"].values)
values = np.asarray(
    merged_fluxes[selection]["psfFlux_mag"].values - \
    merged_fluxes[selection]["mag"].values
)
pulls = merged_fluxes[selection]["psf_pulls"].values

indices = np.isnan(x) | np.isnan(values)

x = x[~indices]
values = values[~indices]
pulls = pulls[~indices]
host_ids = host_ids[~indices]
true_mags = merged_fluxes[selection]["mag"].values[~indices]
df = matches[filter_flags & matches.band.isin(['i'])]
flux_df = merged_fluxes[selection][~indices]

bns = binned_statistic(
    x,
    values,
    statistic="median",
    bins=xbins,
    range=(18, 24),
)
xbincenters = (bns.bin_edges[1:]+bns.bin_edges[:-1])/2

means = bns.statistic
ax_main_mag.plot(xbincenters, means, color="k", lw=2, label="Running Median")

bns = binned_statistic(
    x,
    values,
    statistic=stats.median_abs_deviation,
    bins=xbins,
    range=(18, 24),
)
stds = bns.statistic
hb = ax_main_mag.hexbin(
    x,
    values,
    cmap=stars_cmap(single_color=True),
    linewidths=0,
    gridsize=80,
    mincnt=1,
    extent=(17.5, 25, -0.25, 0.25),
    edgecolors=None,
)
ax_main_mag.axhline(0, linestyle=":", color="k", zorder=-1)

ax_main_mag.plot(
    xbincenters,
    means+stds,
    color="k",
    label=r"Median $\pm\sigma_{MAD}$",
    linestyle="--",
)
ax_main_mag.plot(
    xbincenters,
    means-stds,
    color="k",
    linestyle="--",
)

ax_xhist.hist(
    x,
    bins=xbins,
    histtype="step",
    lw=2,
    color=stars_color(),
    label="All",
    orientation="vertical",
)
ax_xhist.hist(
    x[host_ids>1],
    bins=xbins,
    histtype="step",
    lw=2,
    color=accent_color(),
    label='Hosted',
    orientation="vertical",
)
ax_xhist.hist(
    x[host_ids<=1],
    bins=xbins,
    histtype="step",
    lw=2, linestyle='--',
    color="k",
    label='Hostless',
    orientation="vertical",
)
# ax_yhist.hist(
#     values,
#     bins=ybins,
#     histtype="step",
#     lw=2,
#     label="All",
#     color=stars_color(),
#     orientation="horizontal",
# )

# Shade region to the right of magnitude 22.5 in both plots
mag_cut = 22.5
mask = true_mags > 0
if mag_cut is not None:
    mask = true_mags < mag_cut
    ax_main_mag.axvspan(mag_cut, ax_main_mag.get_xlim()[1], color='gray', alpha=0.2, zorder=-2)
    ax_main_pull.axvspan(mag_cut, ax_main_pull.get_xlim()[1], color='gray', alpha=0.2, zorder=-2)

ax_yhist_mag.hist(
    values[(host_ids>1)&(mask)],
    bins=ybins,
    histtype="step",
    lw=2,
    color=accent_color(),
    density=True,
    label='Hosted',
    orientation="horizontal",
)
ax_yhist_mag.hist(
    values[(host_ids<=1)&(mask)],
    bins=ybins,
    histtype="step",
    lw=2, linestyle='--',
    color="k",
    density=True,
    label='Hostless',
    orientation="horizontal",
)
mvh, medvh, stdvh = sigma_clipped_stats(values[(host_ids>1)&(mask)], sigma=3, maxiters=5)
mv, medv, stdv = sigma_clipped_stats(values[(host_ids<=1)&(mask)], sigma=3, maxiters=5)

text_stdsh = f"Hosted\n$\mu=${mvh:.3f}\n$\sigma=${stdvh:.3f}"
ax_yhist_mag.text(0.5, 0.85,
    text_stdsh,
    color=accent_color(),
    rotation="horizontal",
    transform=ax_yhist_mag.transAxes,
    ha="center",
    va="center",
    fontsize=12)
text_stds = f"Hostless\n$\mu=${mv:.3f}\n$\sigma=${stdv:.3f}"
ax_yhist_mag.text(0.5, 0.15,
    text_stds,
    color="k",
    rotation="horizontal",
    transform=ax_yhist_mag.transAxes,
    ha="center",
    va="center",
    fontsize=12)
# Now repeat for pulls
bns_pulls = binned_statistic(
    x,
    pulls,
    statistic="median",
    bins=xbins,
    range=(18, 24),
)
means_pulls = bns_pulls.statistic
bns_pulls = binned_statistic(
    x,
    pulls,
    statistic=stats.median_abs_deviation,
    bins=xbins,
    range=(18, 24),
)
stds_pulls = bns_pulls.statistic

hb_pull = ax_main_pull.hexbin(
    x,
    pulls,
    cmap=stars_cmap(single_color=True),
    #bins='log',
    linewidths=0,
    gridsize=80,
    mincnt=1,
    extent=(17.5, 25, -3.5, 3.5),
    edgecolors=None,
)
ax_main_pull.axhline(0, linestyle=":", color="k", zorder=-1)
ax_main_pull.plot(xbincenters, means_pulls, color="k", lw=2, label="Running Median")

ax_main_pull.plot(
    xbincenters,
    means_pulls+stds_pulls,
    color="k",
    label=r"Median $\pm\sigma_{MAD}$",
    linestyle="--",
)
ax_main_pull.plot(
    xbincenters,
    means_pulls-stds_pulls,
    color="k",
    linestyle="--",
)
ybins_pulls = np.arange(-5, 5, 0.1)
ax_yhist_pull.hist(
    pulls[(host_ids>1)&mask],
    bins=ybins_pulls,
    histtype="step",
    lw=2,
    color=accent_color(),
    density=True,
    label='Hosted',
    orientation="horizontal",
)
ax_yhist_pull.hist(
    pulls[(host_ids<=1)&mask],
    bins=ybins_pulls,
    histtype="step",
    lw=2, linestyle='--',
    color="k",
    density=True,
    label='Hostless',
    orientation="horizontal",
)
ax_main_pull.axhline(1, lw=0.5, color='k', alpha=0.5)
ax_main_pull.axhline(-1, lw=0.5, color='k', alpha=0.5)

mph, medph, stdph = sigma_clipped_stats(pulls[(host_ids>1)&mask], sigma=3, maxiters=5)
mp, medp, stdp = sigma_clipped_stats(pulls[(host_ids<=1)&mask], sigma=3, maxiters=5)

text_stdsh = f"Hosted\n$\mu=${mph:.3f}\n$\sigma=${stdph:.3f}"
ax_yhist_pull.text(0.5, 0.85,
    text_stdsh,
    color=accent_color(),
    rotation="horizontal",
    transform=ax_yhist_pull.transAxes,
    ha="center",
    va="center",
    fontsize=12)
text_stds = f"Hostless\n$\mu=${mp:.3f}\n$\sigma=${stdp:.3f}"
ax_yhist_pull.text(0.5, 0.15,
    text_stds,
    color="k",
    rotation="horizontal",
    transform=ax_yhist_pull.transAxes,
    ha="center",
    va="center",
    fontsize=12)

widcomp = 0.1
minxcomp = 18
maxxcomp = 24.8
binscomp = np.arange(minxcomp, maxxcomp, widcomp)
associatedCol="isAssocDiaSource"
resultsAssoc = get_efficiency(
    df, xcol=xcol, foundcol=associatedCol, wid=widcomp, bins=binscomp
)
ax_xcompl.plot(
    resultsAssoc["center"],
    resultsAssoc["eff"],
    c="k",
    label="Completeness",
    linestyle="-",
    alpha=0.5,
)
ax_xcompl.set_ylabel(r"Completeness", fontsize=12)
ax_xcompl.set_ylim(0.05, 1.05)

ax_main_mag.axvline(24., color='k', linestyle='-', label='Compl. 50% mag=24.4', alpha=0.5, lw=1)
ax_main_pull.axvline(24., color='k', linestyle='-', label='Compl. 50% mag=24.4', alpha=0.5, lw=1)
ax_xcompl.axvline(24., color='k', linestyle='-', alpha=0.5, lw=1)

ax_main_mag.axvline(23.5, color='k', linestyle='--', label='Compl. 90% mag=23.4', alpha=0.5, lw=1)
ax_main_pull.axvline(23.5, color='k', linestyle='--', label='Compl. 90% mag=23.4', alpha=0.5, lw=1)
ax_xcompl.axvline(23.5, color='k', linestyle='--', alpha=0.5, lw=1)

lines_right, labels_right = ax_xcompl.get_legend_handles_labels()
lines_left, labels_left = ax_xhist.get_legend_handles_labels()
ax_xhist.legend(lines_left + lines_right, labels_left + labels_right, loc=("lower left"), ncol=4, fontsize=8)
ax_xhist.set_yscale('log')

# Colorbar
cb = fig.colorbar(hb, cax=cax)
label = "Points Per Bin"
text = cax.text(0.5, 0.5, label, color="k",
                rotation="vertical",
                transform=cax.transAxes,
                ha="center",
                va="center",
                fontsize=12)
text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])
ax_main_mag.set_xlim(17.5, 24.7)
ax_main_mag.set_ylim(-0.25, 0.25)

cb_pull = fig.colorbar(hb_pull, cax=cax2)
label = "Points Per Bin"
text_pull = cax2.text(0.5, 0.5, label, color="k",
                rotation="vertical",
                transform=cax2.transAxes,
                ha="center",
                va="center",
                fontsize=12)
text_pull.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])
ax_main_pull.set_xlim(17.5, 24.7)
ax_main_pull.set_ylim(-4.8, 4.8)

# Hide duplicated tick labels
ax_xhist.tick_params(labelbottom=False)
ax_yhist_mag.tick_params(labelleft=False)
ax_yhist_pull.tick_params(labelleft=False)

ax_main_mag.legend(loc='lower left', ncols=1)
ax_main_mag.set_ylabel("PSF Mag - True Mag (mag)")
ax_main_pull.set_ylabel("$(f_{PSF} - f_{True})/\sigma_{f_{PSF}}$")
ax_main_pull.set_xlabel("True Mag")
ax_xhist.set_ylabel("Count")
ax_yhist.set_xlabel("Normalized\nCount", fontsize=12)
plt.savefig(figures_filepath / "hexbin_psf_magpull.pdf")
plt.show()

bns = binned_statistic(
    flux_df["mag"].values,
    flux_df["psfFlux_magErr"].values,
    statistic="median",
    bins=xbins,
    range=(18, 24),
)
plt.errorbar(
    xbincenters,
    means,
    yerr=stds,
    fmt=".",
    color="k",
    label="Running Median with MAD",
    lw=0.5,
)

plt.ylim(-0.01, 0.01)
plt.xlim(17.5, 24.5)
ax = plt.gca()
ax.set_ylabel("PSF Mag - True Mag (mag)")
ax.set_xlabel("True Mag")
plt.grid()

plt.plot(xbincenters, stds, label="Median Absolute Deviation of PSF Mag")
plt.plot(xbincenters, bns.statistic, label="Median PsfMagError")
plt.grid()
plt.xlabel("True Mag")
plt.ylabel("MAD")
plt.legend()
plt.xlim(17.5, 24.5)
plt.ylim(0, 0.1)