import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

from lsst.daf.butler import Butler
import lsst.geom

# Set a standard figure size to use
from lsst.utils.plotting import publication_plots
from lsst.utils.plotting import get_multiband_plot_colors, get_multiband_plot_symbols, get_multiband_plot_linestyles


instrument = 'LSSTComCam'
collections = ['skymaps',
               'LSSTComCam/DP1',
               'LSSTComCam/runs/DRP/DP1/v29_0_0/DM-50260' ]
skymap = 'lsst_cells_v1'
butler = Butler("/repo/dp1",
                instrument=instrument,
                collections=collections,
                skymap=skymap)


# Applying the Rubin DP1 paper custom style sheet to all of the graphs to be created in this notebook
publication_plots.set_rubin_plotstyle()
bands_dict = publication_plots.get_band_dicts()
colors = get_multiband_plot_colors()
bands = colors.keys()  # important to get the right order for plot legends

t = butler.get("visit_detector_table")

# This is very low -- 0.135 arcsec seeing is not physical.
# Should these and other such visits be excluded from the dataset?

# This is clearly non-physical 
# Put a lower cut on IQ at 0.6  to exclude non-physical values 
# -- based on SITCOMTN report of 0.65 bing best IQ 
t["psfFwhm"] = t["psfSigma"]*2.355*0.2 
use = t["psfFwhm"] >= 0.6
t = t[use]

# Compute all band summary statistics
# Extract data from visit table for plotting
data = {}
for band in bands:
    bandMask = t["band"] == band
    data[band] = t["psfFwhm"][bandMask]


# IQ histogram - not for inclusion in the paper but interesting
plt.figure()

for label, d in data.items():
    plt.hist(d, bins=20, alpha=0.5,
             linestyle='-',
             color = colors[label],
             label=label)

# Customize plot
plt.xlabel('PSF FWHM (arcsecs)')
plt.ylabel('Fraction of Sensors')
plt.legend()

plt.savefig("image_quality_histo.pdf", 
            bbox_inches='tight',  # Trim whitespace around the figure
            transparent=True,     # Transparent background
            format='png')         # Explicit format specification
plt.show()

# IQ ECDF
plt.figure()

for label, d in data.items():
    ecdf = ECDF(d)
    plt.plot(ecdf.x, ecdf.y, 
             linestyle='-',
             color = colors[label],
             label=label)

# Customize plot
plt.xlabel('PSF FWHM (arcsecs)')
plt.ylabel('Fraction of Sensors')
plt.xlim(0.4, 2.7)
plt.legend(loc="lower right")
plt.savefig("image_quality_ecdf.pdf", 
            bbox_inches='tight',  # Trim whitespace around the figure
            transparent=True,)     # Transparent background         # Explicit format specification
plt.show()
plt.close()

names = ["47 Tucanae", "Fornax", "ECDFS", "EDFS", "Rubin SV 95 -25", "Seagull", "Rubin SV 38 7"]
raMins = [3.8, 39.2, 52.3, 58, 94.2, 105.55, 36.75]
raMaxes = [8.3, 40.9, 54.0, 60.25, 95.85, 107, 39]
decMins = [-72.8, -35.2, -28.8, -49.5, -25.8, -11.2, 5.75]
decMaxes = [-71.4, -33.6, -27.3, -48.0, -24.2, -9.75, 8.1]

fieldData = zip(names, raMins, raMaxes, decMins, decMaxes)
symbols = get_multiband_plot_symbols()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between([60622, 60658], 2, 3, color="grey", alpha=0.2, lw=0)
ax.fill_between([60622, 60658], 4, 5, color="grey", alpha=0.2, lw=0)
ax.fill_between([60622, 60658], 6, 7, color="grey", alpha=0.2, lw=0)
ax.fill_between([60622, 60658], 8, 9, color="grey", alpha=0.2, lw=0)

for (i, (name, raMin, raMax, decMin, decMax)) in enumerate(fieldData):
    mask = (t["ra"] < raMax) & (t["ra"] > raMin) & (t["dec"] < decMax) & (t["dec"] > decMin)
    tField = t[mask]
    for (j, band) in enumerate(bands):
        bandMask = tField["band"] == band

        ys = np.ones(np.sum(bandMask))*(8-i) + (j+1)/8
        xs = tField["expMidptMJD"][bandMask]

        if i == 1:
            plt.plot(xs, ys, color=colors[band], marker=".", ms=5, ls="none", label=band)
        else:
            plt.plot(xs, ys, color=colors[band], marker=".", ms=5, ls="none")

ax.set_yticks([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
ax.set_yticklabels(names)

plt.ylim(2, 9)
plt.legend(loc="lower left")
plt.xlim(60622, 60658)
plt.xlabel("MJD")
plt.subplots_adjust(left=0.25, right=0.95, top=0.95)
plt.savefig("visitDates.pdf")
plt.show()
