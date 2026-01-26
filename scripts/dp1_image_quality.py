import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.ticker import MultipleLocator

from collections import defaultdict
from lsst.daf.butler import Butler
import lsst.geom

# Set a standard figure size to use
from lsst.utils.plotting import publication_plots, set_rubin_plotstyle
from lsst.utils.plotting import get_multiband_plot_colors, get_multiband_plot_symbols, get_multiband_plot_linestyles

instrument = 'LSSTComCam'
collections = ['LSSTComCam/DP1']
skymap = 'lsst_cells_v1'
butler = Butler("dp1",
                instrument=instrument,
                collections=collections,
                skymap=skymap)


# Applying the Rubin DP1 paper custom style sheet to all of the graphs to be created in this notebook
publication_plots.set_rubin_plotstyle()
bands_dict = publication_plots.get_band_dicts()
colors = get_multiband_plot_colors()
bands = colors.keys()  # important to get the right order for plot legends

t = butler.get("visit_detector_table", storageClass = "DataFrame")

# Put a lower cut on IQ at 0.6  to exclude non-physical values 
# -- based on SITCOMTN report of 0.65 bing best IQ 
t["psfFwhm"] = t["psfSigma"]*2.355*0.2 
use = t["psfFwhm"] >= 0.6
t = t[use]

# Compute per- and all- band summary statistics
df = t[['detectorId','visitId', 'band', 'psfFwhm']].copy()

# Compute per-band and all-bands IQ
iq_band = df.groupby('band')['psfFwhm'].quantile(0.5).round(2)
iq_band.index.name = 'band'
iq_band = iq_band.reindex(bands)

data = {}
for band in bands:
    bandMask = t["band"] == band
    data[band] = t["psfFwhm"][bandMask]

##############  IQ histogram
plt.figure()
for label, d in data.items():
    lg_label = f"{label} ($\\tilde{{x}}$={iq_band[label]})" 
    plt.hist(d, bins=60, linewidth=2.0,
             linestyle='-', histtype='step',
             color=colors[label],
             label=lg_label)

# Customize plot 
plt.xlabel('PSF FWHM (arcsecs)')
plt.ylabel('Fraction of Sensors')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("../figures/image_quality_histo.pdf", 
            bbox_inches='tight',
            transparent=True,
            format='pdf')
plt.close()

################### IQ ECDF ############################
plt.figure()
for label, d in data.items():
    ecdf = ECDF(d)
    plt.plot(ecdf.x, ecdf.y, 
             linestyle='-',
             color = colors[label],
             label=label)
    
# Customize plot
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
plt.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
plt.xlabel('PSF FWHM (arcsecs)')
plt.ylabel('Cumulative Probability')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("../figures/image_quality_ecdf.pdf", 
            bbox_inches='tight',
            transparent=True,
            format='pdf')
plt.close()

############## IQ per target field
registry = butler.registry
exposures = registry.queryDimensionRecords('exposure')
exp_df = pd.DataFrame(columns=['id', 'target', 'physical_filter','ra', 'dec'])
for count, info in enumerate(exposures):
    exp_df.loc[count] = [info.id, info.target_name, info.physical_filter, 
                         info.tracking_ra, info.tracking_dec,]

tdf = exp_df.merge(df, left_on='id', right_on='visitId', how='inner')
tdf.loc[tdf['target'] == 'slew_icrs', 'target'] = 'ECDFS'
tdf = tdf[['visitId', 'detectorId', 'target', 'band', 'psfFwhm']]

set_rubin_plotstyle()
plt.figure()
for target in tdf['target'].unique():
    data = tdf[tdf['target'] == target]['psfFwhm']
    median_val = data.median()
    lg_label = f"{target} ($\\tilde{{x}}$={median_val:.2f})"
    plt.hist(data, bins=60, linewidth=2.0,
             histtype='step',
             label=lg_label)

# Customize plot 
plt.xlabel('PSF FWHM (arcsecs)')
plt.ylabel('Fraction of Sensors')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("../figures/image_quality_histo_per_field.pdf", 
            bbox_inches='tight',
            transparent=True,
            format='pdf')
plt.close()

############ Visit Dates ###############################

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
plt.legend(bbox_to_anchor=(0.93, 1.15), ncols=6)
plt.xlim(60622, 60658)
plt.xlabel("MJD")
plt.subplots_adjust(left=0.22, right=0.99, top=0.85, bottom=0.15)
plt.savefig("../figures/visitDates.pdf")
plt.show()