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
# Original notebook written by Clare Saunders. Converted to a python
# script by James Mullaney.

# Contact Authors: Clare Saunders, James Mullaney

from lsst.daf.butler import Butler
import numpy as np
import matplotlib.pyplot as plt

from lsst.utils.plotting import (publication_plots,
                                 divergent_cmap,
                                 accent_color,
                                 stars_color)
publication_plots.set_rubin_plotstyle()

repo = '/repo/main'
instrument = 'LSSTComCam'
collections = ['LSSTComCam/runs/DRP/DP1/v29_0_0_rc6/DM-50098']
skymapname = 'lsst_cells_v1'
butler = Butler(repo,instrument=instrument,
                collections=collections,
                skymap=skymapname)

# Gather data needed for AM1 and dmL2AstroErr
# AM1 and dmL2AstroErr
catalog = "recalibrated_star_association_metrics"

info = list(butler.registry.queryDatasets(catalog))

dataID = []
for run in info:
    dataID.append(run.dataId)
dataID = list(set(dataID))


AM1 = {}
dmL2AstroErr = {}

for dat in dataID:
    cat = butler.get(catalog, dat)

    for i in range(len(cat['stellarAstrometricRepeatability1'])):
        A = cat['stellarAstrometricRepeatability1'][i]

        if "AM1" in A.metric_name.metric:
            name = str(dat['tract'])+"_"+ A.metric_name.metric[0]
            AM1.update({name:A.quantity.value})
    for j in range(len(cat['stellarAstrometricSelfRepeatabilityRA'])):
        B = cat['stellarAstrometricSelfRepeatabilityRA'][j]
        if "dmL2AstroErr_RA" in B.metric_name.metric:
            name = str(dat['tract'])+"_"+ B.metric_name.metric[0]
            dmL2AstroErr.update({name:B.quantity.value})

# Gather data needed for AM1 and dmL1AstroErr
# dmL1AstroErr
catalog = "single_visit_star_association_metrics"

info = list(butler.registry.queryDatasets(catalog))

dataID = []
for run in info:
    dataID.append(run.dataId)
dataID = list(set(dataID))

dmL1AstroErr = {}

for dat in dataID:
    cat = butler.get(catalog, dat)

    for j in range(len(cat['stellarAstrometricSelfRepeatabilityRA'])):
        B = cat['stellarAstrometricSelfRepeatabilityRA'][j]
        if "dmL2AstroErr_RA" in B.metric_name.metric:
            name = str(dat['tract'])+"_"+ B.metric_name.metric[0]
            dmL1AstroErr.update({name:B.quantity.value})

# Make AM1 plot:
am1 = np.array([AM1[tract] for tract in AM1])
am1 = am1[np.isfinite(am1)]

am1Median = np.median(am1)

plt.hist(am1, bins=np.linspace(0, 30, 31), color=stars_color())
ylim = plt.ylim(0, 15)
plt.axvline(am1Median, linestyle='--',
            label='AM1 median = %.1f mas'%(am1Median),
            color=accent_color())
plt.ylim(ylim)
plt.ylabel('Number of tracts')
plt.xlabel('AM1 (mas)')
plt.legend()
plt.savefig('../figures/Astrometry_AM1.pdf')
plt.close()

# Make plot showing dmL1AstroErr and dmL2AstroErr
dml2AstroErr_arr = np.array(list(dmL2AstroErr.values()))
dml2AstroErr_arr = dml2AstroErr_arr[np.isfinite(dml2AstroErr_arr)]

dml2AstroErr_Median = np.median(dml2AstroErr_arr)

plt.hist(dml2AstroErr_arr, bins=np.linspace(0, 30, 31),
        label='Per-tract values after final calibration', color=stars_color())

dml1AstroErr_arr = np.array(list(dmL1AstroErr.values()))
dml1AstroErr_arr = dml1AstroErr_arr[np.isfinite(dml1AstroErr_arr)]

dml1AstroErr_Median = np.median(dml1AstroErr_arr)

plt.hist(dml1AstroErr_arr,
         bins=np.linspace(0, 30, 31), facecolor='None', histtype='step',
         edgecolor=accent_color(),
         label='Per-tract values after initial calibration')

dml2label = 'Median after final calibration = %.1f mas'%(dml2AstroErr_Median)
plt.axvline(dml2AstroErr_Median, linestyle='--', color='k',
            label=dml2label)
dml1label = 'Median after initial calibration = %.1f mas'%(dml1AstroErr_Median)
plt.axvline(dml1AstroErr_Median, linestyle='-.', color='k',
            label=dml1label)
#plt.ylim(ylim)
plt.ylabel('Number of tracts')
plt.xlabel('Mean repeatability in RA per tract (mas)')
plt.legend(fontsize=8)
plt.savefig('../figures/Astrometry_dmAstroErr.pdf')
plt.close()

# Gather AA1 data:
# This was run on DP1 data, but outside the DP1 pipeline,
# and is in a separate collection
butler2 = Butler("/repo/main", collections='u/csaunder/DM-50629')
registry2 = butler2.registry
datasetType = 'sourceTable_visit_gaia_dr3_20230707_match_astrom_metrics'
targetRefCatDeltaMetricsRefs = list(
    registry2.queryDatasets(datasetType, findFirst=True)
    )
aa1_ra = {}
aa1_dec = {}
for metricsRef in targetRefCatDeltaMetricsRefs:
    metrics = butler2.get(metricsRef)
    for metric in metrics['astromDiffMetrics']:
        if "AA1_RA" == metric.metric_name.metric:
            name = str(metricsRef.dataId['visit'])
            aa1_ra.update({name:metric.quantity.value})

        if "AA1_Dec" == metric.metric_name.metric:
            name = str(metricsRef.dataId['visit'])
            aa1_dec.update({name:metric.quantity.value})

# Make the figure
fig, subs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7, 4.5))
subs[0].hist(np.array(list(aa1_ra.values())), bins=30, color=stars_color())
subs[1].hist(np.array(list(aa1_dec.values())), bins=30, color=stars_color())
ra_median = np.median(np.array(list(aa1_ra.values())))
subs[0].axvline(ra_median, color=accent_color(), linestyle='--',
               label=f'Median for all visits={ra_median:.2f} mas')
dec_median = np.median(np.array(list(aa1_dec.values())))
subs[1].axvline(dec_median, color=accent_color(), linestyle='--',
               label=f'Median for all visits={dec_median:.2f} mas')
subs[0].legend(fontsize=8)
subs[1].legend(fontsize=8)
subs[0].set_xlim(-10, 10)
subs[0].set_ylabel('Number of tracts')
subs[0].set_xlabel('$\\delta RA$ (mas)')
subs[1].set_xlabel('$\\delta Dec$ (mas)')
fig.savefig('../figures/Astrometry_AA1.pdf')