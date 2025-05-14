#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import treegp
# treegp is part of rubin-env 
# but need version 1.3.1 to
# compute some stats later on.
print(treegp.__version__)
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from scipy.stats import binned_statistic_2d
from lsst.utils.plotting import set_rubin_plotstyle, stars_cmap, stars_color, accent_color, divergent_cmap
set_rubin_plotstyle()
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pathEffects
from matplotlib import gridspec
from lsst.analysis.tools.actions.plot.plotUtils import sortAllArrays

from lsst.daf.butler import Butler
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstComCam

# %matplotlib widget


# I guess this cmap or the other can in general
# give better idea of physical variation on the
# focal plane from past experience.
# CMAP = plt.cm.inferno
#CMAP = plt.cm.seismic


def plot_residuals(u, v, du, dv, xie, xib, logr, visitID = '2024120700529', tract= None, band = None, fix=True):

    LIM = 40
    point_size = 0.5
    RESDIUAL_LIM = np.std(du[np.isfinite(du)])
    RESDIUAL_LIM = 25
    
    fig = plt.figure(figsize=(10,3))
    gs = gridspec.GridSpec(10, 40)
    ax = fig.add_subplot(gs[:, :10])
    cax = fig.add_subplot(gs[:, 10])
    ax1 = fig.add_subplot(gs[:, 14:24])
    cax1 = fig.add_subplot(gs[:, 24])
    ax2 = fig.add_subplot(gs[:, 31:])
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0.22, left=0.11, right=0.91)
    ax.set_aspect("equal")
    us = u * 60 - 2
    vs = v * 60 - 1
    us_sorted, vs_sorted, du_sorted = sortAllArrays([us, vs, du], sortArrayIndex=2)
    data = ax.scatter(us_sorted, vs_sorted, s=point_size, c=du_sorted, vmin=-RESDIUAL_LIM, vmax=RESDIUAL_LIM,
                      cmap=divergent_cmap())
    if not fix: 
        ax.set_aspect('equal')
    else:
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)

    ax.get_figure().colorbar(data, cax=cax, orientation="vertical")
    text = cax.text(0.5, 0.5, "du (mas)", color="k", rotation="vertical", transform=cax.transAxes,
                    ha="center", va="center", fontsize=10)
    text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])

    ax.set_xlabel('u (arcmin)')
    ax.set_ylabel('v (arcmin)')

    ax1.set_aspect("equal")
    us_sorted, vs_sorted, dv_sorted = sortAllArrays([us, vs, dv], sortArrayIndex=2)

    data1 = ax1.scatter(us_sorted, vs_sorted, s=point_size, c=dv_sorted, vmin=-RESDIUAL_LIM, vmax=RESDIUAL_LIM,
                        cmap=divergent_cmap())
    #xBinEdges = np.linspace(-LIM, LIM, 101)
    #yBinEdges = np.linspace(-LIM, LIM, 101)
    #binnedStats, xEdges, yEdges, binNums = binned_statistic_2d(u*60-2, v*60-1, dv, statistic="median",
    #                                                           bins=(xBinEdges, yBinEdges))
    #data1 = ax1.imshow(binnedStats.T, cmap=divergent_cmap(),
    #                  extent=[xEdges[0], xEdges[-1], yEdges[-1], yEdges[0]],
    #                  vmin=-RESDIUAL_LIM, vmax=RESDIUAL_LIM)
    if not fix: 
        ax1.set_aspect('equal')
    else:
        ax1.set_xlim(-LIM, LIM)
        ax1.set_ylim(-LIM, LIM)

    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="12%", pad=0)
    ax1.get_figure().colorbar(data1, cax=cax1, orientation="vertical")
    text = cax1.text(0.5, 0.5, "dv (mas)", color="k", rotation="vertical", transform=cax1.transAxes,
                    ha="center", va="center", fontsize=10)
    text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])

    ax1.set_xlabel('u (arcmin)')
    ax1.tick_params("y", labelleft=False)

    ax2.scatter(np.exp(logr) * 60, xie, c=stars_color(), label='E-mode', s=15)
    ax2.scatter(np.exp(logr) * 60, xib, c=accent_color(), label='B-mode', s=15)
    ax2.set_xscale('log')
    ax2.axhline(0, color='k', linestyle="--", zorder=-1)
    #ax2.set_xlim(xlim)
    ax2.set_ylabel(r'$\xi_{E/B}$ (mas$^2$)')
    ax2.set_xlabel(r'$\Delta \theta$ (arcmin)')
    #plt.suptitle(f'Tract: {tract} | Band: {band} | Visit: {visitID}')


camera = LsstComCam.getCamera()
def pixel_to_focal(x, y, det):
    """
    Parameters
    ----------
    x, y : array
        Pixel coordinates.
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    fpx, fpy : array
        Focal plane position in millimeters in DVCS
        See https://lse-349.lsst.io/
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    
    return fpx.ravel(), fpy.ravel()


def plot_currentID():
    repo = "/repo/main"
    collection = "LSSTComCam/runs/DRP/DP1/v29_0_0_rc6/DM-50098"

    butler = Butler(repo, collections=collection)

    info = list(butler.registry.queryDatasets('gbdesAstrometricFit_fitStars'))

    dataID = []

    for run in info:
        if run.dataId['skymap'] == 'lsst_cells_v1':
            dataID.append(run.dataId)


    # just want to plot a particular visit in next figure
    for i in range(len(dataID)):
        if dataID[i]['tract'] == 5525 and dataID[i]['physical_filter'] == 'r_03':
            print(i)

    i = 120
    cat  = butler.get('gbdesAstrometricFit_fitStars', dataID[i]).to_pydict()
    visits = cat['exposureName']
    visitID = list(set(visits))

    currentID = '2024120200359' #visitID[0]
    if currentID != 'REFERENCE':
        tract = dataID[i]['tract']
        band = dataID[i]['band']
        Filter = (np.array(visits) == currentID)
        u = np.array(cat['xworld'])[Filter]
        v = np.array(cat['yworld'])[Filter]
        x = np.array(cat['xpix'])[Filter]
        y = np.array(cat['ypix'])[Filter]
        ccdId = np.array(cat['deviceName'])[Filter]
        du = np.array(cat['xresw'])[Filter]
        dv = np.array(cat['yresw'])[Filter]
        dx = np.array(cat['xrespix'])[Filter]
        dy = np.array(cat['yrespix'])[Filter]
        #print(u)
        xie, xib, logr = treegp.comp_eb_treecorr(u, v, du, dv, rmin=20/3600, rmax=0.6, dlogr=0.3)
        plot_residuals(u, v, du, dv, xie, xib, logr, visitID = currentID, tract= dataID[i]['tract'], band = dataID[i]['band'], fix=True)
        plt.savefig('Astrometry_'+currentID+'.pdf')
        plt.show()

    currentID = '2024120700527' #visitID[0]
    if currentID != 'REFERENCE':
        tract = dataID[i]['tract']
        band = dataID[i]['band']
        Filter = (np.array(visits) == currentID)
        u = np.array(cat['xworld'])[Filter]
        v = np.array(cat['yworld'])[Filter]
        x = np.array(cat['xpix'])[Filter]
        y = np.array(cat['ypix'])[Filter]
        ccdId = np.array(cat['deviceName'])[Filter]
        du = np.array(cat['xresw'])[Filter]
        dv = np.array(cat['yresw'])[Filter]
        dx = np.array(cat['xrespix'])[Filter]
        dy = np.array(cat['yrespix'])[Filter]
        #print(u)
        xie, xib, logr = treegp.comp_eb_treecorr(u, v, du, dv, rmin=20/3600, rmax=0.6, dlogr=0.3)
        plot_residuals(u, v, du, dv, xie, xib, logr, visitID = currentID, tract= dataID[i]['tract'], band = dataID[i]['band'], fix=True)
        plt.savefig('Astrometry_'+currentID+'.pdf')
        plt.show()

WRITE = False

if WRITE: 

    visitSet = set()
    dicAll = {}
    for i in tqdm(range(len(dataID))):
        cat  = butler.get('gbdesAstrometricFit_fitStars', dataID[i]).to_pydict()
        visits = np.array(cat['exposureName'])
        visitID = set(visits)
        I = 0
        for visit in visitID:
            currentID = visit
            if currentID != 'REFERENCE' and currentID not in visitSet:
                visitSet.update({currentID})
                tract = dataID[i]['tract']
                band = dataID[i]['band']
                Filter = (visits == currentID)
                u = np.array(cat['xworld'])[Filter]
                v = np.array(cat['yworld'])[Filter]
                x = np.array(cat['xpix'])[Filter]
                y = np.array(cat['ypix'])[Filter]
                ccdId = np.array(cat['deviceName'])[Filter]
                du = np.array(cat['xresw'])[Filter]
                dv = np.array(cat['yresw'])[Filter]
                dx = np.array(cat['xrespix'])[Filter]
                dy = np.array(cat['yrespix'])[Filter]
                xie, xib, logr = treegp.comp_eb_treecorr(u, v, du, dv, rmin=20/3600, rmax=0.6, dlogr=0.3)

                dicAll.update({
                    currentID: {
                        'u': u,
                        'v': v,
                        'x': x,
                        'y': y,
                        'du': du,
                        'dv': dv,
                        'dx': dx,
                        'dy': dy,
                        'ccdId': ccdId,
                        'xie': xie,
                        'xib': xib,
                        'logr': logr,
                        'band': band,
                    }
                })
            I += 1

    fileAstro = open('astroResidual.pkl', 'wb')
    pickle.dump(dicAll, fileAstro)
    fileAstro.close()
    dic = dicAll

else:
    dic = pickle.load(open('astroResidual.pkl', 'rb'))

# mean E/B mode

def EB_mode_plot(tqdm):
    E = []
    B = []
    logr = None

    for visit in tqdm(dic):
        E.append(dic[visit]['xie'])
        B.append(dic[visit]['xib'])
        logr = dic[visit]['logr']

    E = np.array(E)
    B = np.array(B)

    plt.scatter(np.exp(logr) * 60, np.mean(E, axis=0), c=stars_color(), label='mean E-mode')
    plt.scatter(np.exp(logr) * 60, np.mean(B, axis=0), c=accent_color(), label='mean B-mode')
    plt.xscale('log')
    xlim = plt.xlim()
    plt.plot(xlim, [0,0], 'k--', zorder=-1)
    plt.xlim(xlim)
    plt.ylim(-5,100)
    plt.ylabel(r'$\xi_{E/B}$ (mas$^2$)')
    plt.xlabel(r'$\Delta \theta$ (arcmin)')
    plt.legend()
    plt.savefig('Astrometry_EB_mode.pdf')
    plt.close()


def fov_plot(tqdm):
    bin_spacing = 100

    meanify = {}
    for i in range(9):
        meanify.update({i: treegp.meanify(bin_spacing=bin_spacing, statistics='median')})

    for visit in tqdm(dic):
        coord = np.array([dic[visit]['x'], dic[visit]['y']]).T
        for i in range(9):
            filtering = (dic[visit]['ccdId'] == str(i))
            meanify[i].add_field(coord[filtering], dic[visit]['dx'][filtering])

    for i in range(9):
        meanify[i].meanify()

    CMAP = stars_cmap(single_color=False)
    S = 3.5
    MAX = 0.01

    fig, (ax, ax1) = plt.subplots(1, 2, sharey=True, figsize=(7, 3))
    fig.subplots_adjust(hspace=0.0, wspace=0.25, right=0.9, bottom=0.23)
    ax.set_aspect("equal")
    ax1.set_aspect("equal")

    for i in range(9):
        x, y = np.meshgrid(meanify[i]._xedge, meanify[i]._yedge)
        nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
        x = x.reshape(nBin0*nBin1)
        y = y.reshape(nBin0*nBin1)
        x, y = pixel_to_focal(x, y, camera[i])
        x = x.reshape((nBin0, nBin1))
        y = y.reshape((nBin0, nBin1))
        data = ax.pcolormesh(x, y , meanify[i]._average, vmin=-MAX, vmax=MAX, cmap=CMAP)

    ax.set_xlabel('x (mm)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="12%", pad=0)
    ax.get_figure().colorbar(data, cax=cax, orientation="vertical")
    text = cax.text(0.5, 0.5, '$\\delta x$ (pixel)', color="k", rotation="vertical", transform=cax.transAxes,
                    ha="center", va="center", fontsize=10)
    text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])


    meanify = {}
    for i in range(9):
        meanify.update({i: treegp.meanify(bin_spacing=bin_spacing, statistics='median')})

    for visit in tqdm(dic):
        coord = np.array([dic[visit]['x'], dic[visit]['y']]).T
        for i in range(9):
            filtering = (dic[visit]['ccdId'] == str(i))
            meanify[i].add_field(coord[filtering], dic[visit]['dy'][filtering])

    for i in range(9):
        meanify[i].meanify()

    for i in range(9):

        x, y = np.meshgrid(meanify[i]._xedge, meanify[i]._yedge)
        nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
        x = x.reshape(nBin0*nBin1)
        y = y.reshape(nBin0*nBin1)
        x, y = pixel_to_focal(x, y, camera[i])
        x = x.reshape((nBin0, nBin1))
        y = y.reshape((nBin0, nBin1))
        data = ax1.pcolormesh(x, y , meanify[i]._average, vmin=-MAX, vmax=MAX, cmap=CMAP)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="12%", pad=0)
    ax1.get_figure().colorbar(data, cax=cax, orientation="vertical")
    text = cax.text(0.5, 0.5, '$\\delta y$ (pixel)', color="k", rotation="vertical", transform=cax.transAxes,
                    ha="center", va="center", fontsize=10)
    text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])

    ax1.tick_params("y", labelleft=False)
    ax1.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    plt.savefig('Astrometry_FoV.pdf')
    plt.show()


def CCD_plot(tqdm):
# dx

    bin_spacing = 50

    meanify = treegp.meanify(bin_spacing=bin_spacing, statistics='median')

    for visit in tqdm(dic):
        coord = np.array([dic[visit]['x'], dic[visit]['y']]).T
        meanify.add_field(coord, dic[visit]['dx'])

    meanify.meanify()

    CMAP = stars_cmap()
    S = 12
    MAX = 0.01

    fig, (ax, ax1) = plt.subplots(1, 2, sharey=True, figsize=(7, 3))
    fig.subplots_adjust(hspace=0.0, wspace=0.25, right=0.9, bottom=0.23)
    ax.set_aspect("equal")
    ax1.set_aspect("equal")

    x, y = np.meshgrid(meanify._xedge, meanify._yedge)
    nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
    x = x.reshape(nBin0*nBin1)
    y = y.reshape(nBin0*nBin1)
    x, y = pixel_to_focal(x, y, camera[4])
    x = x.reshape((nBin0, nBin1))
    y = y.reshape((nBin0, nBin1))
    data = ax.pcolormesh(x, y , meanify._average, vmin=-MAX, vmax=MAX, cmap=CMAP)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="12%", pad=0)
    ax.get_figure().colorbar(data, cax=cax, orientation="vertical")
    text = cax.text(0.5, 0.5, '$\\delta x$ (pixel)', color="k", rotation="vertical", transform=cax.transAxes,
                    ha="center", va="center", fontsize=10)
    text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    meanify = treegp.meanify(bin_spacing=bin_spacing, statistics='median')

    for visit in tqdm(dic):
        coord = np.array([dic[visit]['x'], dic[visit]['y']]).T
        meanify.add_field(coord, dic[visit]['dy'])

    meanify.meanify()

    x, y = np.meshgrid(meanify._xedge, meanify._yedge)
    nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
    x = x.reshape(nBin0*nBin1)
    y = y.reshape(nBin0*nBin1)
    x, y = pixel_to_focal(x, y, camera[4])
    x = x.reshape((nBin0, nBin1))
    y = y.reshape((nBin0, nBin1))
    data = ax1.pcolormesh(x, y , meanify._average, vmin=-MAX, vmax=MAX, cmap=CMAP)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="12%", pad=0)
    ax1.get_figure().colorbar(data, cax=cax, orientation="vertical")
    text = cax.text(0.5, 0.5, '$\\delta y$ (pixel)', color="k", rotation="vertical", transform=cax.transAxes,
                    ha="center", va="center", fontsize=10)
    text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])

    ax1.set_xlabel('x (mm)')
    plt.savefig('Astrometry_CCD.pdf')
    plt.close()


def AM1_plot():
    # AM1 
    repo = "/repo/main"
    collection = "LSSTComCam/runs/DRP/DP1/v29_0_0_rc6/DM-50098"

    catalog = "recalibrated_star_association_metrics"
    butler = Butler(repo, collections=collection)

    info = list(butler.registry.queryDatasets(catalog))
    dataID = []

    for run in info:
        dataID.append(run.dataId)
    dataID = list(set(dataID))


    AM1 = {}

    for dat in dataID:
        try:
            cat = butler.get(catalog, dat)
                    
            for i in range(len(cat['stellarAstrometricRepeatability1'])):
                A = cat['stellarAstrometricRepeatability1'][i]
            
                if "AM1" in A.metric_name.metric:
                    name = str(dat['tract'])+"_"+ A.metric_name.metric[0]
                    AM1.update({name:A.quantity.value})
        except:
            # print(f"skyp {dat}")
            not_ok = f"skyp {dat}"


# In[ ]:


    am1 = np.array([AM1[tract] for tract in AM1])
    am1 = am1[np.isfinite(am1)]

    am1Median = np.median(am1)

    plt.hist(am1, bins=np.linspace(0, 30, 31), color=stars_color())
    ylim = plt.ylim(0, 15)
    plt.plot([am1Median, am1Median], ylim, ls='--', color=accent_color(), label='AM1 median = %.1f mas'%(am1Median))
    plt.ylim(ylim)
    plt.ylabel('# of tracts')
    plt.xlabel('AM1 (mas)')
    plt.legend()
    plt.savefig('Astrometry_AM1.pdf')
    plt.close()

#fov_plot(tqdm)
#EB_mode_plot(tqdm)
#plot_currentID()
AM1_plot()
#CCD_plot(tqdm)
