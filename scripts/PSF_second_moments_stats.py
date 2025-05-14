#!/usr/bin/env python
# coding: utf-8

# # Analysis / plot for DP1 of PSF and second moment stats

# In[ ]:


from lsst.daf.butler import Butler
import numpy as np
from astropy.table import Table
import treegp
# treegp is part of rubin-env 
# but need version 1.3.1 to
# compute some stats later on.
print(treegp.__version__)
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patheffects as pathEffects
from lsst.utils.plotting import set_rubin_plotstyle, stars_cmap
set_rubin_plotstyle()

import pickle

import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstComCam


# Define transformation from pixel to focal plane coordinates for ComCam.

# In[ ]:


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


# To have 1:1 comparaison, I computed at the same stage, a collection using PSFex and Piff with a second order polynomial interpolation in order to compare to the final DP1 PSF. 

# In[ ]:


repo = "/repo/main"
collection = {
    "PSFex": "u/leget/comcam/DP1_paper/PSFex",
    "Piff poly order: 2": "u/leget/comcam/DP1_paper/PiffOrder2",
    "Piff poly order: 4": "LSSTComCam/runs/DRP/DP1/v29_0_0_rc6/DM-50098",
}

butler = Butler(repo, collections=collection["Piff poly order: 4"])
sourceTable_visit_dsrs = list(butler.registry.queryDatasets("refit_psf_star"))
visit_ids = []

for dsr in sourceTable_visit_dsrs:
    visit_ids.append(dsr.dataId["visit"])

print(len(visit_ids))
print(len(set(visit_ids)))


# Just going through all the visist and gather PSF statistics.

# In[ ]:


WRITE = False

columns_name = [
    'slot_Shape_xx', 'slot_Shape_yy', 'slot_Shape_xy',
    'slot_PsfShape_xx', 'slot_PsfShape_xy', 'slot_PsfShape_yy',
    'coord_ra', 'coord_dec', 'slot_Centroid_x', 'slot_Centroid_y',
    'detector',
]


if WRITE: 

    master_dic = {}

    for PSF in collection:
        dic = {}
        butler = Butler(repo, collections=collection[PSF])
    
        for visit in tqdm(visit_ids):
    
            if True: 
            
                finalized_src_table = butler.get("refit_psf_star", visit=visit, parameters={"columns": columns_name})
                #if 'DRP/DP1' in collection[PSF]:
                visit_summary = butler.get("preliminary_visit_summary", visit=visit)
                band = visit_summary[0]['band']
                
                
                table = finalized_src_table
                table['ixx_src'] = table['slot_Shape_xx']
                table['ixy_src'] = table['slot_Shape_xy']
                table['iyy_src'] = table['slot_Shape_yy']
                
                table['ixx_psf'] = table['slot_PsfShape_xx']
                table['ixy_psf'] = table['slot_PsfShape_xy']
                table['iyy_psf'] = table['slot_PsfShape_yy']
                
                table['T_src'] = table['ixx_src'] + table['iyy_src']
                table['e1_src'] = (table['ixx_src'] - table['iyy_src']) / table['T_src']
                table['e2_src'] = 2*table['ixy_src'] / table['T_src']
                
                table['T_psf'] = table['ixx_psf'] + table['iyy_psf']
                table['e1_psf'] = (table['ixx_psf'] - table['iyy_psf']) / table['T_psf']
                table['e2_psf'] = 2*table['ixy_psf'] / table['T_psf']
        
                dic.update({
                    visit: {
                        'T_src': np.array(table['T_src']),
                        'e1_src': np.array(table['e1_src']),
                        'e2_src': np.array(table['e2_src']),
                        'dT_T': np.array((table['T_src'] - table['T_psf']) / table['T_src']),
                        'de1': np.array(table['e1_src'] - table['e1_psf']),
                        'de2': np.array(table['e2_src'] - table['e2_psf']),
                        'ra': np.array(table['coord_ra']),
                        'dec': np.array(table['coord_dec']),
                        'x': np.array(table['slot_Centroid_x']),
                        'y': np.array(table['slot_Centroid_y']),
                        'detector': np.array(table['detector']),
                        'band': band,
                    }
                })
    
            else:
                dic.update({
                    visit: None,
                })
        master_dic.update({PSF: {"dic": dic, "collection": collection[PSF]}})
    
    f = open('master_dic.pkl', 'wb')
    pickle.dump(master_dic, f)
    f.close()
else:

    master_dic = pickle.load(open('master_dic.pkl', 'rb'))


# Plot second moment stats across visit and project into FoV.

# In[ ]:


def get_fov_plot_distrib(
    master_dic,
    PSF = "Piff",
    key_second_moment='dT_T',
    bin_spacing = 200,
    CMAP = stars_cmap(single_color=True),
    MAX = 0.01,
    MIN = None,
    auto=False,
    camera=LsstComCam.getCamera(), 
    colorlabel=None,
    title=None,
    namefig=None):

    if MIN is None:
        MIN = -MAX
    
    meanify = {}
    
    for i in range(9):
        meanify.update({i: treegp.meanify(bin_spacing=bin_spacing, statistics='median')})
    
    for visit in master_dic[PSF]['dic']:
        if master_dic[PSF]['dic'][visit] is not None:
            for i in range(9):
                filtering = (master_dic[PSF]['dic'][visit]["detector"] == i)
                coord = np.array([master_dic[PSF]['dic'][visit]['x'], master_dic[PSF]['dic'][visit]['y']]).T
                meanify[i].add_field(coord[filtering], master_dic[PSF]['dic'][visit][key_second_moment][filtering])
        else:
            print(visit)
    
    
    for i in range(9):
        meanify[i].meanify()

    if auto:
        M = []
        for i in range(9):
            M.append(meanify[i]._average)
        M = np.concatenate(M)
        MEAN = np.mean(M[np.isfinite(M)])
        STD = np.std(M[np.isfinite(M)])
        MIN = MEAN - 2 * STD
        MAX = MEAN + 2 * STD
    else:
        MIN = -MAX
        MAX = MAX
    
    #plt.figure(figsize=(10,6))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    for i in range(9):
        x, y = np.meshgrid(meanify[i]._xedge, meanify[i]._yedge)
        nBin0, nBin1 = np.shape(x)[0], np.shape(x)[1]
        x = x.reshape(nBin0*nBin1)
        y = y.reshape(nBin0*nBin1)
        x, y = pixel_to_focal(x, y, camera[i])
        x = x.reshape((nBin0, nBin1))
        y = y.reshape((nBin0, nBin1))
        plotOut = plt.pcolormesh(x, y , meanify[i]._average, vmin=MIN, vmax=MAX, cmap=CMAP)

    if colorlabel is None:
        colorlabel = key_second_moment
    axBbox = ax.get_position()
    cax = fig.add_axes([axBbox.x1, axBbox.y0, 0.04, axBbox.y1 - axBbox.y0])
    fig.colorbar(plotOut, cax=cax)
    text = cax.text(0.5, 0.5, colorlabel, color="k", rotation="vertical", transform=cax.transAxes, ha="center", va="center", fontsize=10)
    text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()])

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    if title is None:
        title = f"collection = {collection[PSF]}\nPSF model: {PSF}" 
    ax.set_title(title)
    if namefig is None:
        namefig = f'{key_second_moment}_2d_{PSF}_1.pdf'
    plt.savefig(f'{namefig}')

TITLE = ['PSFex | polynomial order: 2', 'Piff | polynomial order: 2', 'Piff | polynomial order: 4']
NAMEFIG = ["dT_T_PSFEx_poly_order_2.pdf","dT_T_Piff_poly_order_2.pdf", "dT_T_Piff_poly_order_4.pdf"]
colorlab = '$\\left<\\left(T_{\\text{PSF}} - T_{\\text{model}}\\right) \\ / \\ T_{\\text{PSF}}\\right>$'

for PSF, title, name in zip(master_dic, TITLE, NAMEFIG):
    for key in ['dT_T']:
        get_fov_plot_distrib(master_dic, PSF = PSF, key_second_moment=key, 
                             bin_spacing = 120, CMAP = stars_cmap(single_color=True), MAX=0.005,
                             colorlabel=colorlab, 
                             title= title, namefig=name)


# Get key number for table in PSF section.

# In[ ]:


def get_stat(
    master_dic,
    PSF = "Piff",
    key_second_moment='dT_T',
    second_key=None, 
    bands = ['g', 'r', 'i', 'z', 'y']):

    
    meanify = treegp.meanify(bin_spacing=100, statistics='median')
    
    for visit in master_dic[PSF]['dic']:
        if master_dic[PSF]['dic'][visit] is not None:
            if master_dic[PSF]['dic'][visit]['band'] in bands:
                coord = np.array([master_dic[PSF]['dic'][visit]['x'], master_dic[PSF]['dic'][visit]['y']]).T
                if second_key is None:            
                    meanify.add_field(coord, master_dic[PSF]['dic'][visit][key_second_moment])
                else:
                    field = np.sqrt(master_dic[PSF]['dic'][visit][key_second_moment]**2 + master_dic[PSF]['dic'][visit][second_key]**2)
                    meanify.add_field(coord, field)
    params = np.concatenate(meanify.params)
    ff = np.isfinite(params)
    s = np.std(params[ff])
    if second_key is None:
        second_key = ""
    print(PSF, key_second_moment+second_key, "%.5f"%(np.mean(params[ff])), "%.5f"%(s / np.sqrt(np.sum(ff))), s)


for key in ['T_src', 'e1_src', 'e2_src']:
    get_stat(master_dic, PSF = "Piff poly order: 4", key_second_moment=key,  bands = ['u', 'g', 'r', 'i', 'z', 'y'])
    

for key in ['dT_T', 'de1', 'de2']:
    for PSF in master_dic:
        get_stat(master_dic, PSF = PSF, key_second_moment=key,  bands = ['u', 'g', 'r', 'i', 'z', 'y'])

get_stat(master_dic, PSF = "Piff poly order: 4", key_second_moment='e1_src', second_key='e2_src',  bands = ['u', 'g', 'r', 'i', 'z', 'y'])

