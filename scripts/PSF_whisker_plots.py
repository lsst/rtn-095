#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import lsst.daf.butler as dafButler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lsst.utils.plotting import set_rubin_plotstyle, stars_color, accent_color
set_rubin_plotstyle()


# In[ ]:


def get_derived_moments(catalog, band):
    results = {}
    results['star_e1'] = (catalog[f'{band}_ixx'] - catalog[f'{band}_iyy'])/(catalog[f'{band}_ixx'] + catalog[f'{band}_iyy'])
    results['star_m41'] = catalog[f'{band}_hsm_moments_40'] - catalog[f'{band}_hsm_moments_04']
    
    results['star_e2'] = (2*catalog[f'{band}_ixy'])/(catalog[f'{band}_ixx'] + catalog[f'{band}_iyy'])
    results['star_m42'] = 2*(catalog[f'{band}_hsm_moments_13'] + catalog[f'{band}_hsm_moments_31'])

    results['star_t'] = (catalog[f'{band}_ixx'] + catalog[f'{band}_iyy'])
    results['star_rho4'] = catalog[f'{band}_hsm_moments_40'] + catalog[f'{band}_hsm_moments_04'] + 2*catalog[f'{band}_hsm_moments_22']

    
    results['psf_e1'] = (catalog[f'{band}_ixxPSF'] - catalog[f'{band}_iyyPSF'])/(catalog[f'{band}_ixxPSF'] + catalog[f'{band}_iyyPSF'])
    results['psf_m41'] = catalog[f'{band}_hsm_momentsPsf_40'] - catalog[f'{band}_hsm_momentsPsf_04']
    
    results['psf_e2'] = (2*catalog[f'{band}_ixyPSF'])/(catalog[f'{band}_ixxPSF'] + catalog[f'{band}_iyyPSF'])
    results['psf_m42'] = 2*(catalog[f'{band}_hsm_momentsPsf_13'] + catalog[f'{band}_hsm_momentsPsf_31'])

    results['psf_t'] = (catalog[f'{band}_ixxPSF'] + catalog[f'{band}_iyyPSF'])
    results['psf_rho4'] = catalog[f'{band}_hsm_momentsPsf_40'] + catalog[f'{band}_hsm_momentsPsf_04'] + 2*catalog[f'{band}_hsm_momentsPsf_22']

    
    return results

def produce_spin2_catalog(catalog, band, grid_size = 0.1 ):
        
    # subcat['true_spin2_m1'] = psf_new_moments[4]
    # subcat['residual_spin2_m1'] = psf_new_residuals[4]
    # subcat['true_spin2_m2'] = psf_new_moments[5]
    # subcat['residual_spin2_m2'] = psf_new_residuals[5]
    
    # subcat['residual_e1'] = psf_new_residuals[1]
    # subcat['residual_e2'] = psf_new_residuals[2]
        
    ra, dec = np.array(catalog[f'{band}_ra']), catalog[f'{band}_dec']
    ra_min = np.percentile(ra,1) - 1.0
    ra_max = np.percentile(ra,99) + 2.0
    
    dec_min = np.percentile(dec,1) - 1.0
    dec_max = np.percentile(dec,99) + 1.0
    
    x = np.arange(ra_min, ra_max, grid_size)
    y = np.arange(dec_min, dec_max, grid_size)
    # xv, yv = np.meshgrid(x, y)
    
    whisker_ra = []
    whisker_dec = []
    
    e1 = []
    e2 = []
    
    de1 = []
    de2 = []
    
    m1 = []
    m2 = []
    
    dm1 = []
    dm2 = []
    
    cell_length = []
    
    
    
    for i in range(len(x)-1):
        slice_catalog = catalog[catalog[f'{band}_ra']>x[i]]
        slice_catalog = slice_catalog[slice_catalog[f'{band}_ra']<x[i+1]]
        for j in range(len(y)-1):
            cell_catalog = slice_catalog[slice_catalog[f'{band}_dec']>y[j]]
            cell_catalog = cell_catalog[cell_catalog[f'{band}_dec']<y[j+1]]

            derived_cat = get_derived_moments(cell_catalog, band)
            
            psf_new_residual = [derived_cat['psf_t'] - derived_cat['star_t'],
                    derived_cat['psf_e1'] - derived_cat['star_e1'],
                    derived_cat['psf_e2'] - derived_cat['star_e2'],
                    derived_cat['psf_rho4'] - derived_cat['star_rho4'],
                    derived_cat['psf_m41'] - derived_cat['star_m41'], 
                    derived_cat['psf_m42'] - derived_cat['star_m42'] ]
            psf_new_moments = [derived_cat['psf_t'],
                                derived_cat['psf_e1'],
                                derived_cat['psf_e2'],
                                derived_cat['psf_rho4'],
                                derived_cat['psf_m41'], 
                                derived_cat['psf_m42']  ]
            
            cell_length.append(len(cell_catalog))
            
            whisker_ra.append((x[i]+x[i+1])/2)
            whisker_dec.append((y[j]+y[j+1])/2)

            e1.append(np.mean(psf_new_moments[1]))
            e2.append(np.mean(psf_new_moments[2]))

            de1.append(np.median(psf_new_residual[1]))
            de2.append(np.median(psf_new_residual[2]))

            m1.append(np.mean(psf_new_moments[4]))
            m2.append(np.mean(psf_new_moments[5]))

            dm1.append(np.median(psf_new_residual[4]))
            dm2.append(np.median(psf_new_residual[5]))
            
    data = {"ra":whisker_ra,"dec":whisker_dec,"e1":e1,"e2":e2,"de1":de1,"de2":de2,"m1":m1,"m2":m2,"dm1":dm1,"dm2":dm2, 'n':cell_length}
    
    res_df = pd.DataFrame(data=data)
    return res_df[res_df['e1']>-1]

def e1e2_to_ephi(e1,e2):
    
    e_comp = e1 + e2*1j
    
    phi = np.angle(e_comp)/2
    e = np.sqrt(e1**2+e2**2)
    
    return e,phi


def viewmap_compact(subcat, moment_name, f1, f2):
    #field_index = 0
    subcat = subcat[subcat['n']>5]

    ra, dec = np.array(subcat['ra']), np.array(subcat['dec'])

    ra_min = np.percentile(ra,0) - 0.2
    ra_max = np.percentile(ra,100) + 0.2

    dec_min = np.percentile(dec,0) - 0.5
    dec_max = np.percentile(dec,100) + 0.5

    ra_min = 52.1
    ra_max = 54.1
    dec_min = -29.1
    dec_max = -27.1

    single_size = 3*(ra_max - ra_min),  2*(dec_max - dec_min)

    text_x_pos = 0.8*ra_max + 0.2*ra_min
    text_y_pos = 0.9*dec_max + 0.1*dec_min

    if moment_name == "second":
        e1 = np.array(list(subcat['e1']))
        e2 = np.array(list(subcat['e2']))
        de1 = np.array(list(subcat['de1']))
        de2 = np.array(list(subcat['de2']))
        legend_name = r'$e_{\rm PSF}$'

    elif moment_name == "fourth":
        e1 = np.array(list(subcat['m1']))
        e2 = np.array(list(subcat['m2']))
        de1 = np.array(list(subcat['dm1']))
        de2 = np.array(list(subcat['dm2']))
        legend_name = r'$e^{(4)}_{\rm PSF}$'

    e,phi = e1e2_to_ephi(e1,e2)

    de, dphi = e1e2_to_ephi(de1,de2)

    ex = e*np.cos(phi)
    ey = e*np.sin(phi)

    dex = de*np.cos(dphi)
    dey = de*np.sin(dphi)

    #fig = plt.figure(figsize = (single_size[0],single_size[1]+6))
    #fig.subplots_adjust(left=0.17, bottom=0.1, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig = plt.figure(figsize=(5,12))
    ax = plt.subplot(2, 1, 1)
    ax.set_aspect("equal")
    ax1 = plt.subplot(2, 1, 2, sharex=ax)
    ax1.set_aspect("equal")

    for i in range(len(ra)):
        x_cen = ra[i]
        y_cen = dec[i]

        f = f1

        this_x = ex[i]
        this_y = ey[i]

        ax.plot([x_cen - this_x*f, x_cen+this_x*f],[y_cen - this_y*f, y_cen+this_y*f], color=stars_color())

    ax.set_xlim(ra_min, ra_max)
    ax.set_ylim(dec_min, dec_max)
    ax.tick_params("x", labelbottom=False)
    ax.set_ylabel("Dec. (deg)")

    # plt.title('PSF $M{}$ Moment of {}'.format(moment_name, field_name))
    plt.text(0.26, 0.93, legend_name + ' Truth', transform=fig.transFigure, fontsize=10)
    ax.plot([text_x_pos , text_x_pos + 0.2], [text_y_pos - 0.11, text_y_pos - 0.11], color=accent_color())
    plt.text(0.75, 0.92, str(0.1/f), color=accent_color(), transform=fig.transFigure, fontsize=10)

    for i in range(len(ra)):
        x_cen = ra[i]
        y_cen = dec[i]

        f = f2

        this_x = dex[i]
        this_y = dey[i]

        ax1.plot([x_cen - this_x*f, x_cen+this_x*f],[y_cen - this_y*f, y_cen+this_y*f], color=stars_color())

    ax1.set_ylim(dec_min, dec_max)
    ax1.set_xlabel("R. A. (deg)")
    ax1.set_ylabel("Dec. (deg)")
    #plt.title('PSF Residual Moment '+str(pq[0])+ ","+ str(pq[1])+ ' Map')
    #plt.text(0.03, 0.4, "Dec. (deg)", transform=fig.transFigure, rotation="vertical")
    plt.text(0.26, 0.49, legend_name + ' Residual', transform=fig.transFigure, fontsize=10)
    ax1.plot([text_x_pos , text_x_pos + 0.2], [text_y_pos-0.11,text_y_pos-0.11], color=accent_color())
    plt.text(0.75, 0.49, str(0.1/f), color=accent_color(), transform=fig.transFigure, fontsize=10)

    filename = 'psf_{}_whisker.pdf'.format(moment_name)
    plt.draw()
    fig.subplots_adjust(left=0.2, hspace=0, wspace=0, top=0.98, right=0.9)
    plt.savefig(filename)
    plt.show()


# In[ ]:


collection = 'LSSTComCam/runs/DRP/DP1/v29_0_0/DM-50260'

b_embargo = dafButler.Butler("/repo/dp1", collections=collection)
registry = b_embargo.registry


band = 'r'

col_names = [f'{band}_ra', f'{band}_dec',  f'{band}_ixx',f'{band}_iyy',f'{band}_ixy', f'{band}_ixxPSF',f'{band}_iyyPSF',f'{band}_ixyPSF',
             f'{band}_extendedness', f'{band}_sizeExtendedness', f'{band}_blendedness', f'{band}_cModelFlux',
             f'{band}_calib_psf_candidate', f'{band}_calib_psf_reserved', f'{band}_calib_psf_used',
             f'{band}_hsm_moments_30', f'{band}_hsm_momentsPsf_30', f'{band}_hsm_moments_21', f'{band}_hsm_momentsPsf_21',
             f'{band}_hsm_moments_12', f'{band}_hsm_momentsPsf_12',f'{band}_hsm_moments_03', f'{band}_hsm_momentsPsf_03',
             f'{band}_hsm_moments_40', f'{band}_hsm_momentsPsf_40',f'{band}_hsm_moments_31', f'{band}_hsm_momentsPsf_31',
             f'{band}_hsm_moments_22', f'{band}_hsm_momentsPsf_22',f'{band}_hsm_moments_13', f'{band}_hsm_momentsPsf_13',
             f'{band}_hsm_moments_04', f'{band}_hsm_momentsPsf_04',f'{band}_cModelFlux', f'{band}_cModelFluxErr',
             f'{band}_invalidPsfFlag', 'refExtendedness', 'detect_isIsolated' ]




# In[ ]:


tract_id_list = [5063,4849,4848]


# In[ ]:


data_frame_list = []

for tract_id in tract_id_list:
    dataId = {'band': band, 'tract': tract_id, 'skymap': 'lsst_cells_v1'}
    objects = b_embargo.get('object', dataId = dataId, parameters = {"columns": col_names})
    df = objects.to_pandas()
    data_frame_list.append(df)


# In[ ]:


catalog = pd.concat(data_frame_list)


# In[ ]:


catalog_psf = catalog[catalog[f'{band}_calib_psf_used'] == True]
catalog_reserve = catalog[catalog[f'{band}_calib_psf_reserved'] == True]


# In[ ]:


PSF_stars_cat = get_derived_moments(catalog_psf, band)
reserved_stars_cat = get_derived_moments(catalog_reserve, band)


# In[ ]:


psf_new_residual = [PSF_stars_cat['psf_t'] - PSF_stars_cat['star_t'],
                    PSF_stars_cat['psf_e1'] - PSF_stars_cat['star_e1'],
                    PSF_stars_cat['psf_e2'] - PSF_stars_cat['star_e2'],
                    PSF_stars_cat['psf_rho4'] - PSF_stars_cat['star_rho4'],
                    PSF_stars_cat['psf_m41'] - PSF_stars_cat['star_m41'], 
                    PSF_stars_cat['psf_m42'] - PSF_stars_cat['star_m42'] ]
psf_new_moments = [PSF_stars_cat['psf_t'],
                    PSF_stars_cat['psf_e1'],
                    PSF_stars_cat['psf_e2'],
                    PSF_stars_cat['psf_rho4'],
                    PSF_stars_cat['psf_m41'], 
                    PSF_stars_cat['psf_m42']  ]


# In[ ]:


nonpsf_new_residual = [reserved_stars_cat['psf_t'] - reserved_stars_cat['star_t'],
                    reserved_stars_cat['psf_e1'] - reserved_stars_cat['star_e1'],
                    reserved_stars_cat['psf_e2'] - reserved_stars_cat['star_e2'],
                    reserved_stars_cat['psf_rho4'] - reserved_stars_cat['star_rho4'],
                    reserved_stars_cat['psf_m41'] - reserved_stars_cat['star_m41'], 
                    reserved_stars_cat['psf_m42'] - reserved_stars_cat['star_m42'] ]
nonpsf_new_moments = [reserved_stars_cat['psf_t'],
                    reserved_stars_cat['psf_e1'],
                    reserved_stars_cat['psf_e2'],
                    reserved_stars_cat['psf_rho4'],
                    reserved_stars_cat['psf_m41'], 
                    reserved_stars_cat['psf_m42']]


# In[ ]:


label = [r'$\Delta T_{\rm PSF}$', r'$\Delta e_{\rm PSF,1}$',  r'$\Delta e_{\rm PSF,2}$', r'$\Delta \rho^{(4)}_{\rm PSF}$',r'$\Delta M^{(4)}_{\rm PSF,1}$',  r'$\Delta M^{(4)}_{\rm PSF,2}$']


# In[ ]:


whisker_psf = produce_spin2_catalog(catalog_psf, band)


# In[ ]:


viewmap_compact(whisker_psf, "second", 0.2, 10)


# In[ ]:


viewmap_compact(whisker_psf, "fourth", 1, 20)


# In[ ]:




