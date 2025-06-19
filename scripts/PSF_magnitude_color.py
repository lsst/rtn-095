#!/usr/bin/env python
# coding: utf-8

# # Analysis / plot to compare PSF residuals and photometry / color

# In[ ]:


from lsst.daf.butler import Butler
import numpy as np
from astropy.table import Table
from smatch.matcher import Matcher
from tqdm import tqdm
import astropy.units as u
import os

import matplotlib.pyplot as plt
from lsst.utils.plotting import publication_plots, stars_color, accent_color
publication_plots.set_rubin_plotstyle()
import pickle

import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstComCam


from scipy.stats import binned_statistic
import fitsio
import os
from astropy.stats import median_absolute_deviation as mad_astropy
import copy


# Some code to do 1d histogram

# In[ ]:


def biweight_median(sample, CSTD=6.):
    """
    Median with outlier rejection using mad clipping.
    Using the biweight described in Beers 1990 (Used originaly
    for finding galaxy clusters redshfit).

    :param sample: 1d numpy array. The sample where you want
                   to compute the median with outlier rejection.
    :param CSTD:   float. Constant used in the algorithm of the
                   Beers 1990. [default: 6.]
    """
    M = np.median(sample)
    iterate = [copy.deepcopy(M)]
    mu = (sample-M) / (CSTD*mad_astropy(sample))
    Filtre = (abs(mu)<1)
    up = (sample-M) * ((1.-mu**2)**2)
    down = (1.-mu**2)**2
    M += np.sum(up[Filtre])/np.sum(down[Filtre])

    iterate.append(copy.deepcopy(M))
    i = 1
    while abs((iterate[i-1]-iterate[i])/iterate[i])<0.001:
        mu = (sample-M) / (CSTD*mad_astropy(sample))
        Filtre = (abs(mu)<1)
        up = (sample-M) * ((1.-mu**2)**2)
        down = (1.-mu**2)**2
        M += np.sum(up[Filtre])/np.sum(down[Filtre])
        iterate.append(copy.deepcopy(M))
        i += 1
        if i == 100 :
            print('Fail to converge')
            break
    return M

def biweight_mad(sample, CSTD=9.):
    """
    Median absolute deviation with outlier rejection using mad clipping.
    Using the biweight described in Beers 1990 (Used originaly
    for finding galaxy clusters peculiar velocity dispersion).

    :param sample: 1d numpy array. The sample where you want
                   to compute the mad with outlier rejection.
    :param CSTD:   float. Constant used in the algorithm of the
                   Beers 1990. [default: 9.]
    """
    M = biweight_median(sample)
    mu = (sample-M) / (CSTD*mad_astropy(sample))
    Filtre = (abs(mu)<1)
    up = ((sample-M)**2)*((1.-mu**2)**4)
    down = (1.-mu**2)*(1.-5.*mu**2)
    mad = np.sqrt(len(sample)) * (np.sqrt(np.sum(up[Filtre]))/abs(np.sum(down[Filtre])))
    return mad


class meanify1D_wrms():
    """Take data, build a 1d average, and write output average.

    :param bin_spacing: Bin_size, resolution on the mean function. (default=0.3)
    """
    def __init__(self, bin_spacing=0.3):

        self.bin_spacing = bin_spacing

        self.coords = []
        self.params = []
        self.params_err = []

    def add_data(self, coord, param, params_err=None):
        """
        Add new data to compute the mean function. 

        :param coord: Array of coordinate of the parameter.
        :param param: Array of parameter.
        """
        self.coords.append(coord)
        self.params.append(param)
        if params_err is None:
            self.params_err = None
        else:
            self.params_err.append(params_err)

    def sigma_clipping(self, sigma=3.):
        self.std = []
        nvisits = len(self.params)
        for i in range(nvisits):
            self.std.append(np.std(self.params[i]))
        #mean, std = np.mean(self.std), np.std(self.std)
        mean, std = biweight_median(self.std), biweight_mad(self.std)

        I = 0
        for i in range(nvisits):
            if self.std[i] > mean + sigma * std:
                print('PF:', i)
                del self.params[I]
                if self.params_err is not None:
                    del self.params_err[I]
                del self.coords[I]
            else:
                I+=1
        print('Number of visit removed / nvisits: ',
              nvisits - len(self.params), '/', nvisits)

    def meanify(self, x_min=None, x_max=None):
        """
        Compute the mean function.
        """
        # self.sigma_clipping(sigma=3.)
        params = np.concatenate(self.params)
        coords = np.concatenate(self.coords)
        if self.params_err is not None:
            params_err = np.concatenate(self.params_err)
        else:
            params_err = np.ones_like(params)

        weights = 1./params_err**2

        if x_min is None:
            x_min = np.min(coords)
        if x_max is None:
            x_max = np.max(coords)

        nbin = int((x_max - x_min) / self.bin_spacing)

        binning = np.linspace(x_min, x_max, nbin)
        Filter = np.array([True]*nbin)


        sum_wpp, x0, bin_target = binned_statistic(coords, weights*params*params,
                                                   bins=binning, statistic='sum')

        sum_wp, x0, bin_target = binned_statistic(coords, weights*params,
                                                  bins=binning, statistic='sum')

        sum_w, x0, bin_target = binned_statistic(coords, weights,
                                                 bins=binning, statistic='sum')

        average = sum_wp / sum_w
        wvar = (1. / sum_w) * (sum_wpp - 2.*average*sum_wp + average*average*sum_w)
        wrms = np.sqrt(wvar)

        # get center of each bin 
        x0 = x0[:-1] + (x0[1] - x0[0])/2.

        # remove any entries with nan (counts == 0 and non finite value in
        # the 1D statistic computation) 
        self.x0 = x0
        self.average = average
        self.wrms= wrms

    def save_results(self, name_output='wrms_mag.fits'):
        """
        Write output mean function.
        
        :param name_output: Name of the output fits file. (default: 'mean_gp.fits')
        """
        dtypes = [('X0', self.x0.dtype, self.x0.shape),
                  ('AVERAGE', self.average.dtype, self.average.shape),
                  ('WRMS', self.wrms.dtype, self.wrms.shape),
                  ]
        data = np.empty(1, dtype=dtypes)
        
        data['X0'] = self.x0
        data['AVERAGE'] = self.average
        data['WRMS'] = self.wrms

        with fitsio.FITS(name_output,'rw',clobber=True) as f:
            f.write_table(data, extname='average_solution')


# The data from DP1 used 

# In[ ]:


repo = "/repo/main"
collection = "LSSTComCam/runs/DRP/DP1/v29_0_0_rc6/DM-50098"

butler = Butler(repo, collections=collection)
sourceTable_visit_dsrs = list(butler.registry.queryDatasets("refit_psf_star"))
visit_ids = []

for dsr in sourceTable_visit_dsrs:
    visit_ids.append(dsr.dataId["visit"])

print(len(visit_ids))
print(len(set(visit_ids)))


# Do the match at visit level between FGCM photometry and PSF stats from Piff. 

# In[ ]:


if os.path.isfile('master_dic_color.pkl'):
    WRITE = False
else: 
    WRITE = True

if WRITE: 

    
    dic = {}
    butler = Butler(repo, collections=collection)

    for visit in tqdm(visit_ids):

        try: 
        
            finalized_src_table = butler.get("refit_psf_star", visit=visit)
            visitSummary = butler.get("preliminary_visit_summary", visit=visit)

            # Highly inspired but what is done in GBDES for DCR correction.

            colorCatalog = butler.get("fgcm_Cycle5_StandardStars")
            catalogBands = colorCatalog.metadata.getArray("BANDS")
            colorInd1 = catalogBands.index("r")
            colorInd2 = catalogBands.index("z")
            colors = colorCatalog["mag_std_noabs"][:, colorInd1] - colorCatalog["mag_std_noabs"][:, colorInd2]
            mag = colorCatalog["mag_std_noabs"][:, colorInd1]
            goodInd = (colorCatalog["mag_std_noabs"][:, colorInd1] != 99.0) & (
                colorCatalog["mag_std_noabs"][:, colorInd2] != 99.0
            )
            ra_psf = (finalized_src_table['coord_ra'] * u.radian).to(u.degree).value
            dec_psf = (finalized_src_table['coord_dec'] * u.radian).to(u.degree).value

            with Matcher(ra_psf, dec_psf) as matcher:
                idx, idx_starCat, idx_colorCat, d = matcher.query_radius(
                    (colorCatalog[goodInd]["coord_ra"] * u.radian).to(u.degree).value,
                    (colorCatalog[goodInd]["coord_dec"] * u.radian).to(u.degree).value,
                    1. / 3600.0,
                    return_indices=True,
                )
            
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

            dT_T = np.array((table['T_src'] - table['T_psf']) / table['T_src'])
            de1 = np.array(table['e1_src'] - table['e1_psf'])
            de2 = np.array(table['e2_src'] - table['e2_psf'])

        
            dic.update({
                visit: {
                   'dT_T': dT_T[idx_starCat],
                    'de1': de1[idx_starCat],
                    'de2': de2[idx_starCat],
                    'color': colors[goodInd][idx_colorCat],
                    'mag': mag[goodInd][idx_colorCat],
                    'band': visitSummary[0]["band"],
                    
                }
            })

        except:
            dic.update({
                visit: None,
            })
    
    f = open('master_dic_color.pkl', 'wb')
    pickle.dump(dic, f)
    f.close()
    master_dic = pickle.load(open('master_dic_color.pkl', 'rb'))
else:

    master_dic = pickle.load(open('master_dic_color.pkl', 'rb'))


# Do the plot.

# In[ ]:


N = 0
bin_spacing = 0.5
XMIN = 15.5
XMAX = 22.5

key = 'dT_T'

meanifyAll = meanify1D_wrms(bin_spacing=bin_spacing)

meanifyLowColor = meanify1D_wrms(bin_spacing=bin_spacing)
meanifyMidColor = meanify1D_wrms(bin_spacing=bin_spacing)
meanifyHighColor = meanify1D_wrms(bin_spacing=bin_spacing)

for visit in master_dic:
    if master_dic[visit] is not None:

        isFinite = np.isfinite(master_dic[visit][key])
        meanifyAll.add_data(master_dic[visit]['mag'][isFinite],  master_dic[visit][key][isFinite], params_err=None)

        lowColor = isFinite & (master_dic[visit]['color'] < 1)
        meanifyLowColor.add_data(master_dic[visit]['mag'][lowColor],  master_dic[visit][key][lowColor], params_err=None)
        
        
        midColor = isFinite & (master_dic[visit]['color'] > 1) & (master_dic[visit]['color'] < 2)
        meanifyMidColor.add_data(master_dic[visit]['mag'][midColor],  master_dic[visit][key][midColor], params_err=None)

        
        highColor = isFinite & (master_dic[visit]['color'] > 2)
        meanifyHighColor.add_data(master_dic[visit]['mag'][highColor],  master_dic[visit][key][highColor], params_err=None)
        
    else:
        N += 1



meanifyAll.meanify(x_min=XMIN, x_max=XMAX)
meanifyLowColor.meanify(x_min=XMIN, x_max=XMAX)
meanifyMidColor.meanify(x_min=XMIN, x_max=XMAX)
meanifyHighColor.meanify(x_min=XMIN, x_max=XMAX)


#plt.figure(figsize=(10, 6))

plt.subplots_adjust(left=0.2)
plt.scatter(meanifyAll.x0, meanifyAll.average, c=stars_color(), label="All")
plt.scatter(meanifyLowColor.x0, meanifyLowColor.average, c=accent_color(), label=r"$r-z$ = 0. - 1.")
plt.scatter(meanifyMidColor.x0, meanifyMidColor.average, c='#949494', label=r"$r-z$ = 1. - 2.")
plt.scatter(meanifyHighColor.x0, meanifyHighColor.average, c='#029E73', label=r"$r-z$ = 2. - 4.")


plt.plot([15, 23], [0, 0.], 'k--', zorder=-1)
plt.ylim(-0.015, 0.015)
plt.xlim(15.9, 22.5)

plt.xlabel('Magnitude')
plt.ylabel('$\\left<\\left(T_{\\text{PSF}} - T_{\\text{model}}\\right) \\ / \\ T_{\\text{PSF}}\\right>$')
plt.legend()
plt.savefig('dT_T_Piff_poly_4_vs_mag.pdf')


# In[ ]:




