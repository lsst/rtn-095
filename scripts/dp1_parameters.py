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

import os
import csv
import requests
import yaml
import numpy as np
import pandas as pd
import numpy as np
from pathlib import Path
from haversine import haversine, Unit
from astropy import units as u
from tqdm import tqdm

# LSST Science Pipelines
from lsst.daf.butler import Butler

import warnings
warnings.filterwarnings("ignore")

# Create a dictionary to convert numbers to words.
# Used for those numbers best expressed in words.
def num2word(num):
    '''
    Converts select integer values into their word equivalent.
    If the integer is not one of the select values, then the integer
    is returned as a string to ensure consistent return type.
    '''
    num2wordDict = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four',
                    5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight',
                    9: 'Nine', 10: 'Ten', 11: 'Eleven', 12: 'Twelve',
                    13: 'Thirteen', 14: 'Fourteen', 15: 'Fifteen',
                    16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen',
                    19: 'Nineteen', 20: 'Twenty',
                    30: 'Thirty', 40: 'Forty', 50: 'Fifty', 60: 'Sixty',
                    70: 'Seventy', 80: 'Eighty', 90: 'Ninety', 0: 'Zero'}
    if num in num2wordDict:
        return num2wordDict[num]
    else:
        return str(num)

# Function to round to N significant figures
def round_sf(x, sig=3):
    return np.round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

def formatParameter(params, name):
    '''
    Formats a named parameter ready for write-out to tex file.
    '''

    value = params[0][name]
    if isinstance(value, float):
        value = f'{value:.3f}'.rstrip('0').rstrip('.')

    if name in params[1]:
        unit = params[1][name]
        if unit[0:7] not in ['\\arcsec', '\\arcmin', '\\degree']:
            unit = f'\\xspace {unit}'
    else:
        unit = ''

    return f'\\newcommand{{\\{name}}}{{{value}{unit}\\xspace}}\n'

def addParameter(params, name, value, unit=None, **kwargs):

    '''
    Adds a named parameter to the params tuple of dictionaries.

    Ensures that the namings between the values and units
    dictionaries are consistent.
    '''
    if 'sig' in kwargs:
        value = round_sf(value, **kwargs)

    params[0][name] = value
    if unit:
        params[1][name] = unit

    return params

def imageStats(params, imageId):
    '''
    Given an image dataset type name, add the following to the params:
    - Number of dataset types in DP1;
    - HDD size of the first instance of the dataset type in MB;
    - Number of pixels in each x/y dimension;
    - Platescale of image;
    - On-sky field of view of image, both in x/y and area.
    '''

    imageType = imageId[0]
    imageDataId = imageId[1]
    imageName = imageType.replace('_', '')

    # Number of images:
    refs = list(registry.queryDatasets(imageType))
    params = addParameter(params, f'n{imageName}s', len(list(refs)))

    # Image HDD size:
    filepath = butler.getURI(imageType, dataId=imageDataId)
    roughFileSize = round_sf(os.path.getsize(filepath.path) / 1e6, sig=2)
    params = addParameter(params, f'{imageName}hdd',
                          f'{roughFileSize:.0f}', unit='MB')

    # Number of pixels
    image = butler.get(imageType, dataId=imageDataId)
    params = addParameter(params, f'n{imageName}pixx', image.getDimensions().x)
    params = addParameter(params, f'n{imageName}pixy', image.getDimensions().y)

    # Platescale:
    platescale = image.getWcs().getPixelScale().asArcseconds()
    params = addParameter(params, f'{imageName}platescale',
                          f'{platescale:.1f}', unit='\\arcsec per pixel')

    # Field of view
    fovx = image.getDimensions().x * image.getWcs().getPixelScale().asDegrees()
    fovy = image.getDimensions().y * image.getWcs().getPixelScale().asDegrees()
    params = addParameter(params, f'{imageName}fovx',
                          f'{fovx:.2f}', unit='\\degree')
    params = addParameter(params, f'{imageName}fovy',
                          f'{fovy:.2f}', unit='\\degree')
    area = fovx * fovy
    roughArea = round_sf(area, sig=3)
    params = addParameter(params, f'{imageName}fov',
                          f'{roughArea:.3f}', unit='deg$^2$')

    return params

#-------- Manually added parameters --------#
# A small number of parameters cannot be readily generated autimatically.
# These are kept in the manualParameters.csv file.
def manualParameters(params):
    print('Adding manual parameters...')
    with open('../manualParameters.csv', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            if line[2] == '':
                line[2]=None
            params = addParameter(params, line[0], line[1], unit=line[2])

    return params

#-------- Information about the observing campaign --------#
def observingCampaign(params):
    print('Adding observing campaign parameters...')

    # Number of visits:
    visitRecords = registry.queryDimensionRecords('visit')
    params = addParameter(params, 'nvisits', len(list(visitRecords)))

    # Number of exposures. This should be the same as the number of visits.
    exposureRecords = registry.queryDimensionRecords('exposure')
    params = addParameter(params, 'nexposures', len(list(visitRecords)))

    # Number of target fields:
    # Note: slew_icrs covered ECDFS, so they are the same field:
    fields = set([
        record.target_name
        for record in exposureRecords if record.target_name != 'slew_icrs'
        ])
    params = addParameter(params, 'nfields', num2word(len(fields)).lower())

    # First and last date of DP1 observations:
    visit_table = butler.get('visit_table')
    firstVisitTime = min(visit_table['obsStart'])
    lastVisitTime = max(visit_table['obsStart'])
    # The last night started the date before the last observation datetime.
    lastVisitNight = lastVisitTime - np.timedelta64(1,'D')
    params = addParameter(params, 'dponestartdate',
                        np.datetime_as_string(firstVisitTime, unit='D'))
    params = addParameter(params, 'dponeenddate',
                        np.datetime_as_string(lastVisitNight, unit='D'))

    # Median exposure time (non u-band):
    u_selection = visit_table['band'] == 'u'
    params = addParameter(params, 'exposuretime',
                        f'{np.median(visit_table['expTime'][~u_selection]):.0f}',
                        unit='s')

    # Median u-band exposure time:
    params = addParameter(params, 'exposuretimeuband',
                        f'{np.median(visit_table['expTime'][u_selection]):.0f}',
                        unit='s')
    return params

#-------- Observation Quality -------#
def observingQuality(params):
    print('Adding observing quality parameters...')

    visit_detector_table = butler.get('visit_detector_table')
    visit_detector_table["psfFwhm"]=visit_detector_table["psfSigma"]*2.355*0.2

    # Best seeing:
    minSeeing = np.min(
        visit_detector_table['psfFwhm'][visit_detector_table['nPsfStar']>100]
        )
    params = addParameter(params, 'bestimagequality',
                          minSeeing.item(), unit='\\arcsec', sig=2)

    # Median seeing:
    medSeeing = np.median(
        visit_detector_table['psfFwhm'][visit_detector_table['nPsfStar']>100]
        )
    params = addParameter(params, 'medianimagequalityallbands',
                          medSeeing.item(), unit='\\arcsec', sig=3)

    return params

#-------- Stats of different image datasets -------#
def imageDatasets(params):
    print('Adding image dataset stats...')

    datasets = {'raw':
                {'instrument': 'LSSTComCam',
                'detector': 0,
                'exposure': 2024110800245,
                'band': 'i'},
                'visit_image':
                {'instrument': 'LSSTComCam',
                'detector': 0,
                'visit': 2024110800245,
                'band': 'i'},
                'deep_coadd':
                {'band': 'g',
                'skymap': 'lsst_cells_v1',
                'tract': 5063,
                'patch': 14},
                'template_coadd':
                {'band': 'g',
                'skymap': 'lsst_cells_v1',
                'tract': 5063,
                'patch': 14},
                'difference_image':
                {'instrument': 'LSSTComCam',
                'detector': 0,
                'visit': 2024110800245,
                'band': 'i'}
                }
    for dataset in datasets.items():
        params = imageStats(params, dataset)

    # The total number of pixels in a single deep_coadd image:
    params = addParameter(
        params, 'ndeepcoaddpixtotal',
        params[0]['ndeepcoaddpixx']*params[0]['ndeepcoaddpixy']/1e6,
        sig=3, unit='million')

    return params

#-------- Skymap data --------#
def skymapData(params):
    print('Adding skymap parameters...')

    # Total number of tracts across the entire sky:
    params = addParameter(params, 'ntotaltracts', len(skymap))

    # The number of tracts covered by DP1:
    tractRecords = list(
        registry.queryDimensionRecords('tract', where='visit > 0')
        )
    tractIds = set([record.id for record in tractRecords])
    params = addParameter(params, 'ntracts', len(tractIds))
    params = addParameter(params, 'ncoveredtracts', len(tractIds))

    # Area of each tract:
    tract = skymap.generateTract(9000)
    verticesInDegrees = [
        (vertex[0].asDegrees()%360 - 180, vertex[1].asDegrees())
        for vertex in tract.getVertexList()
        ]
    tractArea = (
        haversine(verticesInDegrees[0][::-1],
                verticesInDegrees[1][::-1],
                unit=Unit.DEGREES) *
        haversine(verticesInDegrees[1][::-1],
                verticesInDegrees[2][::-1],
                unit=Unit.DEGREES)
    )
    roughTractArea = round_sf(tractArea, sig=2)
    params = addParameter(params, 'tractarea',
                        f'{roughTractArea:.1f}', unit='deg$^2$')

    # Number of patches:
    numXPatches, numYPatches = skymap[0].getNumPatches()
    numPatches = numXPatches * numYPatches
    params = addParameter(params, 'npatchx', numXPatches)
    params = addParameter(params, 'npatchy', numYPatches)
    params = addParameter(params, 'npatch', numPatches)

    # Area of each patch:
    patchArea = (tractArea / numPatches)
    roughPatchArea = round_sf(patchArea, sig=2)
    params = addParameter(params, 'innerpatcharea',
                        f'{roughPatchArea:.3f}', unit='deg$^2$')

    refs = list(registry.queryDatasets('deep_coadd'))
    coadd = butler.get(refs[0])
    fovx = coadd.getDimensions().x * coadd.getWcs().getPixelScale().asDegrees()
    fovy = coadd.getDimensions().y * coadd.getWcs().getPixelScale().asDegrees()
    roughOuterArea = round_sf(fovx * fovy, sig=2)
    params = addParameter(params, 'outerpatcharea',
                        f'{roughOuterArea:.3f}', unit='deg$^2$')

    return params

#-------- Image selection criteria for deep_coadd -------#
def coaddSelectionCriteria(params):
    print('Adding coadd selection criteria...')
    # The seeing criterion is communicated in the config file:
    refs = list(registry.queryDatasets('selectDeepCoaddVisits_config'))
    config = butler.get(refs[0])
    return addParameter(params, 'deepcoaddmaxfwhm',
                        config.maxPsfFwhm, sig=2, unit='\\arcsec')

#-------- Survey property maps --------#
def surveyPropertyMaps(params):
    print('Adding numbers of survey maps...')
    # Find all HealSparseMap dataset types matching a pattern
    allSuveyPropMaps = []
    for datasetType in butler.registry.queryDatasetTypes():
        if registry.queryDatasets(datasetType).any(
            execute=False, exact=False):
            if (datasetType.storageClass.name == "HealSparseMap"):
                allSuveyPropMaps.append(datasetType.name)
    return addParameter(params, 'nsurveypropertymaps',len(allSuveyPropMaps))

#-------- The numbers of datasets of each type of catalogs -------#
def nCatalogDatasets(params):
    print('Adding catalog dataset stats...')
    refs = list(registry.queryDatasets('source'))
    params = addParameter(params, 'nsourcecatalogs',
                        num2word(len(refs)).lower())

    refs = list(registry.queryDatasets('object'))
    params = addParameter(params, 'nobjectcatalogs',
                        num2word(len(refs)).lower())

    refs = list(registry.queryDatasets('dia_object'))
    params = addParameter(params, 'ndiaobjectcatalogs',
                        num2word(len(refs)).lower())

    refs = list(registry.queryDatasets('dia_source'))
    params = addParameter(params, 'ndiasourcecatalogs',
                        num2word(len(refs)).lower())

    refs = list(registry.queryDatasets('ss_source'))
    params = addParameter(params, 'nsolarsystemsourcecatalogs',
                        num2word(len(refs)).lower())

    refs = list(registry.queryDatasets('visit_table'))
    params = addParameter(params, 'nvisitsummarytables',
                        num2word(len(refs)).lower())

    refs = list(registry.queryDatasets('visit_detector_table'))
    params = addParameter(params, 'nvisitdetectorsummarytables',
                        num2word(len(refs)).lower())

    refs = list(registry.queryDatasets('object_forced_source'))
    params = addParameter(params, 'nobjectforcedcatalogs',
                        num2word(len(refs)).lower())

    refs = list(registry.queryDatasets('dia_object_forced_source'))
    params = addParameter(params, 'ndiaobjectforcedcatalogs',
                        num2word(len(refs)).lower())
    return params


#-------- The number of entries in various tables --------#
def tableLengths(params):
    print('Adding table lengths...')
    # There is only one of each of these dataset types:
    refs = list(registry.queryDatasets('visit_table'))
    table = butler.get(refs[0])
    params = addParameter(params, 'nvisitsummaries', len(table))

    refs = list(registry.queryDatasets('visit_detector_table'))
    table = butler.get(refs[0])
    params = addParameter(params, 'nvisitdetectorsummaries', len(table))

    refs = list(registry.queryDatasets('ss_source'))
    catalog = butler.get(refs[0]) # There is only 1 ss_source catalog in DP1.
    params = addParameter(params, 'nsolarsystemsources', len(catalog))

    refs = list(registry.queryDatasets('ss_object'))
    catalog = butler.get(refs[0])
    params = addParameter(params, 'nsolarsystemobjects', len(catalog))

    return params

#-------- Miscalleneous --------#
def misc(params):
    print('Adding misc data...')
    # Report the number of failed raw-to-visit_image processings as the
    # difference in the number of raws and the number of visit_images.

    params = addParameter(params, 'nsfpfails',
                          params[0]['nraws'] - params[0]['nvisitimages'])
    return params


def totalDP1Area(params):
    print('Adding total DP1 area. This is a slow step...')
    coaddRefs = list(registry.queryDatasets('deep_coadd', where="band = 'r'"))
    pixcount = 0
    for coaddRef in tqdm(coaddRefs):
        im = butler.get('deep_coadd.mask', dataId = coaddRef.dataId)
        pixcount +=  im.array.size - np.sum(
            (im.array & im.getPlaneBitMask('NO_DATA')) > 0
            )
    area = pixcount * ((0.2 / 3600)**2)
    params = addParameter(params, 'totalarea',
                          f'{np.round(area):.0f}', unit='deg$^2$')
    return params

def nObjects(params):
    print('Adding total number of objects...')
    runningTotal = 0
    refs = list(registry.queryDatasets('object'))
    for ref in tqdm(refs):
        catalog = butler.get(ref, parameters = {'columns':'objectId'})
        runningTotal += len(catalog)
    params = addParameter(params, 'nobjects',
                        runningTotal/1e6, sig=2, unit='million')
    return params

def nSources(params):
    print('Adding total number of sources...')
    refs = list(registry.queryDatasets('source'))
    runningTotal = 0
    for ref in tqdm(refs):
        catalog = butler.get(ref, parameters = {'columns':'sourceId'})
        runningTotal += len(catalog)
    params = addParameter(params, 'nsources',
                        runningTotal/1e6, sig=2, unit='million')
    return params

def nDiaObjects(params):
    print('Adding total number of diaObjects...')
    refs = list(registry.queryDatasets('dia_object'))
    runningTotal = 0
    for ref in tqdm(refs):
        catalog = butler.get(ref, parameters = {'columns':'diaObjectId'})
        runningTotal += len(catalog)
    params = addParameter(params, 'ndiaobjects',
                        runningTotal/1e6, sig=2, unit='million')
    return params

def nDiaSources(params):
    print('Adding total number of diaSources...')
    refs = list(registry.queryDatasets('dia_source'))
    runningTotal = 0
    for ref in tqdm(refs):
        catalog = butler.get(ref, parameters = {'columns':'diaSourceId'})
        runningTotal += len(catalog)
    params = addParameter(params, 'ndiasources',
                        runningTotal/1e6, sig=2, unit='million')
    return params

def nForced(params):
    print('Adding total number of forced sources and objects...')
    refs = list(registry.queryDatasets('object_forced_source'))
    runningTotalSrc = 0
    runningTotalObj = 0
    for ref in tqdm(refs):
        catalog = butler.get(ref, parameters = {'columns':'objectId'})
        runningTotalSrc += len(catalog)
        runningTotalObj += len(catalog.to_pandas()['objectId'].unique())
    params = addParameter(params, 'nforcedsources',
                        runningTotalSrc/1e6, sig=3, unit='million')
    params = addParameter(params, 'nforcedobjects',
                        runningTotalObj/1e6, sig=2, unit='million')
    return params

def nDiaForced(params):
    print('Adding total number of DIA forced sources and objects...')

    refs = list(registry.queryDatasets('dia_object_forced_source'))
    runningTotalSrc = 0
    runningTotalObj = 0
    for ref in tqdm(refs):
        catalog = butler.get(ref, parameters = {'columns':'diaObjectId'})
        runningTotalSrc += len(catalog)
        runningTotalObj += len(catalog.to_pandas()['diaObjectId'].unique())
    params = addParameter(params, 'ndiaforcedsources',
                          runningTotalSrc/1e6, sig=3, unit='million')
    params = addParameter(params, 'ndiaforcedobjects',
                        runningTotalObj/1e6, sig=2, unit='million')
    return params

def nStarsGals(params):
    print('Adding total number of stars and gals...')

    nGals = 0
    nStars = 0
    nAll = 0
    refs = list(registry.queryDatasets('object'))
    columns = [band + '_extendedness' for band in ['u', 'g', 'r', 'i', 'z', 'y']]
    for ref in tqdm(refs):
        objectTable = butler.get(ref, parameters = {'columns':columns})
        if len(objectTable) > 0:
            galSelection = (
                ((objectTable['u_extendedness'] > 0.5) &
                ~objectTable['u_extendedness'].mask) |
                ((objectTable['g_extendedness'] > 0.5) &
                ~objectTable['g_extendedness'].mask)  |
                ((objectTable['r_extendedness'] > 0.5) &
                ~objectTable['r_extendedness'].mask)  |
                ((objectTable['i_extendedness'] > 0.5) &
                ~objectTable['i_extendedness'].mask)  |
                ((objectTable['z_extendedness'] > 0.5) &
                ~objectTable['z_extendedness'].mask)  |
                ((objectTable['y_extendedness'] > 0.5) &
                ~objectTable['y_extendedness'].mask)
            )
            nGals += np.sum(galSelection.data)

            starSelection = (
                ((objectTable['u_extendedness'] <= 0.5) &
                ~objectTable['u_extendedness'].mask) &
                ((objectTable['g_extendedness'] <= 0.5) &
                ~objectTable['g_extendedness'].mask)  &
                ((objectTable['r_extendedness'] <= 0.5) &
                ~objectTable['r_extendedness'].mask)  &
                ((objectTable['i_extendedness'] <= 0.5) &
                ~objectTable['i_extendedness'].mask)  &
                ((objectTable['z_extendedness'] <= 0.5) &
                ~objectTable['z_extendedness'].mask)  &
                ((objectTable['y_extendedness'] <= 0.5) &
                ~objectTable['y_extendedness'].mask)
            )
            nStars += np.sum(starSelection.data)

        params = addParameter(params, 'nextendedobjects',
                            nGals/1e6, sig=2, unit='million')
    return params

def nDeepCoaddInputImages(params):
    print('Adding number of deep coadd input images...')
    # Use the /repo/main butler since deep_coadd_input_map isn't available
    # in the public data release.
    refs = list(mainRegistry.queryDatasets('deep_coadd_input_map'))
    visitDetectorPairs = set()
    for ref in tqdm(refs):
        inputMap = mainButler.get(ref)
        keyRoots = [
            key[:5]
            for key in inputMap.metadata
            if key.startswith("B") & key.endswith('CCD')
            ]
        visitDetectorPairs.update(
            {(inputMap.metadata[keyRoot+'VIS'],
                inputMap.metadata[keyRoot+'CCD']) for keyRoot in keyRoots}
                )
    params = addParameter(params, 'ndeepcoaddvisitimages',
                          len(visitDetectorPairs))
    return params

def nTemplateCoaddInputImages(params):
    print('Adding number of template coadd input images...')

    # This method is not as robust as the above (it assumes all visit_images
    # from a selected visit are used), but the equivalent
    # of `deep_coadd_input_map` doesn't exist for template coadds.

    # template_coadd_visit_selection isn't available in the public data release.
    refs = list(mainRegistry.queryDatasets('template_coadd_visit_selection'))
    visits = set()
    for ref in tqdm(refs, desc='Part 1 of 2'):
        inputVisits = mainButler.get(ref)
        visits.update({key for (key,value) in inputVisits.items() if value})

    nTemplateCoaddInputs = 0
    for visit in tqdm(visits, desc='Part 2 of 2'):
        nTemplateCoaddInputs += len(
            list(
                mainRegistry.queryDatasets(
                    'visit_image',
                    visit=visit,
                    instrument='LSSTComCam'
                    ),
                )
        )
    params = addParameter(params,
                          'ntemplatecoaddvisitimages',
                          nTemplateCoaddInputs)
    return params

def depthEcdfs(params):
    print('Adding number ECDFS depths...')

    tracts = []
    with butler.query() as base_query:
        processed_visit_query = (
            base_query.join_dataset_search("visit_summary").where(
                "visit.target_name = 'ECDFS'"
            )
        )
        for row in processed_visit_query.general(["tract"]):
            tracts.append(row["tract"])

    for band in tqdm(bands):
        mags = np.array([])
        for tract in tracts:
            columns = [f'{band}_psfFlux',
                    f'{band}_psfFluxErr',
                    f'{band}_psfFlux_flag',
                    f'{band}_extendedness']
            table = butler.get('object',
                            tract=tract,
                            skymap='lsst_cells_v1',
                            parameters={'columns':columns})
            sn = table[f'{band}_psfFlux'] / table[f'{band}_psfFluxErr']
            if len(table) > 0:
                starSelection = (
                    (~table[f'{band}_psfFlux_flag']) &
                    (table[f'{band}_extendedness'] <= 0.5) &
                    ~table[f'{band}_extendedness'].mask &
                    (sn > 4.9) &
                    (sn < 5.1))
                flux = table[starSelection][f'{band}_psfFlux'] * u.nJy
                mags = np.append(mags, flux.to_value(u.ABmag))
        params = addParameter(params, f'{band}depth', np.median(mags), sig=4)

    return params

if __name__ == "__main__":

    # Globally list the bands:
    bands = ['u', 'g', 'r', 'i', 'z', 'y']

    # Butler setup:
    instrument = 'LSSTComCam'
    collections = [
        'LSSTComCam/DP1',
        'LSSTComCam/runs/DRP/DP1/v29_0_0/DM-50260',
        'skymaps',
        ]
    skymapName = 'lsst_cells_v1'

    # /repo/dp1 butler:
    butler = Butler(
        "/repo/dp1",
        instrument=instrument,
        collections=[
            'LSSTComCam/DP1',
            'LSSTComCam/runs/DRP/DP1/v29_0_0/DM-50260',
            'skymaps'],
            skymap=skymapName
        )
    registry = butler.registry
    skymap = butler.get('skyMap', skymap=skymapName)

    # /repo/main Butler
    mainButler = Butler(
        "/repo/main",
        instrument=instrument,
        collections=['LSSTComCam/runs/DRP/DP1/v29_0_0_rc6/DM-50098'],
        skymap=skymapName)
    mainRegistry = mainButler.registry

    params = (dict(), dict())

    # Using functions allows for easier debugging.
    params = manualParameters(params)
    params = observingCampaign(params)
    params = observingQuality(params)
    params = imageDatasets(params)
    params = skymapData(params)
    params = coaddSelectionCriteria(params)
    params = surveyPropertyMaps(params)
    params = nCatalogDatasets(params)
    params = tableLengths(params)
    params = misc(params)
    #Calculating area is slow, so it is included in manualParameters.csv
    #params = totalDP1Area(params)
    params = nObjects(params)
    params = nSources(params)
    params = nDiaObjects(params)
    params = nDiaSources(params)
    params = nForced(params)
    params = nDiaForced(params)
    params = nStarsGals(params)
    params = nDeepCoaddInputImages(params)
    params = nTemplateCoaddInputImages(params)
    params = depthEcdfs(params)

    #-------- Write the parameters to file --------#
    boilerPlate = '''% These parameters are automatically generated by the dp1_parameters notebook.
% Do NOT manually edit this file.
% If you need to change/add a parameter, please edit the dp1_parameters notebook and re-run it.\n
'''
    with open("../parameters.tex", "w") as f:
        f.write(boilerPlate)
        for name in params[0]:
            f.write(formatParameter(params, name))
    f.close()
