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

# Script for generating a series of tables for the DP1 paper.
# Original notebook written by James Mullaney. Converted to a python
# script by James Mullaney.

# Contact Authors: James Mullaney

from lsst.daf.butler import Butler
from collections import defaultdict
import numpy as np
import copy

# Butler setup
instrument = 'LSSTComCam'
collections = ['LSSTComCam/DP1',
               'LSSTComCam/runs/DRP/DP1/v29_0_0/DM-50260',
               'skymaps', ]
skymap = 'lsst_cells_v1'
butler = Butler("/repo/dp1",
                instrument=instrument,
                collections=collections,
                skymap=skymap)
registry = butler.registry
skymap = butler.get('skyMap', skymap=skymap)

# Table of available dimensions:

dimensionDescriptions = {
    'day_obs':'A day and night of observations that rolls over during daylight hours.',
    'visit':"A sequence of observations processed together; synonymous with ``exposure'' in DP1.",
    'exposure':'A single exposure of all nine ComCam detectors.',
    'instrument':'The instrument name.',
    'detector':'A ComCam detector.',
    'skymap':'A set of tracts and patches that subdivide the sky into rectangular regions with simple projections and intentional overlaps.',
    'tract':'A large rectangular region of the sky.',
    'patch':'A rectangular region within a tract.',
    'physical_filter':'An astronomical filter.',
    'band':'An astronomical wave band.',
}

detectors = [record.id for record in list(registry.queryDimensionRecords('detector'))]
skymap = list(registry.queryDimensionRecords('skymap'))[0].name
patches = set([record.id for record in list(registry.queryDimensionRecords('patch', datasets='template_coadd'))])
dimensionValues = {
    'day_obs':'YYYYMMDD',
    'visit':'YYYYMMDD\\#\\#\\#\\#\\#',
    'exposure':'YYYYMMDD\\#\\#\\#\\#\\#',
    'instrument':'LSSTComCam',
    'detector':f'{min(detectors)} - {max(detectors)}',
    'skymap':f'\\texttt{{{skymap.replace('_','\\_')}}}',
    'tract':'See Table \\ref{tab:dp1_tracts}',
    'patch':f'{min(patches)} - {max(patches)}',
    'physical_filter': 'u\\_02, g\\_01, i\\_06, r\\_03, z\\_03, y\\_04',
    'band':'u, g, r, i, z, y',
}

with open("../tables/dp1_dimension_summary.tex", "w") as f:
    f.write(r"""%%%%% This table is auto generated from data, DO NOT EDIT
\begin{deluxetable}{lp{3.5cm}p{8cm}}
\caption{Descriptions of and valid values for the key data dimensions in DP1. YYYYMMDD signifies date and \# signifies a single 0-9 digit.
\label{tab:dp1_dimensions} }
\tablehead{
  \colhead{\textbf{Dimension}} & \colhead{\textbf{Format/Valid values}} & \colhead{\textbf{Description}}\\
}
\startdata
""")
    for dimension in dimensionValues:
        latexName = dimension.replace('_', '\\_')
        f.write(f'\\texttt{{{latexName}}}&{dimensionValues[dimension]}&{dimensionDescriptions[dimension]}\\\\\n')
    f.write(r"""\enddata
\end{deluxetable}
""")
f.close()

# Tracts covering each field
fields = defaultdict(set)
with butler.query() as base_query:
    processed_visit_query = base_query.join_dataset_search("visit_summary").where('visit > 0')
    for row in processed_visit_query.general(["tract", "visit"], "visit.target_name"):
        fields[row["visit.target_name"]].add(row["tract"])

fieldOrder = ['47_Tuc', 'ECDFS', 'EDFS_comcam', 'Fornax_dSph', 'Rubin_SV_095_-25', 'Rubin_SV_38_7', 'Seagull']
with open("../tables/dp1_field_tracts.tex", "w") as f:
    f.write(r"""%%%%% This table is auto generated from data, DO NOT EDIT
\begin{deluxetable}{lp{4.5cm}}
\caption{Tract coverage of each DP1 field. The size of a tract is larger than the LSSTCam field of view; however, since each observed field extends across more than one tract, each field covers multiple tracts.
\label{tab:dp1_tracts}}
\tablehead{
  \colhead{\textbf{Field Code}} & \colhead{\textbf{Tract ID}}
}
\startdata
""")
    for field in fieldOrder:
        latexName = field.replace('_', '\\_')
        tracts = f'{np.array2string(np.sort(list(fields[field])), separator=', ')}'
        f.write(f'{latexName}&{tracts.strip('[]')}\\\\\n')
    f.write(r"""\enddata
\end{deluxetable}
""")
f.close()

# Number of raw images per field and band
fields = defaultdict(list)
with butler.query() as base_query:
    processed_visit_query = base_query.join_dataset_search("raw")
    for row in processed_visit_query.general(["band","detector"], "visit.target_name"):
        fields[row["visit.target_name"]].append(row["band"])

bandCounts = {'u':0, 'g':0, 'r':0, 'i':0, 'z':0, 'y':0}
rawCounts = {}
for field in fields.keys():
    rawCounts[field] = copy.deepcopy(bandCounts)

for field in fields.keys():
    for band in bandCounts.keys():
        rawCounts[field][band] += np.sum(np.array(fields[field]) == band)
        if field == 'slew_icrs':
            rawCounts['ECDFS'][band] += np.sum(np.array(fields[field]) == band)

fieldOrder = ['47_Tuc', 'ECDFS', 'EDFS_comcam', 'Fornax_dSph', 'Rubin_SV_095_-25', 'Rubin_SV_38_7', 'Seagull']
bandTotalCounts = {'u':0, 'g':0, 'r':0, 'i':0, 'z':0, 'y':0}

with open("../tables/rawbreakdown.tex", "w") as f:
    f.write(r"""%%%%% This table is auto generated from data, DO NOT EDIT
\setlength{\tabcolsep}{6pt}  % default is 6pt
\begin{deluxetable}{lccccccc}
\tablecaption{Number of \texttt{raw} images per field and band.
\label{tab:rawbreakdown} }

\tablehead{
  \colhead{\textbf{Field Code}} & \multicolumn{6}{c}{\textbf{Band}} & \textbf{Total}\\
  \cline{2-7}
   &u&g&r&i&z&y&
}
\startdata
""")
    for fieldName in fieldOrder:
        if fieldName == 'slew_icrs':
            continue
        latexName = fieldName.replace('_', '\\_')
        f.write(f'{latexName}')
        total = 0
        for band in ['u','g','r','i','z','y']:
            total += rawCounts[fieldName][band]
            bandTotalCounts[band] += rawCounts[fieldName][band]
            f.write(f'&{rawCounts[fieldName][band]}')
        f.write(f'&{total}\\\\\n')
    f.write('\\cline{1-8}\n')
    f.write('Total')
    bandTotal = 0
    for band in ['u','g','r','i','z','y']:
        bandTotal += bandTotalCounts[band]
        f.write(f'&{bandTotalCounts[band]}')
    f.write(f'&{bandTotal}\\\\\n')
    f.write(r"""\enddata
\end{deluxetable}
""")
f.close()

# Number and primary dimensions of each type of dataset
datasetTypes = {'raw':'raw',
                'visit_image':'visit_image',
                'deep_coadd':'deep_coadd',
                'template_coadd':'template_coadd',
                'difference_image':'difference_image',
                'Source':'source',
                'Object':'object',
                'ForcedSource':'object_forced_source',
                'DiaSource':'dia_source',
                'DiaObject':'dia_object',
                'ForcedSourceOnDiaObject':'dia_object_forced_source',
                'CCDVisit':'visit_detector_table',
                'SSObject':'ss_object',
                'SSSource':'ss_source',
                'Visit':'visit_table',
               }

with open("../tables/dp1_butler_datasets.tex", "w") as f:
    f.write(r"""%%%%% This table is auto generated from data, DO NOT EDIT
\setlength{\tabcolsep}{6pt}  % default is 6pt
\begin{deluxetable}{llcc}
\tablecaption{The name and number of each type of data product in the Butler and the dimensions required to identify a specific dataset.
\label{tab:butlerdatasets} }

\tablehead{
  \textbf{Data Product} &
  \textbf{Name in Butler} &
  \textbf{Required Dimensions} &
  \textbf{Number in DP1}\
}
\startdata
""")

    for datasetType in datasetTypes:
        reqDims = registry.getDatasetType(datasetTypes[datasetType]).dimensions.required
        reqDimsString = f'{reqDims}'.strip('{}')
        if len(reqDims) == 0:
            reqDimsString = '--'
        nDatasets = len(list(registry.queryDatasets(datasetTypes[datasetType])))
        latexName = datasetType.replace('_', '\\_')
        latexButlerName = datasetTypes[datasetType].replace('_', '\\_')
        f.write(f'\\texttt{{{latexName}}}&\\texttt{{{latexButlerName}}}&{reqDimsString}&{nDatasets}\\\\\n')
    f.write(r"""\enddata
\end{deluxetable}
""")
f.close()