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

# Script for generating a table that contains the 5sigma depths of the
# DP1 fields.

# Contact Authors: James Mullaney

from lsst.daf.butler import Butler
from collections import defaultdict
import numpy as np
from astropy import units as u

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

fields = defaultdict(set)
with butler.query() as base_query:
    processed_visit_query = base_query.join_dataset_search("visit_summary").where('visit > 0')
    for row in processed_visit_query.general(["tract", "band"], "visit.target_name"):
        if row["visit.target_name"] != 'slew_icrs':
            fields[row["visit.target_name"]].add(row["tract"])

fieldOrder = ['47_Tuc', 'ECDFS', 'EDFS_comcam', 'Fornax_dSph', 'Rubin_SV_095_-25', 'Rubin_SV_38_7', 'Seagull']
bands = ['u','g','r','i','z','y']

with open("../tables/dp1_m5_coadded_depths.tex", "w") as f:
    f.write(r"""%%%%% This table is auto generated from data, DO NOT EDIT
\setlength{\tabcolsep}{6pt}  % default is 6pt
\begin{deluxetable}{lcccccc}
\tablecaption{Median $5\sigma$ coadd detection limits per field and band.
\label{tab:dp1_m5_depths} }

\tablehead{
  \textbf{Field Code} & \multicolumn{6}{c}{\textbf{Band}}\\
  \cline{2-7}
   &u&g&r&i&z&y
}
\startdata
""")

    for fieldName in fieldOrder:
        latexName = fieldName.replace('_', '\\_')
        medmags = ''
        for band in bands:
            mags = np.array([])
            tracts = fields[fieldName]
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
                    mags = np.append(mags,(table[starSelection][f'{band}_psfFlux'] * u.nJy).to_value(u.ABmag))
            medmagstr = f'{np.median(mags):.2f}'
            if 'nan' in medmagstr:
                medmagstr = '-'
            medmags = medmags + '&' + medmagstr
        f.write(f'{latexName}{medmags}\\\\\n')
    f.write(r"""\enddata
\end{deluxetable}
""")
f.close()