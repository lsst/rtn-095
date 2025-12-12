#!/usr/bin/env python
#
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

# Script for generating the various latex parameters referred to in the DP1
# paper.

# Contact Authors: James Mullaney

# LSST Science Pipelines
from lsst.daf.butler import Butler


def misc_outputs():
    '''Outputs to stdio various pieces of information that included in the
    DP1 paper.
    These are information that can't be readily included in the
    dp1_parameters.py script.

    Items include:
    - The number of missing visit_images per field;
    '''

    # Number of images:
    raw_refs = list(registry.queryDatasets('raw'))
    pvi_refs = list(registry.queryDatasets('visit_image'))

    raw_dataIds = [(ref.dataId['detector'], ref.dataId['exposure']) for ref in raw_refs]
    pvi_dataIds = [(ref.dataId['detector'], ref.dataId['visit']) for ref in pvi_refs]

    diff_dataIds = set(raw_dataIds) - set(pvi_dataIds)

    fails = []
    for ccd, visit in diff_dataIds:
        dataId = {'visit': visit, 'detector':ccd, 'instrument':'LSSTComCam'}
        record = list(registry.queryDimensionRecords('visit', dataId=dataId))[0]
        fails.append(record.target_name)

    print("Failed raws to visit_images:")
    for field_name in set(fails):
        print(f'{field_name}: {fails.count(field_name)}')


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
        collections=collections,
        skymap=skymapName
        )
    registry = butler.registry
    skymap = butler.get('skyMap', skymap=skymapName)

    misc_outputs()
