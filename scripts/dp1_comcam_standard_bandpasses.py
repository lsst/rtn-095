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

# Script for generating a plot of the standard ComCam bandpasses.
# Original notebook written by Eli Rykoff and Leanne Guy. Converted to
# a python script by James Mullaney.

# Contact Authors: Eli Rykoff, Leanne Guy, James Mullaney

import matplotlib.pyplot as plt

from lsst.daf.butler import Butler
from lsst.utils.plotting import publication_plots
from lsst.utils.plotting import (get_multiband_plot_colors,
                                 get_multiband_plot_symbols)

# Set publication style
publication_plots.set_rubin_plotstyle()
colors = get_multiband_plot_colors()
bands = colors.keys()
symbols = get_multiband_plot_symbols()

# Is there a non-user collection that contains LSSTComCam standard_passband?
print("*** CAUTION: Currently plotting LSSTCam's standard_passband ***")
collections = ['u/erykoff/LSSTCam/calib/fgcmcal/DM-51274']
butler = Butler("/repo/main", instrument="LSSTCam", collections=collections)

# Get the throughputs for ComCam filters
throughputs = {}
for band in bands:
    std_bp = butler.get("standard_passband", band=band)
    throughputs[band] = std_bp

# Plot
plt.figure()
for band, std_bp in throughputs.items():

    tpt = std_bp['throughput']/100
    plt.plot(std_bp['wavelength'], tpt,
             color = colors[band[0]],
             label=band)

# Force a y-axis zero
plt.ylim(0)

# Add axis labels and title
plt.xlabel('Wavelength (nm)')
plt.ylabel('Standard Bandpass')

# Annotate with band label
plt.text(360, 0.06, 'u')
plt.text(470, 0.22, 'g')
plt.text(620, 0.28, 'r')
plt.text(740, 0.30, 'i')
plt.text(860, 0.22, 'z')
plt.text(960, 0.12, 'y')

plt.savefig("../figures/dp1_comcam_std_bandpasses.pdf",
            bbox_inches='tight', transparent=True, format='pdf')