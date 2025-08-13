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

# Script for generating ISR Anomaly figures for DP1 paper.
# Original notebook written by Christopher Waters. Converted to a python
# script by James Mullaney.

# Contact Authors: Christopher Waters, James Mullaney

# Set up variables for the repo, collections, etc.
REPO = "/repo/main"
INSTRUMENT = "LSSTComCam"
COLLECTION = "LSSTComCam/runs/DRP/DP1/v29_0_0_rc6/DM-50098"

# Set up the butler.  The environment hack may not be needed anymore.
import os
os.environ['AWS_RESPONSE_CHECKSUM_VALIDATION'] = 'WHEN_REQUIRED'

# All figures in this directory
from pathlib import Path
figures_filepath = Path("../figures")

import lsst.daf.butler as dB
butler = dB.Butler(REPO)
camera = butler.get("camera", instrument=INSTRUMENT,
                    collections="LSSTComCam/calib/DM-48650")

# Grab standard things.
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Things we need to display the plots for checking:
from IPython.display import IFrame
import base64

from matplotlib import cm, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lsst.utils.plotting import make_figure, set_rubin_plotstyle
import matplotlib.patheffects as pathEffects

def exposure_to_pdf(exposure,
                    filename,
                    title="",
                    center=None,
                    size=None,
                    units=""):
    """Convert an exposure to a PDF file, with specified view window.

    Parameters
    ----------
    exposure : lsst.afw.image.Exposure
        The image to convert.
    filename : str
        Output filename.
    center : tuple, optional
        Pixel to center in the view.
    size : tuple, optional
        Size of the view in pixels.
    """
    try:
        array = exposure.image.array
    except AttributeError:
        # I said it had to be an exposure, but maybe not?
        array = exposure.array

    # Setup axes and figure:
    fig = make_figure()
    # set_rubin_plotstyle needs to happen prior to instantiating the axes.
    set_rubin_plotstyle()

    ax = fig.gca()
    ax.clear()

    # Get image scaling from data:
    cmap = cm.gray
    # This was using summit_utils.getQuantiles, but that
    # was blowing out the scaling more than I wanted.
    q25, q50, q75 = np.nanpercentile(array, [25, 50, 75])
    scale = 3.0 * 0.74 * (q75 - q25)
    quantiles = np.arange(q50 - scale, q50 + scale, 2.0 * scale / cmap.N)
    norm = colors.BoundaryNorm(quantiles, cmap.N)
    print(q25, q50, q75, scale)

    # Do the image display:
    im = ax.imshow(array, norm=norm,
                   interpolation='None', cmap=cmap, origin='lower')
    if center is not None and size is not None:
        ax.set_xlim(center[0] - size[0], center[0] + size[0])
        ax.set_ylim(center[1] - size[1], center[1] + size[1])
    ax.set_aspect("equal")

    # Add colorbar
    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="10%", pad=0.0)
    # This removes the black strip on the right side.
    cax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, cax=cax)
    # This sets the tick marker formatting.
    cbar.formatter = ticker.StrMethodFormatter("{x:.3f}")


    label = f"{units}"
    text = cax.text(0.5, 0.5, label, color="k",
                    rotation="vertical", transform=cax.transAxes,
                    ha="center", va="center", fontsize=10)
    text.set_path_effects([pathEffects.Stroke(linewidth=3, foreground="w"),
                           pathEffects.Normal()])

    # Final plot properties:
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    fig.suptitle(title)

    fig.savefig(filename)


def display_pdf(filename):
    """I love technology so much and how simple things are!

    Parameters
    ----------
    filename : `str`
        path to PDF to open.

    Returns
    -------
    iframe : `IFrame`
        Whatever.  You can see the plot now.

    Notes
    -----
    I had a series of words here, that I have since deleted.
    This trick appears to work only if the image section is less than 1k x 1k.
    "Why does the PDF viewer care how much image went into the plot?"
    Friend, if only I knew the answer to that.
    """
    with open(filename, "rb") as pdf:
        content = pdf.read()
    # encode PDF
    base64_pdf = base64.b64encode(content).decode("utf-8")

    # return encoded form
    return IFrame(f"data:application/pdf;base64,{base64_pdf}",
                  width=600, height=550)

### Plots ###
# Vampire pixels
filename = (figures_filepath /
            "dp1_isr_anomalies-vampire_pixel.pdf")
flat = butler.get("flat", instrument=INSTRUMENT,
                  band="r", exposure=2024120600239, detector=4,
                  collections="LSSTComCam/calib")
exposure_to_pdf(flat, filename,
                # title="flat detector=R22_S11; filter=r_03",
                center=(2537, 958), size=(300, 300),
                units="normalized throughput"
               )

display_pdf(filename)

# Phosphorescence
# g band flat
filename = (figures_filepath /
            "dp1_isr_anomalies-phosphorescence.pdf")
flat = butler.get("flat", instrument=INSTRUMENT, band="g",
                  exposure=2024112600178, detector=1,
                  collections="LSSTComCam/calib")
exposure_to_pdf(flat, filename,
                # title="flat R22_S01 g_01",
                center=(300, 3700), size=(300, 300),
                units="normalized throughput"
               )
display_pdf(filename)

# Crosstalk residual
filename = (figures_filepath /
             "dp1_isr_anomalies-crosstalk_residual.pdf")
exp = butler.get("post_isr_image", instrument="LSSTComCam",
                 exposure=2024120600239, detector=2,
                 collections=COLLECTION)
exposure_to_pdf(exp, filename,
                center=(3200, 800), size=(800, 800),
                units="electrons"
               )
display_pdf(filename)

# ITL Dip
# detector 8 R22_S22 is worst
filename = (figures_filepath /
            "dp1_isr_anomalies-itl_dip.pdf")
exp = butler.get("post_isr_image", instrument="LSSTComCam",
                 exposure=2024121000503, detector=8,
                 collections=COLLECTION)
exposure_to_pdf(exp, filename,
                center=(2300, 2000), size=(300, 300),
                units="electrons"
               )
display_pdf(filename)