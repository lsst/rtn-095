import lsst.daf.butler as dafButler
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import patches
import numpy as np
from lsst.utils.plotting import set_rubin_plotstyle, get_multiband_plot_colors


def outlinePlot(points, paths, allBands, ax):
    uInside = np.zeros(len(points), dtype=bool)
    gInside = np.zeros(len(points), dtype=bool)
    rInside = np.zeros(len(points), dtype=bool)
    iInside = np.zeros(len(points), dtype=bool)
    zInside = np.zeros(len(points), dtype=bool)
    yInside = np.zeros(len(points), dtype=bool)
    allInside = np.ones(len(points), dtype=bool)

    for (path, band) in zip(paths, allBands):
        if band == "u":
            uInside |= path.contains_points(points)
        elif band == "g":
            gInside |= path.contains_points(points)
        elif band == "r":
            rInside |= path.contains_points(points)
        elif band == "i":
            iInside |= path.contains_points(points)
        elif band == "z":
            zInside |= path.contains_points(points)
        elif band == "y":
            yInside |= path.contains_points(points)

    insides = [uInside, gInside, rInside, iInside, zInside, yInside]
    colors = get_multiband_plot_colors()
    bands = ["u", "g", "r", "i", "z", "y"]
    paths = []
    n = 0
    for (band, inside) in zip(bands, insides):
        inside = inside.reshape(xx.shape)
        color = colors[band]
        contour = ax.contour(xx, yy, inside.astype(int), levels=[0.5], colors=[color], linewidths=[1])
        path = [Path(seg) for seg in contour.allsegs[0]]
        if np.sum(inside) > 0:
            print(band)
            n += 1
            patch = patches.PathPatch(path[0], facecolor=color, alpha=0.2)
            ax.add_patch(patch)
            paths.append(path[0])

    if n == 6:
        for path in paths:
            allInside &= path.contains_points(points)
        allInside = allInside.reshape(xx.shape)
    return ax

collections = ["LSSTComCam/runs/DRP/DP1/v29_0_0/DM-50260",
               "LSSTComCam/DP1",
               "skymaps",]
dataId = {"skymap":'lsst_cells_v1', "instrument":"LSSTComCam"}

butler = dafButler.Butler("/repo/dp1", collections=collections)

visitRecords = butler.registry.queryDimensionRecords("visit")

paths = []
refs = []
allBands = []
for ref in butler.registry.queryDatasets("visit_summary*", collections=collections, dataId=dataId):
    refs.append(ref)
    t = butler.get(ref).asAstropy()
    allCorners = zip(t["raCorners"], t["decCorners"])
    for corners in allCorners:
        raCornerList = list(corners[0])
        raCornerList.append(raCornerList[0])
        decCornerList = list(corners[1])
        decCornerList.append(decCornerList[0])
        path = Path(list(zip(raCornerList, decCornerList)))
        paths.append(path)
        allBands.append(t["band"][0])

print(t.columns)

set_rubin_plotstyle()
fig = plt.figure(figsize=(7.25, 7.0))

# 47 Tuc
ax1 = fig.add_subplot(331)
name = "47 Tucanae"
raMin = 3.8
raMax = 8.3
decMin = -72.8
decMax = -71.4
print(name)
xx, yy = np.meshgrid(np.linspace(raMin, raMax, 1000), np.linspace(decMin, decMax, 1000))
points = np.column_stack([xx.ravel(), yy.ravel()])
ax1 = outlinePlot(points, paths, allBands, ax1)
ax1.set_title(name)

# Fornax
name = "Fornax"
raMin = 39.2
raMax = 40.9
decMin = -35.2
decMax = -33.6
print(name)
xx, yy = np.meshgrid(np.linspace(raMin, raMax, 1000), np.linspace(decMin, decMax, 1000))
points = np.column_stack([xx.ravel(), yy.ravel()])
ax2 = fig.add_subplot(332)
ax2 = outlinePlot(points, paths, allBands, ax2)
ax2.set_title(name)

# ECDFS
name = "ECDFS"
raMin = 52.3
raMax = 54.0
decMin = -28.8
decMax = -27.3
print(name)
xx, yy = np.meshgrid(np.linspace(raMin, raMax, 1000), np.linspace(decMin, decMax, 1000))
points = np.column_stack([xx.ravel(), yy.ravel()])
ax3 = fig.add_subplot(333)
ax3 = outlinePlot(points, paths, allBands, ax3)
ax3.set_title(name)

# EDFS
name = "EDFS"
raMin = 58
raMax = 60.25
decMin = -49.5
decMax = -48.0
print(name)
xx, yy = np.meshgrid(np.linspace(raMin, raMax, 1000), np.linspace(decMin, decMax, 1000))
points = np.column_stack([xx.ravel(), yy.ravel()])
ax4 = fig.add_subplot(334)
ax4 = outlinePlot(points, paths, allBands, ax4)
ax4.set_title(name)
ax4.set_ylabel("Dec. (deg)")

# Rubin SV
name = "Rubin SV 95 -25"
raMin = 94.2
raMax = 95.85
decMin = -25.8
decMax = -24.2
print(name)
xx, yy = np.meshgrid(np.linspace(raMin, raMax, 1000), np.linspace(decMin, decMax, 1000))
points = np.column_stack([xx.ravel(), yy.ravel()])
ax5 = fig.add_subplot(335)
ax5 = outlinePlot(points, paths, allBands, ax5)
ax5.set_title(name)

# Seagull
name = "Seagull"
raMin = 105.55
raMax = 107
decMin = -11.2
decMax = -9.75
print(name)
xx, yy = np.meshgrid(np.linspace(raMin, raMax, 1000), np.linspace(decMin, decMax, 1000))
points = np.column_stack([xx.ravel(), yy.ravel()])
ax6 = fig.add_subplot(336)
ax6 = outlinePlot(points, paths, allBands, ax6)
ax6.set_title(name)

# Rubin SV 38 7
name = "Rubin SV 38 7"
raMin = 36.75
raMax = 39
decMin = 5.75
decMax = 8.1
print(name)
xx, yy = np.meshgrid(np.linspace(raMin, raMax, 1000), np.linspace(decMin, decMax, 1000))
points = np.column_stack([xx.ravel(), yy.ravel()])
ax8 = fig.add_subplot(338)
ax8 = outlinePlot(points, paths, allBands, ax8)
ax8.set_title(name)
ax8.set_xlabel("R.A. (deg)")

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax8]:
    ax.xaxis.set_inverted(True)

# Legend
colors = get_multiband_plot_colors()
ax7 = fig.add_subplot(337)
for band in ["u", "g", "r", "i", "z", "y"]:
    ax7.plot([0,1], [0,1], color=colors[band], label=band)
ax7.set_xlim(2, 3)
ax7.set_ylim(2, 3)
ax7.get_xaxis().set_visible(False)
ax7.get_yaxis().set_visible(False)
ax7.set_axis_off()
plt.legend(ncols=6, bbox_to_anchor=(3.28, 4.21))

plt.subplots_adjust(bottom=0.1, top=0.90, right=0.95, wspace=0.4, hspace=0.4)
plt.savefig("fieldSummaries.pdf")
plt.show()
