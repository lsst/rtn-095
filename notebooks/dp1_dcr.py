# This file is part of rtn-095.
#
# Developed for the LSST Data Management System.
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


# Script for computing the Differential Chromatic Refraction (DCR) effect by
# matching astrometric and photometric sources from LSST deep coadds. It
# selects the visit set with the highest expected DCR value using the
# `DcrMetric` class, then estimates blackbody temperatures, effective
# wavelengths, and positional offsets for each object. An optional plotting
# utility visualizes the DCR effect as described in the DP1 publication.

# Note: To generate plots in the correct DP1 format, the `lsst.utils` branch
# `tickets/DM-50892` must be set up prior to execution.

# Contact Authors: Audrey Budlong, Ian Sullivan

__all__ = ["DcrEffect", "DcrMetric"]

import esutil
import lsst.geom
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.table import vstack, hstack
from lsst.afw.coord import differentialRefraction
from lsst.daf.butler import Butler
from lsst.ip.diffim import calculateImageParallacticAngle
from lsst.utils.plotting import stars_cmap, accent_color
import matplotlib.patheffects as pathEffects
from scipy.optimize import curve_fit


class DcrEffect:
    """Class for computing the Differential Chromatic Refraction (DCR) effect
    by matching astrometric and photometric sources and deriving relevant
    physical and observational parameters for each object.

    The DCR calculation begins with the use of the `DcrMetric` class to
    evaluate the expected DCR metric for each set of visits contributing to
    deep coadd exposures. The visit set with the highest expected DCR value is
    selected to define the subset of data used in the subsequent analysis.

    Matched photometric and astrometic sources from the selected visits are
    used to estimate the associated blackbody temperatures by fitting the
    fluxes across the available LSST bands ('g', 'r', 'i', and 'z'). From the
    fitted blackbody spectrum, the effective and reference wavelengths are
    computed. These parameters, along with astrometric measurements, are used
    to determine the angular separation between source and reference positions—
    quantifying the DCR-induced positional offset.

    An optional plotting method is available to visualize the DCR effect in a
    format consistent with the associated DP1 publication, including both
    linear and logarithmic representations of the effect's parallel and
    perpendicular components. Note: The DCR effect is only visible in the
    parallel component.
    """
    def __init__(
        self,
        instrument="LSSTComCam",
        skymap="lsst_cells_v1",
        repo="/repo/dp1",
        collections=None,
        *args,
        **kwargs,
    ):
        if collections is None:
            collections = [
                "LSSTComCam/DP1/defaults",
                "LSSTComCam/runs/DRP/DP1/v29_0_0/DM-50260",
                "skymaps",
            ]
        self.butler = Butler(
            repo, instrument=instrument, collections=collections, skymap=skymap
        )

    def run(self, doHexbin=False):
        datareferences = self.butler.query_datasets("deep_coadd", where="band='g'")
        dcrMetrics, dcrVisits = self.calculateExpectedDcr(datareferences)

        # Sort data by expected DCR metric results and length of visit set
        dcrData = {"DCR Visits": dcrVisits, "DCR Metric": dcrMetrics}
        dcrMetricResults = pd.DataFrame(data=dcrData)
        dcrMetricResults.sort_values(
            by="DCR Metric", inplace=True, ascending=False, ignore_index=True
        )
        dcrMetricResults.sort_values(
            "DCR Visits",
            key=lambda s: [len(x) for x in s],
            ascending=False,
            ignore_index=True,
        )

        # Select set of visits with the highest DCR metric
        selectedVisits = dcrMetricResults["DCR Visits"][0]
        selectedVisits = list(selectedVisits)

        # Get references for set of visits with highests DCR metric
        references = [
            self.butler.query_datasets(
                "preliminary_visit_image", where="band='g'" and f"visit={v}"
            )
            for v in selectedVisits
        ]

        # Flatten results
        refs = [ref for element in references for ref in element]

        resultTables = []
        for ref in refs:
            singleVisitResult = self.matchSources(ref, "i", "g")
            resultTables.append(singleVisitResult)

        testAllData = vstack(resultTables)
        allData = testAllData.to_pandas()

        # Calculate blackbody temperatures
        blackbodyTemperatures = findBlackbodyTemp(allData)
        allData["blackbody temperature"] = blackbodyTemperatures

        # Calculate the effective and reference wavelengths
        effWvl = computeEffectiveWavelength(blackbodyTemperatures)
        refTemp = np.mean(blackbodyTemperatures)
        refWavelength = computeEffectiveWavelength([refTemp])

        floatEW = [float(i) for i in effWvl]
        allData["effective wavelength"] = floatEW

        refEW = [float(i) for i in refWavelength]
        allData["reference effective wavelength"] = len(allData) * [
            refEW,
        ]

        # Calculate the differential refraction
        differentialRefraction_blackbody = computeDifferentialRefraction(allData, floatEW, refEW[0])
        allData["differential refraction"] = differentialRefraction_blackbody

        if doHexbin:
            # Generate a hexbin plot illustrating the differential chromatic
            # refraction (DCR) effect as seen in the input dataset. This
            # visualization is intended specifically for the DP1 paper.
            self.hexbin(allData)

    def calculateExpectedDcr(self, datareferences):
        """Calculate the expected differential chromatic refraction (DCR) value
        and associated visits from deep coadds using the DcrMetric class.

            Parameters
            ----------
            datareferences : `list` of `lsst.daf.butler.DeferredDatasetHandle`
                The data references to the deep coadd exposures.

            Returns
            -------
            dcrMetricComCam : `list`
                Expected DCR metric values associated with each set of visits.
            dcrVisits : `list`
                List of sets of visits associated with deep coadd
                exposures that contribute to the calculated expected DCR
                metric.
            """
        dcrMetricComCam = []
        dcrVisitsSet = []
        for ref in datareferences:
            templateCoadds = self.butler.get("deep_coadd", dataId=ref.dataId)
            templateInputs = templateCoadds.getInfo().getCoaddInputs()
            visits = templateInputs.ccds["visit"]
            ccds = templateInputs.ccds["ccd"]
            data = {"visit": visits, "ccd": ccds}
            dataDF = pd.DataFrame(data=data)
            dataDF.drop_duplicates(
                subset=["visit"], keep="last", inplace=True, ignore_index=True
            )
            dcrVisitsSet.append(set(visits))

            templateVinfos = [
                self.butler.get(
                    "preliminary_visit_image.visitInfo",
                    dataId={"band": "g", "detector": ccd, "visit": visit},
                )
                for (visit, ccd) in zip(dataDF["visit"], dataDF["ccd"])
            ]

            dcrMetric1 = DcrMetric()
            [dcrMetric1.addVisitInfo(vinfo) for vinfo in templateVinfos]
            dcrMetricComCam.append(dcrMetric1.calculateMetric())

        dcrVisits = [list(element) for element in dcrVisitsSet]

        return dcrMetricComCam, dcrVisits

    def retrievePsfStars(self, dataref):
        """Retrieve the `single_visit_psf_star_footprints` catalog for a given
        data reference.

            Parameters
            ----------
            dataref : `lsst.daf.butler.DeferredDatasetHandle`
                The data reference of exposures.

            Returns
            -------
            psfsources : `pandas.DataFrame`
                DataFrame of `single_visit_psf_star_footprints` catalog.
            """
        psfSourceTable = self.butler.get(
            "single_visit_psf_star_footprints", dataId=dataref.dataId
        )
        psfsources = psfSourceTable.asAstropy()
        return psfsources

    def retrieveStarFootprints(self, dataref):
        """Retrieve the `single_visit_star_footprints` catalog for a given
        data reference.

            Parameters
            ----------
            dataref : `lsst.daf.butler.DeferredDatasetHandle`
                The data reference of exposures.

            Returns
            -------
            psfsources : `pandas.DataFrame`
                DataFrame of `single_visit_star_footprints` catalog.
            """
        sourceTable = self.butler.get("single_visit_star_footprints", dataId=dataref.dataId)
        sources = sourceTable.asAstropy()
        return sources

    def retrieveAstrometry(self, dataref):
        """Retrieve the `initial_astrometry_match_detector` catalog for the
        specified data reference and compute the angular separation between
        the reference and source coordinates for each object.

            Parameters
            ----------
            dataref : `lsst.daf.butler.DeferredDatasetHandle`
                The data reference of exposures.

            Returns
            -------
            astrometry : `pandas.DataFrame`
                DataFrame of `initial_astrometry_match_detector` catalog with
                calculated angular separations between reference and source
                coordinates for each object.
            """
        astrometryTable = self.butler.get(
            "initial_astrometry_match_detector", dataId=dataref.dataId
        )
        astrometry = astrometryTable.asAstropy()

        astrometryLen = len(astrometry)
        visitInfo = self.butler.get("preliminary_visit_image.visitInfo", dataId=dataref.dataId)
        wcs = self.butler.get("preliminary_visit_image.wcs", dataId=dataref.dataId)
        airmass = visitInfo.boresightAirmass
        elevation = visitInfo.getBoresightAzAlt().getLatitude()
        boresightParAngle = visitInfo.boresightParAngle
        observatory = visitInfo.getObservatory()

        # Calculate the difference in reference and source coordinates in terms
        # of angular separation in arcseconds.
        angularSepArcsec = []
        for i in range(astrometryLen):
            c1 = SkyCoord(
                ra=astrometry["ref_coord_ra"][i] * u.radian,
                dec=astrometry["ref_coord_dec"][i] * u.radian,
                frame="icrs",
            )
            c2 = SkyCoord(
                ra=astrometry["src_coord_ra"][i] * u.radian,
                dec=astrometry["src_coord_dec"][i] * u.radian,
                frame="icrs",
            )
            angularSeparation = (c1.separation(c2).radian * u.radian).to(u.arcsec)
            angularSepArcsec.append(angularSeparation)

        # Append angular separation between reference and source coords to
        # astrometry catalog.
        astrometry["angular_separation"] = angularSepArcsec
        refX, refY = wcs.skyToPixelArray(
            ra=astrometry["ref_coord_ra"], dec=astrometry["ref_coord_dec"]
        )
        srcX, srcY = wcs.skyToPixelArray(
            ra=astrometry["src_coord_ra"], dec=astrometry["src_coord_dec"]
        )

        parallacticAngle = calculateImageParallacticAngle(visitInfo, wcs)

        dx = refX - srcX
        dy = refY - srcY
        amplitude = [np.sqrt(x**2 + y**2)*wcs.getPixelScale() for (x, y) in zip(dx, dy)]  # as an Angle
        angle = [lsst.geom.Angle(np.arctan2(y, x) + (np.pi / 2)) for (x, y) in zip(dx, dy)]
        perpendicular = [
            amp * np.sin(float(ang - parallacticAngle))
            for (amp, ang) in zip(amplitude, angle)
        ]
        parallel = [
            amp * np.cos(float((ang - parallacticAngle)))
            for (amp, ang) in zip(amplitude, angle)
        ]

        astrometry["airmass"] = astrometryLen * [airmass,]
        astrometry["elevation"] = astrometryLen * [elevation,]
        astrometry["boresightParAngle"] = astrometryLen * [boresightParAngle,]
        astrometry["observatory"] = astrometryLen * [observatory,]
        astrometry["perpendicular"] = perpendicular
        astrometry["parallel"] = parallel

        return astrometry

    def retrievePhotometry(self, dataref, band1, band2):
        """Retrieve the `initial_photometry_match_detector` catalog for a
        specified data reference and compute the magnitudes associated with the
        fluxes in the two selected photometric bands.

            Parameters
            ----------
            dataref : `lsst.daf.butler.DeferredDatasetHandle`
                The data reference of exposures.
            band1: `string`
                LSST band (ex. 'u', 'g', 'r', 'i', 'z').
            band2: `string`
                Comparison LSST band (ex. 'u', 'g', 'r', 'i', 'z'). This must
                be different than band1.

            Returns
            -------
            photometry : `pandas.DataFrame`
                DataFrame of `initial_photometry_match_detector` catalog with
                magnitudes and difference in magnitudes of the fluxes
                associated with the two chosen bands.
            """
        photometryTable = self.butler.get(
            "initial_photometry_match_detector", dataId=dataref.dataId
        )
        photometry = photometryTable.asAstropy()
        photometry = photometry[~photometry["src_sky_source"]]

        # convert nJy to AB mag
        b1_mag = photometry[f"ref_monster_ComCam_{band1}_flux"].to(u.ABmag)
        b2_mag = photometry[f"ref_monster_ComCam_{band2}_flux"].to(u.ABmag)

        # append AB magnitudes
        photometry[f"{band1}Mag"] = b1_mag
        photometry[f"{band2}Mag"] = b2_mag

        # calculate the difference in magnitude and append
        photometry[f"{band2}-{band1} mag"] = b2_mag - b1_mag

        return photometry

    def matchSources(self, dataref, band1, band2):
        """Retrieve the sources from the photometry and astrometry catalogs.

            Parameters
            ----------
            dataref : `lsst.daf.butler.DeferredDatasetHandle`
                The data reference of exposures.
            band1: `string`
                LSST band (ex. 'u', 'g', 'r', 'i', 'z').
            band2: `string`
                Comparison LSST band (ex. 'u', 'g', 'r', 'i', 'z'). This must
                be different than band1.

            Returns
            -------
            matchAstrometryPhotometry : `pandas.DataFrame`
                DataFrame of  matched astrometry and photometry results. This
                dataFrame contains the angular separation (astrometry) along
                with the magnitudes and difference in magnitudes of the fluxes
                associated with the two chosen bands (photometry).
            """
        stars = self.retrieveStarFootprints(dataref)
        psf_stars = self.retrievePsfStars(dataref)
        astrometry = self.retrieveAstrometry(dataref)
        photometry = self.retrievePhotometry(dataref, band1, band2)

        matched_ids = stars["psf_id"] > 0
        # Note: psf_stars[matches] and stars[matched_ids] are the matching
        # objects.
        matches = np.searchsorted(psf_stars["id"], stars["psf_id"][matched_ids])

        # Only non-zero psf_ids have cross-matches.
        # matched_ids = photometry_matches["src_psf_id"] > 0
        matches = esutil.numpy_util.match(
            astrometry["src_id"], photometry["src_psf_id"]
        )

        astrometry = astrometry[matches[0]]
        photometry = photometry[matches[1]]
        astrometry.sort("src_id")
        photometry.sort("src_psf_id")
        matchAstrometryPhotometry = hstack([astrometry, photometry])

        return matchAstrometryPhotometry


class DcrMetric:
    """For a given set of visits, calculate a differential chromatic refraction
    (dcr) metric to determine if it is possible to constrain a new observation
    with the dcr algorithm.
    """
    def __init__(self, max_airmass=1.8, base_nbins=6):
        self.max_airmass = max_airmass
        self.hist_range = [1 - max_airmass, max_airmass - 1]
        self.base_nbins = 6
        self.nbin_multipliers = [1, 2, 4]
        self.kde = None
        self.hist = []
        self.weight = []
        for multiplier in self.nbin_multipliers:
            self.weight.append(np.prod(self.nbin_multipliers) / multiplier)
            self.hist.append(
                np.histogram(
                    np.nan, self.base_nbins * multiplier, range=self.hist_range
                )[0]
            )

        self.table = pd.DataFrame(
            columns=["airmass", "hour_angle"], dtype="float"
        )  # visit will be the index

    @property
    def metricThreshold(self):
        """Calculate the minimum metric necessary to constrain the DCR
        model.

        Returns
        -------
        metric : `TEXT`
            Minimum metric necessary to contrain the DCR model.
        """
        metrics = [
            weight / nbins for nbins, weight in zip(self.nbin_multipliers, self.weight)
        ]
        metric = np.sum(metrics) / np.sum(self.weight)
        return metric

    def getBin(self, measure, h_ind):
        """Find the bin of the histogram for a given visit measure.

        """
        binsize = (self.hist_range[1] - self.hist_range[0]) / (
            self.base_nbins * self.nbin_multipliers[h_ind]
        )
        h_bin = int(np.floor((measure - self.hist_range[0]) / binsize))
        return h_bin

    def _updateHist(self, visit_measure, delete=False):
        """Update histogram.

        Parameters
        ----------
        visit_measure: `TEXT`
            TEXT
        """
        for h_ind, multiplier in enumerate(self.nbin_multipliers):
            hist = np.histogram(
                visit_measure, self.base_nbins * multiplier, range=self.hist_range
            )[0]
            if delete:
                self.hist[h_ind] -= hist
            else:
                self.hist[h_ind] += hist

    def parameterizeVisit(self, airmass, hour_angle):
        """Convert airmass and hour angle to visit measure.

        Parameters
        ----------
        airmass: `np.ndarray`, (N,)
            A healpix map with the airmass value of each healpixel. (unitless)
            or airmass are updated.
        hour_angle: `float`
            Hour angle coordinate of the current exposure.

        Returns
        -------
        visit_measure : `TEXT`
            Converted airmass and hour angle to visit measure.
        """
        visit_measure = (airmass - 1.0) * np.sign(hour_angle)
        return visit_measure

    def addVisit(self, visit, airmass, hour_angle):
        """Update the database with a new observation.

        Parameters
        ----------
        visit: ``
            TEXT
        airmass: `np.ndarray`, (N,)
            A healpix map with the airmass value of each healpixel. (unitless)
            or airmass are updated.
        hour_angle: `float`
            Hour angle coordinate of the current exposure.

        Raises
        ------
        KeyError
            Raised if visit is already present in the table and cannot be
            added.

        """
        if visit in self.table.index:
            raise KeyError(
                f"Visit {visit} is already present in the table, and cannot \
                be added. Skipping it."
            )
        self.table.loc[visit] = [airmass, hour_angle]
        visit_measure = self.parameterizeVisit(airmass, hour_angle)
        self._updateHist(visit_measure)

    def addVisitInfo(self, visitInfo):
        """Update the database with a new observation using an exposures
        visitInfo.

        Parameters
        ----------
        visitInfo: `lsst.afw.image.VisitInfo`
            Metadata for the exposure.
        """
        visit = visitInfo.getId()  # //100
        hour_angle = visitInfo.getBoresightHourAngle().asRadians()
        airmass = visitInfo.getBoresightAirmass()
        self.addVisit(visit, airmass, hour_angle)

    def delVisit(self, visit):
        """Update the database by removing a visit.

        Parameters
        ----------
        visit: `TEXT`
            TEXT

        Raises
        ------
        KeyError
            Raised if visit not present in the table could not be removed.
        """
        try:
            airmass = self.table["airmass"][visit]
            hour_angle = self.table["hour_angle"][visit]
            self.table.drop(visit, inplace=True)
        except Exception:
            raise KeyError(
                f"Could not remove visit {visit}, it is not in the \
                           table."
            )
        else:
            visit_measure = self.parameterizeVisit(airmass, hour_angle)
            self._updateHist(visit_measure, delete=True)

    def testAddVisit(self, airmass, hour_angle, threshold=1):
        """Calculate the metric improvement if a new observation were added.

        Parameters
        ----------
        airmass: `np.ndarray`, (N,)
            A healpix map with the airmass value of each healpixel. (unitless)
            or airmass are updated.
        hour_angle: `float`
            Hour angle coordinate of the current exposure.

        Returns
        -------
        improvement : `TEXT`
            Metric improvement if a new observation is added.
        """
        visit_measure = self.parameterizeVisit(airmass, hour_angle)
        improvement = 0.0
        for h_ind, multiplier in enumerate(self.nbin_multipliers):
            hist = np.histogram(
                visit_measure, self.base_nbins * multiplier, range=self.hist_range
            )[0]
            h_bin = np.argmax(hist)
            if self.hist[h_ind][h_bin] == (threshold - 1):
                improvement += (
                    hist[h_bin] * self.weight[h_ind] / (len(hist) * np.sum(self.weight))
                )
        return improvement

    def testDelVisit(self, visits, threshold=1):
        """Calculate the metric regression if an existing observation were
        excluded.

        Parameters
        ----------
        visits: `list`
            List of visits.

        Returns
        -------
        improvement : `TEXT`
            Metric regression if an existing observation were excluded.
        """
        visit_measures = []
        improvement = 0.0
        if not isinstance(visits, list):
            visits = [
                visits,
            ]
        for visit in visits:
            airmass = self.table["airmass"][visit]
            hour_angle = self.table["hour_angle"][visit]
            visit_measures.append(self.parameterizeVisit(airmass, hour_angle))
        for h_ind, multiplier in enumerate(self.nbin_multipliers):
            hist = np.histogram(
                visit_measures, self.base_nbins * multiplier, range=self.hist_range
            )[0]
            count0 = np.sum(self.hist[h_ind] >= threshold)
            count1 = np.sum((self.hist[h_ind] - hist) >= threshold)
            improvement += (
                (count1 - count0)
                * self.weight[h_ind]
                / (len(hist) * np.sum(self.weight))
            )
        return improvement

    def calculateMetric(self, exclude=None, threshold=1):
        """Calculate the DCR metric for the observations in the database.

        Returns
        -------
        metric : `TEXT`
            DCR metric for the observations in the dataset.
        """
        metrics = [
            np.mean(hist >= threshold) * weight
            for hist, weight in zip(self.hist, self.weight)
        ]
        metric = np.sum(metrics) / np.sum(self.weight)
        return metric

    def calculateKde(self):
        """Calculate a Kernel Density Estimator for the observations in the
        database.
        """
        visit_measures = []
        for visit in self.table.index:
            airmass = self.table["airmass"][visit]
            hour_angle = self.table["hour_angle"][visit]
            measure = self.parameterizeVisit(airmass, hour_angle)
            if measure >= self.hist_range[0] and measure <= self.hist_range[1]:
                visit_measures.append(measure)
        kde_width = 2 * (self.hist_range[1] - self.hist_range[0]) / self.base_nbins
        self.kde = []
        for multiplier in self.nbin_multipliers:
            density = kde_width / multiplier
            weight_kde = scipy.stats.gaussian_kde(visit_measures, bw_method=density)
            weights = 1 / weight_kde(visit_measures)
            self.kde.append(
                scipy.stats.gaussian_kde(
                    visit_measures, bw_method=density, weights=weights
                )
            )

    def evaluateVisit(self, airmass, hour_angle, doPlot=True, fig_id=1000):
        """Use the KDE to test whether a well-constrained template can be made
        for an observation.

        Parameters
        ----------
        airmass: `np.ndarray`, (N,)
            A healpix map with the airmass value of each healpixel. (unitless)
            or airmass are updated.
        hour_angle: `float`
            Hour angle coordinate of the current exposure.

        Returns
        -------
        TEXT : `TEXT`
            TEXT

        """
        if self.kde is None:
            self.calculateKde()
        visit_measure = self.parameterizeVisit(airmass, hour_angle)
        density = []
        if doPlot:
            visit_measures = []
            xv = np.arange(self.hist_range[0] - 0.3, self.hist_range[1] + 0.3, 0.01)
            for visit in self.table.index:
                airmass = self.table["airmass"][visit]
                hour_angle = self.table["hour_angle"][visit]
                visit_measures.append(self.parameterizeVisit(airmass, hour_angle))
        for h_ind, multiplier in enumerate(self.nbin_multipliers):
            kde = self.kde[h_ind]
            if doPlot:
                nVisits = len(visit_measures)
                plt.figure(fig_id + h_ind)
                plt.hist(
                    visit_measures,
                    range=self.hist_range,
                    bins=self.base_nbins * multiplier,
                )
                plt.plot(visit_measures, np.ones(nVisits), "+")
                plt.plot(xv, kde(xv), "-")
                plt.plot(visit_measure, 2.0, "x")
                plt.legend(
                    [
                        "Visit measures",
                        "KDE",
                        "Science observation visit measure",
                        "Histogram of visit measures",
                    ]
                )
            density.append(kde(visit_measure))
        metrics = [d * weight for d, weight in zip(density, self.weight)]
        metric = np.sum(metrics) / np.sum(self.weight)
        return metric * self.calculateMetric()

    def evaluateVisitInfo(self, visitInfo):
        """Use the KDE to test whether a well-constrained template can be made
        for an observation, using the visitInfo.

        Parameters
        ----------
        visitInfo: `lsst.afw.image.VisitInfo`
            Metadata for the exposure.

        Returns
        -------
        TEXT : `TEXT`
            TEXT

        """
        hour_angle = visitInfo.getBoresightHourAngle().asRadians()
        airmass = visitInfo.getBoresightAirmass()
        return self.evaluateVisit(airmass, hour_angle)


def findBlackbodyTemp(dataset):
    """Calculate the blackbody temperature for each source by fitting the flux
    values in each band ('g', 'r', 'i', 'z').

        Parameters
        ----------
        dataset: `pandas.DataFrame`
            DataFrame of  matched astrometry and photometry results. This
            dataFrame contains the angular separation (astrometry) along with
            the magnitudes and difference in magnitudes of the fluxes
            associated with the two chosen bands (photometry).

        Returns
        -------
        temperatures : `list`
            List of blackbody temperatures, where each entry corresponds to a
            specific source.
        """

    # Define the blackbody function.
    def bbFit(wavelength, temp, scale):
        tempK = temp * u.K
        blackbody = models.BlackBody(temperature=tempK)
        flux = (
            scale
            * blackbody(wavelength * u.nm)
            .to(u.nJy / u.sr, equivalencies=u.spectral())
            .value
        )
        return flux

    temperatures = []

    # Individually iterate through each of the objects in the catalog.
    for i in range(len(dataset)):
        gpoint = (478.5, dataset["ref_monster_ComCam_g_flux_1"][i])
        rpoint = (650.0, dataset["ref_monster_ComCam_r_flux_1"][i])
        ipoint = (754.6, dataset["ref_monster_ComCam_i_flux_1"][i])
        zpoint = (910.0, dataset["ref_monster_ComCam_z_flux_1"][i])

        data = (gpoint, rpoint, ipoint, zpoint)

        # Only keep points that are not `nan.`
        cleaned_data = [(x, y) for x, y in data if not (math.isnan(y))]

        xdata = [x for x, y in cleaned_data]
        ydata = [y for x, y in cleaned_data]

        bounds = ([1000, 0], [10000, 1e-18])
        initial_guess = [4000, 5 * 10e-22]

        # Perform the fit with bounds
        popt, pcov = curve_fit(bbFit, xdata, ydata, p0=initial_guess, bounds=bounds)

        temperatures.append(popt[0])

    return temperatures


def computeEffectiveWavelength(blackbodyTemps):
    """Compute the effective wavelength corresponding to each specified
    blackbody temperature.

        Parameters
        ----------
        blackbodyTemps: `list`
            List of blackbody temperatures, where each entry corresponds to a
            specific source.

        Returns
        -------
        effectiveWavelengths : `list`
            List of effective temperatures derived from the blackbody
            temperatures. Each entry corresponds to a specific source.
        """

    # Define variable for effective wavelengths.
    effectiveWavelengths = []

    # Iterate through the temperatures of all the objects individually.
    for temp in blackbodyTemps:
        # Define the blackbody spectrum from the temperature.
        spectrum = []
        for wavelength in range(400, 551):
            bbModel = models.BlackBody(temperature=temp * u.K)
            bbSpectrum = (
                bbModel(wavelength * u.nm)
                .to(u.nJy / u.sr, equivalencies=u.spectral())
                .value
            )
            spectrum.append(bbSpectrum)

        # Find the spectrum sum.
        spectrumSum = 0
        spectrumTotal = 0
        for wavelengthIndex, wavelength in enumerate(np.linspace(400, 550, 151)):
            spectrumSum += wavelength * spectrum[wavelengthIndex]
            spectrumTotal += spectrum[wavelengthIndex]

        # Determine the effective wavelength.
        effectiveWavelength = spectrumSum / spectrumTotal

        effectiveWavelengths.append(effectiveWavelength)

    return effectiveWavelengths


def computeDifferentialRefraction(dataset, wavelengths, referenceWavelengths):
    """Compute the expected shift in apparent position due to
    wavelength-dependent atmospheric refraction, the differential chromatic
    refraction offset, for each source based on its effective and
    reference wavelengths.

        Parameters
        ----------
        dataset: `pandas.DataFrame`
            DataFrame of  matched astrometry and photometry results. This
            dataFrame contains the angular separation (astrometry) along with
            the magnitudes and difference in magnitudes of the fluxes
            associated with the two chosen bands (photometry).
        wavelength: `list`
            List of effective wavelengths; each value corresponding to an
            individual source.
        refWavelength: `list`
            List of reference wavelengths.

        Returns
        -------
        dRefraction : `numpy.array`
            Array of differential refraction values for each source.
    """
    elevation = dataset["elevation"][0]
    observatory = dataset["observatory"][0]
    dRefraction = []
    for w in wavelengths:
        refraction = differentialRefraction(w, referenceWavelengths, elevation, observatory)
        refractionInArcsec = refraction.asArcseconds()
        dRefraction.append(refractionInArcsec)
    dRefraction = np.array(dRefraction)
    return dRefraction


def hexbinDp1Paper(data):
    """Generate a hexbin plot illustrating the differential chromatic
    refraction (DCR) effect as seen in the input dataset. This visualization is
    intended specifically for the DP1 paper.

        Parameters
        ----------
        data: `pandas.DataFrame`
            DataFrame containing matched astrometric and photometric results.
            Must include angular separation (astrometry), magnitudes and
            magnitude differences between two specified bands (photometry), as
            well as the blackbody temperature, effective wavelength, and
            reference wavelength for each source.
        """
    fig, ax = plt.subplots(ncols=2, nrows=2, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0, left=0.12, bottom=0.15)

    xlim = data["differential refraction"].min(), data["differential refraction"].max()
    ylim = data["g-i mag"].min(), data["g-i mag"].max()
    hb = ax[0, 0].hexbin(
        data["parallel"], data["g-i mag"], gridsize=50, cmap=stars_cmap(), mincnt=1
    )
    ax[0, 0].set(xlim=xlim, ylim=ylim)
    ax[0, 0].set_title("Parallel", fontsize=15)
    ax[0, 0].axvline(x=0, color=accent_color(), linestyle="--")
    ax[0, 0].tick_params("x", labelbottom=False)

    ax[0, 0].text(
        0.01, 0.4, r"Mag$_{{AB}}$ (g-i)", rotation="vertical", transform=fig.transFigure
    )
    ax[0, 0].text(0.35, 0.05, r"Angular Offset (arcsec)", transform=fig.transFigure)

    hb = ax[0, 1].hexbin(
        data["perpendicular"],
        data["g-i mag"],
        gridsize=50,
        cmap=stars_cmap(),
        mincnt=1,
    )
    ax[0, 1].set(xlim=xlim, ylim=ylim)
    ax[0, 1].set_title("Perpendicular", fontsize=15)
    ax[0, 1].axvline(x=0, color=accent_color(), linestyle="--")
    ax[0, 1].tick_params("x", labelbottom=False)

    label = "Number of Sources"
    axBbox = ax[0, 1].get_position()
    cax = fig.add_axes([axBbox.x1, axBbox.y0, 0.04, axBbox.y1 - axBbox.y0])
    fig.colorbar(hb, cax=cax)
    text = cax.text(
        0.5,
        0.5,
        label,
        color="k",
        rotation="vertical",
        transform=cax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )
    text.set_path_effects(
        [pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()]
    )

    hb = ax[1, 0].hexbin(
        data["parallel"],
        data["g-i mag"],
        gridsize=50,
        bins="log",
        cmap=stars_cmap(),
        mincnt=1,
    )
    ax[1, 0].set(xlim=xlim, ylim=ylim)
    ax[1, 0].axvline(x=0, color=accent_color(), linestyle="--")

    hb = ax[1, 1].hexbin(
        data["perpendicular"],
        data["g-i mag"],
        gridsize=50,
        bins="log",
        cmap=stars_cmap(),
        mincnt=1,
    )
    ax[1, 1].set(xlim=xlim, ylim=ylim)
    ax[1, 1].axvline(x=0, color=accent_color(), linestyle="--")
    label2 = "Log(Number of Sources)"
    axBbox = ax[1, 1].get_position()
    cax = fig.add_axes([axBbox.x1, axBbox.y0, 0.04, axBbox.y1 - axBbox.y0])
    fig.colorbar(hb, cax=cax)
    text = cax.text(
        0.5,
        0.5,
        label2,
        color="k",
        rotation="vertical",
        transform=cax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )
    text.set_path_effects(
        [pathEffects.Stroke(linewidth=3, foreground="w"), pathEffects.Normal()]
    )

    plt.show()
