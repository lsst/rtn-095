import numpy as np
import treecorr


class compEbTreecorr:

    def __init__(self, x, y, dx, dy, rmin=5.0 / 3600.0, rmax=1.5, dlogr=0.05):

        self.catalog = treecorr.Catalog(x=x, y=y, v1=dx, v2=dy)
        self.vvCorr = treecorr.VVCorrelation(min_sep=rmin, max_sep=rmax, bin_size=dlogr)

    def vcorr(self):

        self.vvCorr.process(self.catalog)

    def xiB(self, logr, xiplus, ximinus):
        """
        Return estimate of pure B-mode correlation function
        """
        # Integral of d(log r) ximinus(r) from r to infty:
        dlogr = np.zeros_like(logr)
        dlogr[1:-1] = 0.5 * (logr[2:] - logr[:-2])
        tmp = np.array(ximinus) * dlogr
        integral = np.cumsum(tmp[::-1])[::-1]
        return 0.5 * (xiplus - ximinus) + integral

    def comp_eb(self):
        self.vcorr()
        xib = self.xiB(self.vvCorr.logr, self.vvCorr.xip, self.vvCorr.xim)
        xie = self.vvCorr.xip - xib
        return xie, xib, self.vvCorr.logr


def comp_eb_treecorr(u, v, du, dv, **kwargs):
    """
    Compute E/B decomposition of astrometric error correlation function

    Parameters
    ----------
    u, v : array_like. positions of objects.
    du, dv : array_like. astrometric shift.

    returns
    -------
    xie, xib, logr : array_like. E-mode, B-mode,
    and log of binned distance separation in 2-point correlation function.
    """
    cebt = compEbTreecorr(u, v, du, dv, **kwargs)
    xie, xib, logr = cebt.comp_eb()
    return xie, xib, logr