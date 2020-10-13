import numpy as np
import pandas as pd
from scipy import stats
from scipy import special
from scipy import optimize
import matplotlib.pyplot as plt
from itertools import combinations

class ndimSkewNormal():
        
    def __init__(self, loc=0, scale=1, cov=1, alpha=0):
        
        '''
        For a skew gaussian of N dimension:
          loc    = 1d array of shape N
          scale = 1d array of shape N
          cov   = 2d array of shape NxN
          skew  = 1d array of shape N
         
        ndimSkewNormal(X) = Normal( X.T * cov * X ) x NormalCumulative( alpha.T * X )
        X -> (X-loc)/scale
        '''
        
        # conversion in numpy array
        loc  = np.array(loc)
        scale = np.array(scale)
        cov   = np.array(cov)
        alpha = np.array(alpha)

        # Convert parameter into numpy array for 1D case
        if loc.ndim == 0:
            loc    = loc[np.newaxis]
            scale = scale[np.newaxis]
            cov   = cov[np.newaxis, np.newaxis]
            alpha = alpha[np.newaxis]

        # Regularization of covariance matrix if at least a matrix
        if loc.ndim > 0:
            cov = cov + np.eye(loc.shape[0])*1e-12
                    
        # Assign attributes
        self.loc   = loc
        self.alpha = alpha
        self.scale = scale
        self.cov   = cov
        self.dim   = loc.shape[0]

        
    def pdf(self, x):
        
        '''
        Compute the PDF value in a point in the N-dimension space.
        Two shapes are accepted for x:
          - 1 point : 1D array of shape N and pdf(x) returns a number
          - n points: 2D array of shape nxN and pdf(x) returns an array of shape n.
        '''
        
        # Format input properly
        x = np.array(x)
        if x.ndim == 0:
            x = x[np.newaxis]
        else:
            x = x.reshape(-1, self.dim)

        scale = np.diag(self.scale)
        cov   = scale @ self.cov @ scale
            
        # Compute the gaussian part
        multinorm = stats.multivariate_normal.pdf(x-self.loc, cov=cov)        
        
        # Compute the skewed part
        alpha = np.matmul(self.alpha, np.linalg.inv(scale))[np.newaxis, :]   
        xNorm = (x-self.loc)[..., np.newaxis]
        beta  = np.squeeze(np.matmul(alpha, xNorm))
        cdf   = stats.norm.cdf(beta)
        
        # Return the result
        return 2 * multinorm * cdf

    def _check1D(self):
        if self.dim > 1:
            raise NameError('skewnorm must be of 1 dimension')
        else:
            return True
        
    # Get knowns quantity taken from
    # https://en.wikipedia.org/wiki/Skew_normal_distribution
    def alpha1D(self):
        ''' Return alpha, as a scalar for 1D PDF '''
        if self._check1D():
            return np.squeeze(self.alpha)
        
    def cdf1D(self, x):
        if self._check1D():
            x = np.squeeze( (x-self.loc) / self.scale )
            s = np.squeeze(self.alpha)
            return stats.norm.cdf(x) - 2 * special.owens_t(x, s)

    def mode1D(self):
        if self._check1D():
            loc   = np.squeeze(self.loc)
            scale = np.squeeze(self.scale)
            skew  = np.squeeze(self.alpha)
            muZ   = skew / np.sqrt(1+skew**2) * np.sqrt(2.0/np.pi)
            gam1  = self.skew1D()
            if skew != 0:
                sign = skew / np.abs(skew)
                m0 = muZ - gam1*np.sqrt(1-muZ**2)/2  - sign/2 * np.exp(-2*np.pi / np.abs(skew))
                return loc + scale * m0
            else:
                return loc

    def mean1D(self):
        if self._check1D():
            loc   = np.squeeze(self.loc)
            scale = np.squeeze(self.scale)
            skew  = np.squeeze(self.alpha)
            delta = skew / np.sqrt(1+skew**2)
            return loc + scale * delta * np.sqrt(2.0/np.pi)
    
    def variance1D(self):
        if self._check1D():
            scale = np.squeeze(self.scale)
            skew  = np.squeeze(self.alpha)
            delta = skew / np.sqrt(1+skew**2)
            return scale**2 * (1 - 2*delta**2/np.pi)

    def std1D(self):
        if self._check1D():
            return self.variance1D()**0.5
    
    def skew1D(self):
        if self._check1D():
            loc   = np.squeeze(self.loc)
            scale = np.squeeze(self.scale)
            skew  = np.squeeze(self.alpha)
            delta = skew / np.sqrt(1+skew**2)
            return (4-np.pi)/2 * delta**3 * (2/np.pi)**(3./2.) / (1-2*delta**2/np.pi)**(3./2.)
        
    def interval1D(self, x0, x1):
        if self._check1D():
            return self.cdf1D(x1) - self.cdf1D(x0)
        
    def borders(self, nSigma=3):
        '''
        Return an array of shape (ndim, 2) with xmin, xmax along
        each dimension corresponding to mode-nSigma*errMin,  
        mode+nSigma*errPos.
        '''
        borders = []
        for i in range(self.dim):
            loc, scale, alpha = self.loc[i], self.scale[i], self.alpha[i]
            sn_tmp = ndimSkewNormal(loc=loc, scale=scale, alpha=alpha)
            v, m, p = sn_tmp.measAsymError()
            borders.append([v-nSigma*m, v+nSigma*p])
        return np.array(borders)

    def measAsymError(self, CI=0.68):
        '''
        return central value, negative error, positive error
        corresponding to a confidence interval of 68%. This
        function search for [x1, x2] such as
          1. x1 < mode < x2, where mode is defined by pdf(mode) being maxmimal.
          2. pdf(x1) = pdf(x2)
          3. interval(x1, x2) = 68% (or any confidence interval specified as CI).
        The function returns mode, mode-x1, x2-mode
        '''
        
        mode = self.mode1D()
        std  = self.std1D()
    
        def beMin(x):
            eNeg, ePos = x
            cl = self.interval1D(mode-eNeg, mode+ePos)
            penalty = self.pdf(mode-eNeg) - self.pdf(mode+ePos)
            return np.abs(cl-CI) + np.abs(penalty)

        # Minimisation
        x0 = [std, std]
        res = optimize.minimize(beMin, x0, tol=1e-4, method='Nelder-Mead')
        eNeg, ePos = res.x
        
        # Return central, negative, positive values
        return mode, eNeg, ePos
        

    def plot(self, borders=[], varNames=[], nPoints=50, kwargs_scatter={}, kwargs_coutour={}, **kwargs):

        '''
        Visualisation of the multi-dimensional PDF using input parameters to build
         - 1D PDF on diagonal element
         - 2D PDF on off-diagonal elements

        borders  = list of pairs being (min, max) along each dimension.
        varNames = list of string being variable names
        nPoints  = number of points to be scanned along each dimension
        '''
        
        # Plot scatter matrix
        N = self.dim
        
        # Range for plotting
        if len(borders) == 0:
            borders = self.borders()
        ranges = [np.linspace(borders[i][0], borders[i][1], nPoints) for i in range(N)]
        
        fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(13, 13))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        for ax in axes.flat:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        for i in range(N):
            for j in range(N):  
                    
                if i == j:
                    loc, scale, alpha = self.loc[i], self.scale[i], self.alpha[i]
                    snDiag = ndimSkewNormal(loc=loc, scale=scale, alpha=alpha)
                    axes[i, j].plot(ranges[i], snDiag.pdf(ranges[i]))
                else:
                    locIJ   = [  self.loc[j],   self.loc[i]]
                    scaleIJ = [self.scale[j], self.scale[i]]
                    alphaIJ = [self.alpha[j], self.alpha[i]]
                    rho     = self.cov[j, i]
                    covIJ  = [
                        [   1, rho],
                        [ rho,   1],
                    ]
                    sn = ndimSkewNormal(locIJ, scaleIJ, covIJ, alphaIJ)
                    plotPdf2D(ranges[j], ranges[i], sn.pdf, ax=axes[i, j],
                              kwargs_scatter=kwargs_scatter,
                              kwargs_coutour=kwargs_coutour,
                              **kwargs)
                
                # Axis labels
                if varNames:
                    axes[i, j].set_xlabel(varNames[j])
                    axes[i, j].set_ylabel(varNames[i])
                
                # Make ticks and label appear only of left & bottom part
                if i == 0:
                    axes[j, i].yaxis.set_visible(True)
                    
                if j == self.dim-1:
                    axes[j, i].xaxis.set_visible(True)


    def generateDataFromGauss(self, N=10000, chunck=10000, sigmaScale=3):

        '''
        [DEPRECIATED]
        
        Sampling based on gaussians PDFs.
        N          : number of wanted toys
        chunck     : number of one run
        sigmaScale : scale factor for the sigma of the underlying gaussian 
                        -> sigma = max(eNeg, ePos) * sigmaScale
        '''

        # Useful parameters
        Ndim  = self.dim
        covG = np.zeros_like(self.cov)
        muG  = np.zeros_like(self.loc)
        siG  = np.zeros_like(self.loc)

        # Get (mean, eNeg, ePos) in each dimension
        for i in range(Ndim):
            sn = ndimSkewNormal(loc=self.loc[i], scale=self.scale[i], alpha=self.alpha[i])
            val, eNeg, ePos = sn.measAsymError()
            muG[i] = val
            siG[i] = max(eNeg, ePos) * sigmaScale

        # Compute the covariance matrix
        for i in range(Ndim):
            for j in range(Ndim):
                covG[i, j] = self.cov[i, j] * siG[i] * siG[j]

        nToys, data = 0, []
        while nToys<N:
            Xs = np.random.multivariate_normal(mean=muG, cov=covG, size=chunck)
            Ys = np.random.rand(chunck) * np.max(self.pdf(Xs))
            data.append(Xs[self.pdf(Xs)>=Ys])
            nToys += data[-1].shape[0]

        return np.concatenate(data)
                    

    def generateDataLog(self, n=200000, borders=[]):
    
        '''
        Return data following log(PDF) and weights 
        to get back to a distrubtion following the 
        original PDF. 

        Advantages wrt apdf.generateData(pdf)
          - better tail population
          - better sampling efficiency

        Drawbacks wrt apdf.generateData(pdf)
          - weighted sample 

        return data, weights
          - data   : 2D array of shape (Naccepted, Ndim)
          - weights: 1D array of shape (Naccepted)
        '''

        # Range for data generations
        if len(borders) == 0:
            borders = self.borders(10)

        # PDF
        pdf = self.pdf
            
        # Manage borders and dimension
        borders = np.array(borders)
        nDim    = borders.shape[0]
        
        # Initial space {x1, ... , xn} sampling
        Xs = np.array([np.random.rand(n)*(borders[i][1]-borders[i][0]) + borders[i][0] for i in range(nDim)]).T 
        
        # Final space y sampling
        logY = np.log(pdf(Xs) + 1e-120)
        yMin, yMax =  np.min(logY), np.max(logY)    
        Ys = np.random.rand(n) * (yMax-yMin) + yMin
        
        # Select accepted points
        data = Xs[logY>=Ys]
        
        # Compute weights
        weights = pdf(data)
        
        # Return data and weights
        return data, weights


    
def paramFromMeas(cVal, eNeg, ePos):
    '''
    Return the 3 parameters (loc, scale, alpha) of a 1D skew normal 
    PDF corresponding to a measurement (ie central value, negative 
    and positive error). These parameters are obtained assuming
    the numbers relate to a 68% confidence interval.
    '''
    c, m, p = cVal, eNeg, ePos
    r = p/m
    df = pd.read_csv('ParameterTable.csv')
    dfClosest = df.iloc[(df['pos/neg']-r).abs().argsort()[:1]]
    mode   = dfClosest['mode'].values[0]
    alpha  = dfClosest['alpha'].values[0]
    scaleP = p / dfClosest['pos'].values[0]
    scaleN = m / dfClosest['neg'].values[0]
    scale = (scaleP+scaleN) / 2.0
    loc = c - mode*scale
    
    if np.abs(alpha)>=18:
        print('Warning: shape parameter reached maximum tabulated value.')
        print('-> the PDF will not accurately describe the measurement')
    
    return loc, scale, alpha

    
def nDimGrid(*Xs):
    
    '''
    Create N-dimension grid, as list of Nxn points from N 1D array of n values.
    It returns a 2D array of a shape (Nxn, N), each line being a point 
    in the n-dim space.
   
    For example, to create a 3D grid in a [0, 1] x [0, 1] x [0, 1] cube
      - 10 values along the x-axis
      - 15 values along the y-axis
      - 20 values along the z-axis
    
    One would do the following
    
    >>> Xs = np.linspace(0, 1, 10)
    >>> Ys = np.linspace(0, 1, 15)
    >>> Zs = np.linspace(0, 1, 20)
    >>> grid = nDimGrid(Xs, Ys, Zs)
    >>> print(grid.shape) 
    >>>
    >>> (3000, 3)
    '''    
    return np.array(np.meshgrid(*Xs)).T.reshape(-1, len(Xs))


def plotPdf2D(Xs, Ys, PDF2d, kwargs_scatter={}, kwargs_coutour={}, **kwargs):
    
    '''
    Xs, Ys: 1D-array
    PDF: callable f(points) where points is a 2D array of 
         a shape (n, 2) returning 1d array of n values.
    '''

    # Default keywords arguments
    ax, contour, scatter = None, True, True
    if 'ax' in  kwargs:
        ax = kwargs['ax']
    if 'contour' in kwargs:
        contour = kwargs['contour']
    if 'scatter' in kwargs:
        scatter = kwargs['scatter']
    
    # Creates all (x, y) points from Xs and Ys
    points = nDimGrid(Xs, Ys)

    # Compute the proba for all these pairs
    proba = PDF2d(points)

    # Plot on existing axes of create a new plot
    obj = plt
    if ax: obj = ax
        
    # Plot it using a scatter
    if scatter:
        x, y = points[:, 0], points[:, 1]
        obj.scatter(x, y, c=proba, **kwargs_scatter);

    # Produce a contour plot, coming back to meshgrid
    if contour:
        XX, YY = np.meshgrid(Xs, Ys)
        PP = proba.reshape(len(Xs), len(Ys)).T
        obj.contour(XX, YY, PP, **kwargs_coutour);


def plotScatterMatrix(data, varNames=[], nPoints=1000, kwargs_hist={}, kwargs_scatter={}):

    '''
    Plot projection over all variables pairs (Xi, Xj) of a
    dataset with N points of n-dimension.
    
    data: array with a shape (N, n)
    '''
    
    N = data.shape[1]
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(13, 13))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        
    step = int( data.shape[0] / nPoints )
    if step == 0 or nPoints <= 0:
        step = 1
    d = data[::step]
        
    for i in range(N):
        for j in range(N):
            
            if i == j:
                axes[i, i].hist(d[:, i], **kwargs_hist)
            else:
                axes[i, j].scatter(d[:, j], d[:, i], **kwargs_scatter)
                
            # Axis labels
            if varNames:
                axes[i, j].set_xlabel(varNames[j])
                axes[i, j].set_ylabel(varNames[i])
                
            # Make ticks and label appear only of left & bottom part
            if i == 0:
                axes[j, i].yaxis.set_visible(True)
                    
            if j == N-1:
                axes[j, i].xaxis.set_visible(True)

        
def plotPdfProj2D(points, PDF, kwargs_scatter={}):
    
    '''
    Plot projection over all variables pairs (Xi, Xj) of a dimension n
    probability density function evaluated overs N points {(x1, ..., xNdim)}_N:
     --> probability = PDF(X1, ..., XNdim)
    
    points: array with a shape (N, n)
    PDF: callable f(points) where points is a 2D array of 
         a shape (N, n) returning 1d array of N values.
    '''

    # Compute the probabilities
    proba = PDF(points)
    
    # Sort points into increasing proba - for cosmetic purpose
    sIdx = np.argsort(proba)
    probaS = proba[sIdx]
    pointS = points[sIdx]

    kwargs_scatter['c'] = probaS
    plotDataProj2D(pointS, kwargs_scatter)


def generateData(pdf, n=10000, borders=[[-5, 5]]):
    
    '''
    Return a dataset distributed according the given pdf. This is an array of
    shape (Naccepted, Ndim) where Naccepted is a priori unknown.
    
    pdf    : callable pdf(x) where x can be a 2D array of shape (Npoints, Ndim)
    n      : number of generated toys
    borders: 2D array of shape (Ndim, 2) containing limits which defined variable space scan.
    '''
    
    # Limits and dimension of initial space
    borders = np.array(borders)
    nDim    = borders.shape[0]
    
    # Border values for PDF
    Xs = np.array([np.random.rand(n)*(borders[i][1]-borders[i][0]) + borders[i][0] for i in range(nDim)]).T
    Ys = np.random.rand(n) * np.max(pdf(Xs))
    
    # Return x values of kept points
    return Xs[pdf(Xs)>=Ys]


def generateDataLog(pdf, n=10000, borders=[[-5, 5]]):
    
    '''
    Return data following log(PDF) and weights 
    to get back to a distrubtion following PDF.
    '''
    
    # Manage borders and dimension
    borders = np.array(borders)
    nDim    = borders.shape[0]
    
    # Initial space {x1, ... , xn} sampling
    Xs = np.array([np.random.rand(n)*(borders[i][1]-borders[i][0]) + borders[i][0] for i in range(nDim)]).T 
    
    # Final space y sampling
    logY = np.log(pdf(Xs) + 1e-120)
    yMin, yMax =  np.min(logY), np.max(logY)    
    Ys = np.random.rand(n) * (yMax-yMin) + yMin

    # Select accepted points
    data = Xs[logY>=Ys]

    # Compute weights
    weights = pdf(data)
    
    # Return data and weights
    return data, weights


def fitWeightedDistribution(data, weights, nbins=100):

    '''
    This functions performs a fit of weighted binned data by
    a one dimension skew normal function.

    data    = 1D array of shape N
    weights = 1D array of shape N
    nbins   = number of bins

    Return skew-normal parmaters: (loc, scale, alpha)
    '''
    
    # Weighted histogram
    Nbins    = nbins
    bins     = np.linspace(data.min(), data.max(), Nbins)
    ydata, _ = np.histogram(data, weights=weights, bins=bins, density=True)

    # Bin center as x data
    xdata = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

    # Loss function
    def loss(x):
        l, s, a = x
        f = apdf.ndimSkewNormal(loc=l, scale=s, alpha=a).pdf
        w = np.sqrt(np.abs(xdata - xdata.mean()))
        w = w/w.max()
        dy2 = (ydata - f(xdata))**2
        return np.sum(dy2*w)

    l0 = np.average(data, weights=weights) 
    s0 = np.sqrt( np.average((data-l0)**2, weights=weights) )
    a0 = 1.

    res = optimize.minimize(loss, tol=1e-6, x0=[l0, s0, a0], method='Nelder-Mead')
    loc, scale, alpha = res.x
    
    return loc, scale, alpha
