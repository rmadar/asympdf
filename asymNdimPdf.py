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
        
        # Assign attributes
        self.loc   = loc
        self.alpha = alpha
        self.scale = np.diag(scale)
        self.cov   = self.scale @ cov @ self.scale
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
        
        # Compute the gaussian part
        multinorm = stats.multivariate_normal.pdf(x-self.loc, cov=self.cov)        
        
        # Compute the skewed part
        alpha = np.matmul(self.alpha, np.linalg.inv(self.scale))[np.newaxis, :]   
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
            penalty = np.abs(self.pdf(mode-eNeg) - self.pdf(mode+ePos))**2
            return np.abs(cl-CI) + np.abs(penalty)

        # Minimisation
        x0 = [std, std]
        res = optimize.minimize(beMin, x0, tol=1e-3, method='Nelder-Mead')
        eNeg, ePos = res.x

        # Return central, negative, positive values
        return mode, eNeg, ePos
        

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
    loc = c-mode*scale
    
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


def plotPdf2D(Xs, Ys, PDF2d, kwargs_scatter={}, kwargs_coutour={}):
    
    '''
    Xs, Ys: 1D-array
    PDF: callable f(points) where points is a 2D array of 
         a shape (n, 2) returning 1d array of n values.
    '''
    
    # Creates all (x, y) points from Xs and Ys
    points = nDimGrid(Xs, Ys)

    # Compute the proba for all these pairs
    proba = PDF2d(points)
    
    # Plot it using a scatter
    x, y = points[:, 0], points[:, 1]
    plt.scatter(x, y, c=proba, **kwargs_scatter);

    # Produce a contour plot, coming back to meshgrid
    XX, YY = np.meshgrid(Xs, Ys)
    PP = proba.reshape(len(Xs), len(Ys)).T
    plt.contour(XX, YY, PP, **kwargs_coutour);


def plotDataProj2D(data, kwargs_scatter={}):
    '''
    Plot projection over all variables pairs (Xi, Xj) of a
    dataset with N points of n-dimension.
    
    data: array with a shape (N, n)
    '''

    # Create all variable pairs (with indices)
    pairIdx = list(combinations(range(data.shape[1]), 2))

    # Plot scatter matrix
    N = len(pairIdx)/2
    plt.figure(figsize=(5*N, 5*N))

    # Plot projection for each pair of variables
    for i, j in pairIdx:
        iPlot = i*N + j
        plt.subplot(N, N, iPlot)
        plt.scatter(data[:, j], data[:, i], **kwargs_scatter)
        plt.xlabel('var{}'.format(j))
        plt.ylabel('var{}'.format(i))

        
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


def generateData(pdf, n=10000, xLim=[[-5, 5]]):
    
    '''
    Return a dataset distributed according the given pdf. This is an array of
    shape (Naccepted, Ndim) where Naccepted is a priori unknown.
    
    pdf : callable pdf(x) where x can be a 2D array of shape (Npoints, Ndim)
    n   : number of generated toys
    xLim: 2D array of shape (Ndim, 2) containing limits which defined variable space scan.
    '''
    
    # Limits and dimension of initial space
    xLim = np.array(xLim)
    nDim = xLim.shape[0]
    
    # Border values for PDF
    Xs = np.array([np.random.rand(n)*(xLim[i][1]-xLim[i][0]) + xLim[i][0] for i in range(nDim)]).T
    Ys = np.random.rand(n) * np.max(pdf(Xs))
    
    # Return x values of kept points
    return Xs[pdf(Xs)>=Ys]
