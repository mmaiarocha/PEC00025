# -*- coding: utf-8 -*-

import sys
import gzip   as gz
import pickle as pk
import numpy  as np
import pandas as pd

from   warnings          import warn
from   scipy.interpolate import interp1d
from   scipy.optimize    import curve_fit

import matplotlib.pyplot as plt

#=============================================================================
#=============================================================================
class MRPy(np.ndarray):
#=============================================================================
#=============================================================================
# 1. Class initialization
#=============================================================================
    
    def __new__(cls, np_array, fs=None, Td=None):

        X  =  np.asarray(np_array).view(cls)

        if (X.size == 0):
            sys.exit('Empty array not allowed for new objects!')
        
        sh =  X.shape
        
        if (len(sh) == 1): 
            X  = np.reshape(X,(1,sh[0]))
        elif (sh[0] > sh[1]):
            X = X.T
            
        sh   =  X.shape
        X.NX =  sh[0]
        X.N  =  sh[1]
        
        if (X.N < 2):
            sys.exit('Come on!!! Start with at least 2 elements!')

        err =  1.0
        if (np.mod(X.N, 2) != 0):         # enforce N to be even...
            X   =  X[:,:-1]               # ... odd element is discarded!!!
            err = (X.N - 1)/X.N           # correction over Td
            X.N =  X.N - 1

        if (fs != None):                  # if fs is prescribed...
            X.fs = np.float(fs)
            X.Td = X.N/X.fs               # ... Td is calculated

        elif (Td != None):                # but if Td is prescribed...
            X.Td = err*np.float(Td)
            X.fs = X.N/X.Td               # ... fs is calculated

        else: sys.exit('Either fs or Td must be provided!')

        X.M  = X.N//2 + 1
        return X

#-----------------------------------------------------------------------------

    def __array_finalize__(self, X):
        
        if X is None: return
        
        self.fs = getattr(X, 'fs', None)
        self.Td = getattr(X, 'Td', None)
        self.NX = getattr(X, 'NX', None)
        self.N  = getattr(X, 'N',  None)
        self.M  = getattr(X, 'M',  None)

#=============================================================================
# 2. Class constructors from other sources
#=============================================================================

    def from_file(filename, form='mrpy'):
        """
        Load time series from file. Please contact the author for 
        including other types of datafile.
 
        Parameters:  filename: file to be loaded, including path,
                               whithout file extension
                     form:     data formatting. Options are
                               'mrpy'      - default gzip pickle loading
                               'excel'     - excel generated with pandas
                               'columns'   - t, X1, [X2, X3 ...]
                               'invh     ' - csv file by iNVH app
                               'mpu6050'   - gzip excel 6 axis data
        """

        try:
            
#-------------        
            if (form.lower() == 'mrpy'):
                with gz.GzipFile(filename+'.csv.gz', 'rb') as target:                 
                    return MRPy(*pk.load(target))

#---------------     
            elif (form.lower() == 'excel'):
                with open(filename+'.xlsx', 'rb') as target:
                    
                    data =  pd.read_excel(target, 
                                          index_col=0, 
                                          sheet_name='MRPy')
                    
                    ti   =  np.array(data.index, dtype=float)
                    return MRPy.resampling(ti, data.values)

#--------------- 
            elif (form.lower() == 'columns'):
                with open(filename+'.txt', 'rb') as target:
    
                    data = np.genfromtxt(target, 
                                         delimiter='\t')
        
                    ti   = data[:,0]
                    return MRPy.resampling(ti, data[:,1:])

#---------------    
            elif (form.lower() == 'invh'):
                with open(filename+'.csv', 'rb') as target:
                
                    data =  np.genfromtxt(target, 
                                          delimiter=',',
                                          skip_header=1)
                    
                    ti   =  data[:,0]
                    return MRPy.resampling(ti, data[:,1:-1])
    
#---------------    
            elif (form.lower() == 'mpu6050'):
                with gz.open(filename+'.csv.gz', 'rb') as target:
                
                    data =  np.genfromtxt(target, 
                                          delimiter=',')
                    
                    ti   =  data[:,0] - data[0,0]
                    return MRPy.resampling(ti, data[:,1:]/16384)

#--------------- 
            else:
                sys.exit('Data formatting not available!')
                return None
            
        except:
            sys.exit('Could not read file "{0}"!'.format(filename))
            return None

#-----------------------------------------------------------------------------
    
    def from_periodogram(Sx, fs):
        """
        Simulate RPs from given spectral densities.
 
        Parameters:  Sx:  spectral densities as ndarray (must have odd
                          length, otherwise it will be truncated by 1 and
                          the length of simulation will not be as expected!)
                          The largest dimension of Sx is assumed to be the
                          frequency axis.
                     fs:  sampling frequency in Hz
        """

        sh  =  Sx.shape

        if (len(sh) == 1): 
            Sx = np.reshape(Sx,(1,sh[0]))
        elif (sh[0] > sh[1]):
            Sx = Sx.T

        sh  =  Sx.shape
        NX  =  sh[0]
        M0  =  sh[1]
        M   =  M0 - (np.mod(M0,2) == 0)     # ensure M is odd
        N   =  2*(M - 1)
        
        Sx  =  N*fs*np.abs(Sx[:,0:M])/2
        X   =  np.empty((NX, N))
        
        for k in range(NX):

            phi =  2*np.pi*np.random.rand(M);
            phi[0] = 0.

            Pw  =  np.sqrt(Sx[k,:]) * (np.cos(phi) + 1j*np.sin(phi))
            Pw  =  np.hstack((Pw, np.conj(Pw[-2:0:-1])))

            X[k,:]  =  np.real(np.fft.ifft(Pw))

        return MRPy(X, fs) 

#-----------------------------------------------------------------------------

    def from_autocov(Cx, Tmax):
        """
        Simulate RPs from given autocovariance functions.
 
        Parameters:  Cx:   autocovariances as ndarray (must have odd
                           length, otherwise it will be truncated by 1 and
                           the length of simulation will not be as expected!)
                           The largest dimension of Cx is assumed to be the
                           time gap axis.
                     Tmax: largest time gap, associated to the last element
                           in array Cx. Defines process duration, which 
                           will be approximately 2Tmax.
        """

        Sx, fs = MRPy.Cx2Sx(Cx, Tmax)
        return MRPy.from_periodogram(Sx, fs)

#-----------------------------------------------------------------------------

    def from_pseudo(Sa, Tmax, T, zeta=0.05):  # NOT READY!!!
        """
        Simulate ground acceleration records from a given pseudo acceleration 
        spectra.  
 
        Parameters:  Sa:   acceleration pseudo spectra as ndarray (must have 
                           odd length, otherwise it will be truncated by 1 and
                           the length of simulation will not be as expected!)
                           The largest dimension of Sa is assumed to be the
                           period axis.
                     Tmax: largest period, associated to the last element
                           in array Sa. Defines process duration, which
                           will be approximately 2Tmax. 
                     T:    tuple (T1, T2, T0) defining envelope timing, where
                           T1 is the end of attack time, T2 is the end of
                           sustain time, and T0 is the time constant of the
                           exponential amplitude decay.
                     zeta: system damping (ratio of critical) can be 
                           provided or default value of 5% is assumed.
        """

        sh  =  Sa.shape

        if (len(sh) == 1): 
            Sa = np.reshape(Sa,(1,sh[0]))
        else:
            if (sh[0] > sh[1]):
                Sa = Sa.T

        sh  =  Sa.shape
        NX  =  sh[0]
        M0  =  sh[1]
        M   =  M0 - (np.mod(M0,2) == 0)     # ensure M is odd
        N   =  2*(M - 1)

        err =  M/M0             
        fs  = (M - 1)/(err*Tmax)            # eventually corrects for Tmax
        f   =  np.linspace(0, fs/2, M)
        tau =  np.linspace(0, Tmax, M)
        
        fi      =  np.zeros(M)
        fi[1:]  =  1/tau[:0:-1]  
        
        X   =  np.empty((NX,N))
        
        for k in range(NX):

            Sxi     = np.zeros(M)                        
            Sxi[1:] = Sa[k,1:]
            fSx = interp1d(fi, Sxi, kind='quadratic') 
            
            X[k]  = MRPy.from_periodogram(fSx(f), fs)
            X[k] *= Sa.max()/X[k].max()

        return MRPy(X, fs).Kanai().envelope(T)

#=============================================================================
# 3. Class constructors by modification
#=============================================================================

    def zero_mean(self):
        """
        Clean mean values.
        """

        X   = MRPy.copy(self)
        Xm  = X.mean(axis=1) 

        for k in range(self.NX):  
            X[k,:] -= Xm[k]

        return X

#-----------------------------------------------------------------------------

    def superpose(self, weight=1.):
        """
        Add up all series in MRPy weighted by 'weight', which is a
        scalar or list with weights for summation. This method can be
        used, for instance, for combining modal responses according to 
        modal shape coordinate at a given structural node.
        """

        if ~hasattr(weight, "__len__"):
            weight = weight*np.ones(self.NX)
            
        elif (len(weight) != self.NX):
            sys.exit('Weight length must equal number of processes!')

        X   = np.zeros((1, self.N))

        for kX, row in enumerate(self):  
            X[0,:] += weight[kX]*row

        return MRPy(X, self.fs)

#-----------------------------------------------------------------------------

    def double(self):
        """
        Double MRPy duration by filling with mean values.
        """

        Xm  = self.mean(axis=1) 
        X   = np.hstack((self, np.tile(Xm,(self.N, 1)).T))

        return MRPy(X, self.fs)

#-----------------------------------------------------------------------------

    def extract(self, segm=(1/4, 3/4), by='fraction'):
        """
        Extract a central segment of time range. The lower and upper
        cutting point as defined as a tuple or list, which meaning is
        defined by a code 'by':

        Parameters:  segm: tuple or list with the lower and upper 
                           cutting points.
                     by:   code indicating the meaning of cutting points:
                           'fraction': default meaning 
                           'time'    : time axis related
                           'index'   : directly through indexing
        """

        if (segm[0] >= segm[1]):
            sys.exit('Upper limit must be larger than lower limit!')

        if (by.lower() == 'fraction'):            
            i0  = int(segm[0]*self.N)
            i1  = int(segm[1]*self.N)
            
        elif (by.lower() == 'time'):            
            i0  = int(segm[0]*self.fs)
            i1  = int(segm[1]*self.fs)
            
        elif (by.lower() == 'index'):            
            i0  = int(segm[0])
            i1  = int(segm[1])
        
        else: 
            sys.exit('Segment definition code is unknown!')
            return None

        i1  = i1 - np.mod(i1-i0, 2)      # ensure even length
        if (i0 < 0     ): i0 = 0         # do not go over boundaries
        if (i1 > self.N): i1 = self.N

        return MRPy(self[:,i0:i1], self.fs)

#-----------------------------------------------------------------------------

    def envelope(self, T):
        """
        Apply an amplitude envelope with exponential attack e decay.
        This type of envelope is used for simulation of seismic acceleration.
    
        Parameters:  T: tuple (T1, T2, T0) defining envelope timing, where
                        T1 is the end of attack time, T2 is the end of
                        sustain time, and T0 is the time constant of the
                        exponential amplitude attack and decay.
        """

        t   = self.t_axis()
        X   = MRPy.copy(self)
        env = np.ones(self.N)

        env[t < T[0]] = (1 - np.exp(-t[t < T[0]]/T[0]))/(1 - np.exp(-1))
        env[t > T[1]] =  np.exp((T[1] - t[t > T[1]])/T[2])

        for k in range(self.NX):  
            X[k,:] *= env

        return X

#-----------------------------------------------------------------------------
    
    def mov_average(self, n=3, win='tri'):
        """
        Apply moving average with specified window.
    
        Parameters:  n:   window width (truncated to be odd integer)
                     win: window type. Available windows are:
                          'rec': rectangular
                          'tri': triangular (default)
        """

        n =  np.int(n)               # truncate to integer
        n =  n - (1 - np.mod(n,2))   # n is odd or will be decreased by 1
        m = (n -  1)//2 + 1          # window center
        W =  np.ones(n)              # default rectangular window

        if (win.lower()   == 'rec'):
            pass
            
        elif (win.lower() == 'tri'):
            W[   :m] =  np.linspace(1/m, 1.,  m)
            W[m-1: ] =  np.linspace(1.,  1/m, m)
        
        else:
            sys.error('Averaging window type not available!')

        m  =  m - 1
        W  =  W/W.sum()
        X  =  MRPy.copy(self)

        for kX in range(self.NX):     
            
            for k in range(0, m):
                k0 = m - k
                W0 = W[k0:]/np.sum(W[k0:])
                X[kX,k] = np.sum(W0*self[kX,:k+m+1])

            for k in range(m, self.N-m-1):
                X[kX,k] = np.sum(W*self[kX,k-m:k+m+1])

            for k in range(self.N-m-1, self.N):
                k0 = m - k + self.N
                W0 = W[:k0]/np.sum(W[:k0])
                X[kX,k] = np.sum(W0*self[kX,k-m:])

        return X

#-----------------------------------------------------------------------------

    def filtered(self, band, mode='pass'):
        """
        Apply filtering in frequency domain. Series size is doubled by 
        trailing zeros before filtering, in order to minimize aliasing.
 
        Parameters:  band: frequency band as tuple or list: [f_low, f_high]
                     mode: filter type. Available:
                           'pass': band pass (default)
                           'stop': band stop
        """
  
        X   =  self.double()
        f   =  X.f_axis()
        
        b0, b1 = MRPy.check_band(self.f, band)

        for kX in range(X.NX):
            
            Xw  =  np.fft.fft(X[kX,:])[0:X.M]

            if mode.lower() == 'pass':
                Xw[(f <  b0) | (f >= b1)] = 0.
        
            elif mode.lower() == 'stop':
                Xw[(f >= b0) & (f < b1)] = 0. 

            else:
                warn('Filter type not available!')

            X[kX,:] = np.real(np.fft.ifft(
                      np.hstack((Xw, np.conj(Xw[-2:0:-1])))))

        return MRPy(X[:,0:self.N], self.fs)

#-----------------------------------------------------------------------------

    def Kanai(self, H1=(4.84, 0.60), H2=(0.97, 0.60)):
        """
        Apply Kanai/Tajimi filtering, with low frequency range attenuation 
        to avoid integration drifting. This filter is used for simulation
        of seismic acceleration.
 
        Parameters:  H1: tuple (f1, zeta1) for first filter part,
                         where default values represent firm soil condition.
                     H2: tuple (f2, zeta2) for second filter part, which
                         must properly attenuate low frequency range.
        """

        X = np.empty((self.NX, self.N))

        for kX, row in enumerate(self):

            Xw  =  np.fft.fft(row)[0:self.M]
            
            w1  =  self.f_axis()/H1[0]
            w2  =  self.f_axis()/H2[0]
            
            Hw1 = (1 + 2j*H1[1]*w1)/(1 - w1*w1 + 2j*H1[1]*w1)
            Hw2 =           (w2*w2)/(1 - w2*w2 + 2j*H2[1]*w2)
            
            Xw  =  Xw*Hw1*Hw2
            Xk  =  np.real(np.fft.ifft(np.hstack((Xw, np.conj(Xw[-2:0:-1])))))

            X[kX,:] =  Xk[0:self.N]

        return MRPy(X, self.fs)

#-----------------------------------------------------------------------------

    def integrate(self, band=None):
        """
        Frequency domain integration with passing band.
 
        Parameters:  band: frequency band to keep, tuple: (f_low, f_high)
        """

        b0, b1 = MRPy.check_band(self.fs, band)

        X  = np.empty((self.NX, self.N))
        f  = self.f_axis(); f[0] = f[1]     # avoid division by zero

        for kX, row in enumerate(self):        
        
            Xw = np.fft.fft(row)[0:self.M]
            Xw = Xw / (2j*np.pi*f)          # division means integration
            
            Xw[0]  = 0.                     # disregard integration constant
            
            Xw[(f <= b0) | (f > b1)] = 0.
            
            X[kX,:] = np.real(np.fft.ifft(
                      np.hstack((Xw, np.conj(Xw[-2:0:-1])))))

        return MRPy(X, self.fs)

#-----------------------------------------------------------------------------
    
    def differentiate(self, band=None):
        """
        Frequency domain differentiation with passing band.
 
        Parameters:  band: frequency band to keep, tuple: (f_low, f_high)
        """

        b0, b1 = MRPy.check_band(self.fs, band)

        X  = np.empty((self.NX, self.N))
        f  = self.f_axis(); f[0] = f[1]     # avoid division by zero

        for kX, row in enumerate(self):        
        
            Xw = np.fft.fft(row)[0:self.M]
            Xw = Xw * (2j*np.pi*f)          # multiplication means derivation
            
            Xw[(f <= b0) | (f > b1)] = 0.
            
            X[kX,:] = np.real(np.fft.ifft(
                      np.hstack((Xw, np.conj(Xw[-2:0:-1])))))

        return MRPy(X, self.fs)

#-----------------------------------------------------------------------------

    def sdof_fdiff(self, fn, zeta, U0=0., V0=0.):
        """
        Integrates the dynamic equilibrium differential equation by 
        the central finite differences method.
        The input is assumed to be an acceleration (force over mass),
        otherwise the result must be divided by system mass to have
        displacement unit.
        System properties (frequency and damping) may be provided as 
        scalars or lists. If they are scalars, same properties are used
        for all series in the MRP. The same applies for initial conditions
        U0 (displacement) and V0 (velocity)
    
        Parameters:  fn:   sdof natural frequency (Hz)
                     zeta: sdof damping as ratio of critial  (nondim)
                     U0:   initial position (default is all zero)
                     V0:   initial velocity (default is all zero)
        """
        
        if ~hasattr(fn, "__len__"):
            fn   = fn*np.ones(self.NX)
    
        if ~hasattr(zeta, "__len__"):
            zeta = zeta*np.ones(self.NX)
        
        if ~hasattr(U0, "__len__"):
            U0   = U0*np.ones(self.NX)
        
        if ~hasattr(V0, "__len__"):
            V0   = V0*np.ones(self.NX)

        dt  =  1/self.fs
        X   =  MRPy(np.empty((self.NX, self.N)), self.fs)

        for kX, row in enumerate(self):

            zt  =  zeta[kX]
            wn  =  2*np.pi*fn[kX]   

            b1 = (   zt*wn + 1/dt)/dt
            b2 = (   zt*wn - 1/dt)/dt
            b3 = (dt*wn*wn - 2/dt)/dt
        
            X[kX,0]  =  U0
            X[kX,1]  =  U0 + V0*dt + row[0]*dt*dt/2
            
            for k in range(2,self.N): 
                X[kX,k] = (row[k-1] + b2*X[kX,k-2] - b3*X[kX,k-1])/b1
    
        return X

#-----------------------------------------------------------------------------

    def sdof_Duhamel(self, fn, zeta, U0=0., V0=0.):
        """
        Integrates the dynamic equilibrium differential equation by Duhamel.
        The input is assumed to be an acceleration (force over mass),
        otherwise the result must be divided by system mass to have
        displacement unit.
        System properties (frequency and damping) may be provided as 
        scalars or lists. If they are scalars, same properties are used
        for all series in the MRP. The same applies for initial conditions
        U0 (displacement) and V0 (velocity)
    
        Parameters:  fn:   sdof natural frequency (Hz)
                     zeta: sdof damping as ratio of critial  (nondim)
                     U0:   initial position (default is all zero)
                     V0:   initial velocity (default is all zero)
        """
        
        if ~hasattr(fn, "__len__"):
            fn   = fn*np.ones(self.NX)
    
        if ~hasattr(zeta, "__len__"):
            zeta = zeta*np.ones(self.NX)
        
        if ~hasattr(U0, "__len__"):
            U0   = U0*np.ones(self.NX)
        
        if ~hasattr(V0, "__len__"):
            V0   = V0*np.ones(self.NX)

        t   =  self.t_axis()
        dt  =  1/self.fs
        X   =  MRPy(np.empty((self.NX, self.N)), self.fs)

        for kX, row in enumerate(self):

            zt  =  zeta[kX]
            wn  =  2*np.pi*fn[kX]   
            wd  =  wn*np.sqrt(1 - zt**2)
        
            et  =  np.exp(zt*wn*t)
            st  =  np.sin(wd*t)
            ct  =  np.cos(wd*t)
    
            X[kX,:] = (U0[kX]*ct + (V0[kX] + U0[kX]*zt*wn)*st/wd)/et
    
            A = dt*np.cumsum(row*et*ct)
            B = dt*np.cumsum(row*et*st)
    
            X[kX,:] += (A*st - B*ct)/et/wd
    
        return X

#-----------------------------------------------------------------------------
    
    def sdof_Fourier(self, fn, zeta):
        """
        Integrates the dynamic equilibrium differential equation by Fourier.
        The input MRPy is assumed to be an acceleration (force over mass),
        otherwise the result must be divided by system mass to have
        displacement unit.
        System properties (frequency and damping) may be provided as 
        scalars or lists. If they are scalars, same properties are used
        for all series in the MRP.
    
        Parameters:  fn:   sdof natural frequency (Hz)
                     zeta: sdof damping  (nondim)
        """
        
        if ~hasattr(fn, "__len__"):
            fn   = fn*np.ones(self.NX)
    
        if ~hasattr(zeta, "__len__"):
            zeta = zeta*np.ones(self.NX)

        X   =  MRPy(np.empty((self.NX, self.N)), self.fs)

        for kX, row in enumerate(self):

            zt  =  zeta[kX]
            wn  =  2*np.pi*fn[kX]        
            K   =  wn*wn
            
            b   =  2*np.pi*self.f_axis()/wn
            Hw  = (K*((1.0 - b**2) + 1j*(2*zt*b)))**(-1)
            Hw  =  np.hstack((Hw,np.conj(Hw[-2:0:-1])))

            X[kX,:]  =  np.real(np.fft.ifft(Hw*np.fft.fft(row)))
        
        return X

#-----------------------------------------------------------------------------

    def random_decrement(self, div=4, thr=1.0, ref=0):
        """
        MATHEUS CARINI HAS FOUND A PROBLEM HERE! MUST BE REVISED!!!
        
        Estimate the free decay response of a dynamic system from the
        response to a wide band excitation by the random decrement (RD) 
        method. 
        Parameters:  div: number of divisions of total length, N//n,
                          to define the length of decrement series.
                          The divided length will be eventually truncated
                          to be even.
                     thr: threshold level that defines the reference
                          upcrossing level, given as a multiple of the 
                          standard deviation of the reference MRP.
                     ref: row of MRPy to be used as reference series.
                          The other series will be splitted at the same
                          crossing points, what implies phase consistency.
        """

        n    =  self.N//div                 # convert to length
        n    =  n  - (np.mod(n,2) == 1)     # force length to be even       
        Xm   =  self.mean(axis=1)           # mean values are zero

        Xref =  self[ref,:]                 # reference series
        X0   =  thr*(Xref.std())            # crossing reference level

        kref = ( ((Xref[0:(self.N-1)] < X0) & (Xref[1:self.N] >= X0)) |
                 ((Xref[0:(self.N-1)] > X0) & (Xref[1:self.N] <= X0)) )

        nk   =  sum(kref)        
        Y    =  MRPy(np.zeros((self.NX, n)), self.fs)

        for kX, row in enumerate(self):
            row -=  Xm[kX]                  # remove mean value
            
            for k in range(self.N - n):
                if kref[k]: 
                    Y[kX,:] += row[k:(k+n)]

        return Y/nk

#-----------------------------------------------------------------------------
    
    def fit_decay(self):
        """
        Fit the theoretical free decay function of a sdof dynamic system 
        to the provided MRP. The MRPy mean value is discarded. The fitted
        parameters are output as a tuple P = (Xp, fn, zt, ph), where
        Xp is the amplitude, fn is the fundamental (undamped) frequency,
        zt is the damping as the ratio of critical, and ph is the phase
        angle with respect with the cosinus function. This method is 
        typically used to fit the output of the random decrement method.
        """
        
       #-------------------------------------------------------
        def decay(t, Xp, fn, zt, ph):
            
            wn =  2*np.pi*fn
            wd =  wn*np.sqrt(1. - zt*zt)
            
            return Xp*np.exp(-zt*wn*t)*np.cos(wd*t - ph)
       #-------------------------------------------------------

        t  =  self.t_axis()
        f  =  self.f_axis()
        
        P  =  np.zeros((self.NX, 4))
        X  =  self.zero_mean()
        
        Sx, fs = X.periodogram()
        
        for kX, row in enumerate(X):

            Xp =  np.max(row)             # initial amplitude value  
            fn =  f[np.argmax(Sx[kX,:])]  # initial natural frequency
            zt =  0.03                    # initial damping
            ph =  0.00                    # initial phase
    
            Pmin = (0.5*Xp,  1/t[-1], 0.0, -np.pi)     # lower bounds
            P0   = (    Xp,  fn,      zt,   ph   )     # initial guesses
            Pmax = (1.5*Xp,  1*f[-1], 0.5,  np.pi)     # upper bounds
            
            try:                
                P[kX,:], cv = curve_fit(decay, t, row, 
                                        p0=P0, bounds=(Pmin, Pmax))
            except:
                P[kX,:] = np.zeros(5)
                print('Not able to fit decay function!!!')
                pass
            
            X[kX,:] = decay(t, *P[kX,:])
    
        return MRPy(X, fs), P

#=============================================================================
# 4. Class constructors from conceptual properties
#=============================================================================

    def zeros(NX=1, N=1024, fs=None, Td=None):
        """
        Add up all series in MRPy weighted by 'weight'.
        
        Parameters: NX: number of processes in the MRPy object.
                    N:  length of each process.
                    fs: sampling frequency (in Hz), or alternatively
                    Td: processes duration (second)
        """

        fs, Td = MRPy.check_fs(N, fs, Td)
        return MRPy(np.zeros((NX,N)), fs)

#-----------------------------------------------------------------------------

    def Dirac(NX=1, N=1024, t0=0.0, fs=None, Td=None):
        """
        Add up all series in MRPy weighted by 'weight'.
        
        Parameters: NX: number of processes in the MRPy object.
                    N:  length of each process.
                    t0: time at which impulse must be given
                    fs: sampling frequency (in Hz), or alternatively
                    Td: processes duration (second)
        """

        fs, Td  =  MRPy.check_fs(N, fs, Td)
        i0      =  int(t0//fs)
        X       =  np.zeros((NX,N))
        X[:,i0] =  1.0

        return MRPy(X, fs)

#-----------------------------------------------------------------------------

    def Heaviside(NX=1, N=1024, t0=0.0, fs=None, Td=None):
        """
        Add up all series in MRPy weighted by 'weight'.
        
        Parameters: NX: number of processes in the MRPy object.
                    N:  length of each process.
                    t0: time at which step must be given
                    fs: sampling frequency (in Hz), or alternatively
                    Td: processes duration (second)
        """

        fs, Td   = MRPy.check_fs(N, fs, Td)
        i0       = int(t0*fs)
        X        = np.zeros((NX,N))
        X[:,i0:] = 1.0

        return MRPy(X, fs)

#-----------------------------------------------------------------------------

    def harmonic(NX=1, N=1024, fs=None, Td=None, X0=1.0, f0=1.0, phi=0.0):
        """
        Creates an instance with harmonic functions with unity amplitude.
        
        Parameters: NX:   number of processes in the MRPy object.
                    N:    length of each process.
                    fs:   sampling frequency (in Hz), or alternatively
                    Td:   processes duration (second)
                    X0:   process amplitude
                    f0:   signal frequency (in Hz)
                    phi:  signal phase (rad).
        """

        fs, Td = MRPy.check_fs(N, fs, Td)        
        X      = np.zeros((NX,N))
        
        if ~hasattr(X0,  "__len__"):
            X0  = X0*np.ones(NX)
        
        if ~hasattr(f0,  "__len__"):
            f0  = f0*np.ones(NX)
    
        if ~hasattr(phi, "__len__"):
            phi = phi*np.ones(NX)

        t  = np.linspace(0, Td, N)

        for kX in range(NX):
            X[kX,:] = X0[kX]*np.sin(2*np.pi*f0[kX]*t + phi[kX])

        return MRPy(X, fs)

#-----------------------------------------------------------------------------

    def white_noise(NX=1, N=1024, fs=None, Td=None, band=None):
        """
        Simulates a band-limited Gaussian white noise'.
        
        Parameters: NX:   number of processes in the MRPy object.
                    N:    length of each process.
                    fs:   sampling frequency (in Hz), or alternatively
                    Td:   processes duration (second)
                    band: band with nonzero power (in Hz)
        """

        fs, Td  = MRPy.check_fs(N, fs, Td)
        b0, b1  = MRPy.check_band(fs, band)
        
        M       = N//2 + 1
        Sx      = np.ones((NX, M))

        k0 = np.int(2*M*b0/fs)
        k1 = np.int(2*M*b1/fs)
            
        Sx[:,:k0] = 0.
        Sx[:,k1:] = 0.
        
        for kX in range(NX):
            Sx[kX] = Sx[kX]/np.trapz(Sx[kX], dx=1./Td)

        return MRPy.from_periodogram(Sx, fs)

#-----------------------------------------------------------------------------

    def pink_noise(NX=1, N=1024, fs=None, band=None):
        """
        Add up all series in MRPy weighted by 'weight'.
        
        Parameters: NX:    number of processes in the MRPy object.
                    N:     length of each process.
                    fs:    sampling frequency (in Hz), or alternatively
                    Td:    processes duration (second)
                    band: band with nonzero power (in Hz)
        """

        fs, Td  = MRPy.check_fs(N, fs, Td)
        b0, b1  = MRPy.check_band(fs, band)
        
        M       = N//2 + 1
        Sx      = np.ones((NX, M))

        k0 = np.int(2*M*b0/fs)
        k1 = np.int(2*M*b1/fs)
            
        Sx[:,:k0] = 0.
        Sx[:,k1:] = 0.
        
        for kX in range(NX):
            Sx[kX] = Sx[kX]/np.trapz(Sx[kX], dx=1./Td)
 
        return MRPy.from_periodogram(Sx, fs)

#=============================================================================
# 5. MRPy properties (as non-MRPy outputs)
#=============================================================================

    def periodogram(self):
        """
        Estimates the one-side power spectrum of a MRP.
        """

        Sx  =  np.empty((self.NX, self.M))

        for kX in range(self.NX):
            
            Fx  =  np.fft.fft(self[kX,:] - self[kX,:].mean())
            Sxk =  np.real(Fx*Fx.conj())*2/self.N/self.fs
            
            Sx[kX,:] =  Sxk[:self.M]
        
        return Sx, self.fs

#-----------------------------------------------------------------------------
    
    def autocov(self):
        """
        Estimates the autocovariance functions of a MRP.
        """

        Tmax = (self.M - 1)/self.fs
        Cx   =  np.empty((self.NX, self.M))

        Sx, fs  =  self.periodogram()

        for kX in range(self.NX):
            
            Sxk      =  np.hstack((Sx[kX,:], Sx[kX,-2:0:-1]))
            Cxk      =  np.fft.ifft(Sxk)*fs/2
            Cx[kX,:] =  np.real(Cxk[:self.M])

        return Cx, Tmax

#-----------------------------------------------------------------------------
    
    def autocorr(self):
        """
        Estimates the autocorrelation function of a MRP.
        """
        
        Xs  =  self.std(axis=1)
        Rx  =  np.empty((self.NX,self.M))
        
        Cx, Tmax =  self.autocov()

        for kX in range(self.NX):
            
            Rx[kX,:] =  Cx[kX,:]/Xs[kX]/Xs[kX]

        return Rx, Tmax

#-----------------------------------------------------------------------------

    def pseudo(self, zeta=0.05):
        """
        Estimates the pseudo spectra, which are the peak response 
        amplitudes of a single degree of freedom system, as a function of 
        system natural period of vibration. The usual application is for
        seismic acceleration records.
        
        Parameters:  zeta: system damping (ratio of critical) can be 
                           provided or default value of 5% is assumed.
        """

        Tmax = (self.M - 1)/self.fs
        Sx   =  np.zeros((self.NX, self.M))
        T    =  self.T_axis()

        for k in range(self.M):       # this may take long...
            
            if (T[k] > 8/self.fs):
                
                    X    = self.sdof_Duhamel(1/T[k], zeta)
                    umax = np.abs(X).max(axis=1)
                    
                    if ~np.any(np.isnan(umax)): 
                        Sx[:,k] = umax
        
        return Sx, Tmax

#-----------------------------------------------------------------------------
    
    def Davenport(self, T=-1.):
        """
        Peak factor of a MRPy by Davenport's formula.
        
        Parameters:  T: observation time for estimating peak factor.
                        The default value is -1, that means the total
                        duration of MRP, Td, is to be used.
        """

        if (T < 0.): T = self.Td
        
        e      = 0.5772156649
        f      = self.f_axis()
        df     = 1/self.Td

        Sx, fs = self.periodogram()
        gX     = np.zeros(self.NX)

        for kX in range(self.NX):
        
            m0 = np.trapz(Sx[kX,:],     dx=df)
            m2 = np.trapz(Sx[kX,:]*f*f, dx=df)
            
            nu = T*np.sqrt(m2/m0)
            if (nu < 1): nu = 1
        
            Lg = np.sqrt(2*np.log(nu))
            if (Lg < np.sqrt(e)): Lg = np.sqrt(e)
            
            gX[kX] = Lg + e/Lg
        
        return gX

#-----------------------------------------------------------------------------
    
    def splitmax(self, T=-1.):
        """
        Peak factor of a MRPy by the "splitmax" method.
        
        Parameters:  T: observation time for estimating peak factor.
                        The default value is -1, that means the total
                        duration of MRP, Td, is to be used.
        """   
        
       #-----------------------------------------------
        def split(X):
        
            X1 = X[0::2]
            X2 = X[1::2]
        
            if not len(X1): 
                return np.array([])
        
            if len(X1) > len(X2):
                X1 = X1[:-1]
        
            return np.max(np.vstack((X1, X2)), axis=0)
        
       #-----------------------------------------------

        if (T < 0.):      T = self.Td
        if (T > self.Td): T = self.Td
        
        gX   = np.zeros(self.NX)

        for kX, row in enumerate(self):
    
            Y    =  split(np.abs(row))
            nmax =  np.array([])
            Xmax =  np.array([])
        
            while np.size(Y):
                nmax = np.append(nmax,self.N/len(Y))
                Xmax = np.append(Xmax,Y.mean())
                Y    = split(Y)
            
            f = interp1d(np.log(nmax), Xmax, kind='quadratic')
            
            gX[kX] = float(f(np.log(T*self.fs))/self.std())
        
        return gX

#=============================================================================
# 6. Utilities
#=============================================================================

    def printAttrib(self):
        
        s1 =  ' fs = {0:.1f}Hz\n Td = {1:.1f}s\n'
        s2 =  ' NX = {0}\n N  = {1}\n M  = {2}\n'   
        
        print(s1.format(self.fs, self.Td))
        print(s2.format(self.NX, self.N, self.M))

#-----------------------------------------------------------------------------

    def t_axis(self):        
        return np.linspace(0, self.Td, self.N)

#-----------------------------------------------------------------------------
    
    def f_axis(self):        
        return np.linspace(0, self.fs/2, self.M)
    
#-----------------------------------------------------------------------------
    
    def T_axis(self):        
        return np.linspace(0, (self.M - 1)/self.fs, self.M)
    
#-----------------------------------------------------------------------------
    
    def subplot_shape(self):        

        sp0 = self.NX
        sp1 = 1

        if   (sp0 > 12):
            sp0 = 4
            sp1 = 5
        elif (sp0 == 8):
            sp0 = 4
            sp1 = 2
        elif (sp0 > 6):
            sp0 = 4
            sp1 = 3
        elif (sp0 > 3): 
            sp0 = 3
            sp1 = 2
        
        return sp0, sp1

#-----------------------------------------------------------------------------
    
    def plot_time(self, fig=0, figsize=(12, 8), axis_t=None):

        plt.figure(fig, figsize=figsize)
        plt.suptitle('Time Domain Amplitude', fontsize=14)
        
        t  = self.t_axis()
        
        if (axis_t == None): 
            axis_t = [0, self.Td, 1.2*self.min(), 1.2*self.max()]

        sp0, sp1 = self.subplot_shape()
        lines    = []

        for kX, row in enumerate(self):
            
            plt.subplot(sp0,sp1,kX+1)
            lines.append(plt.plot(t, row, lw=0.5))
        
            plt.axis(axis_t)
            plt.ylabel('Amplitude {0}'.format(kX))
            plt.grid(True)
            
        plt.xlabel('Time (s)')
            
        return lines

#-----------------------------------------------------------------------------
    
    def plot_freq(self, fig=0, figsize=(12, 8), axis_f=None):
        
        plt.figure(fig, figsize=figsize)
        plt.suptitle('Spectral Density Estimator', fontsize=14)
        
        Sx, fs = self.periodogram()
        
        f  = self.f_axis()

        if (axis_f == None): 
            axis_f = [0, self.fs/2, 0, 1.2*Sx.max()]

        sp0, sp1 = self.subplot_shape()
        lines    = []

        for kX, row in enumerate(Sx):
            
            plt.subplot(sp0,sp1,kX+1)
            lines.append(plt.plot(f, row, lw=0.5))
        
            plt.axis(axis_f)
            plt.ylabel('Power {0}'.format(kX))
            plt.grid(True)

        plt.xlabel('Frequency (Hz)')

        return lines

#-----------------------------------------------------------------------------
    
    def plot_corr(self, fig=0, figsize=(12, 8), axis_T=None):

        plt.figure(fig, figsize=figsize)
        plt.suptitle('Normalized Autocorrelation', fontsize=14)
        
        Rx, Tmax = self.autocorr()
        
        T  = self.T_axis()

        if (axis_T == None): 
            axis_T = [0, Tmax, -1.2, 1.2]

        sp0, sp1 = self.subplot_shape()
        lines    = []

        for kX, row in enumerate(Rx):
            
            plt.subplot(sp0,sp1,kX+1)
            lines.append(plt.plot(T, row, lw=0.5))
        
            plt.axis(axis_T)
            plt.ylabel('Autocorrelation {0}'.format(kX))
            plt.grid(True)

        plt.xlabel('Time gap (s)')

        return lines

#-----------------------------------------------------------------------------
    
    def plot_pseudo(self, fig=0, figsize=(12, 8), axis_T=None):
        
        plt.figure(fig, figsize=figsize)
        plt.suptitle('Displacement Response Spectrum', fontsize=14)
        
        Sx, Tmax = self.pseudo()
        
        T  = self.T_axis()

        if (axis_T == None): 
            axis_T = [0, Tmax, 0, 1.2*Sx.max()]

        sp0, sp1 = self.subplot_shape()
        lines    = []

        for kX, row in enumerate(Sx):
            
            plt.subplot(sp0,sp1,kX+1)
            lines.append(plt.plot(T, row, lw=0.5))
        
            plt.axis(axis_T)
            plt.ylabel('Peak response {0}'.format(kX))
            plt.grid(True)

        plt.xlabel('Vibration period (s)')

        return lines

#-----------------------------------------------------------------------------
    
    def to_file(self, filename, form='mrpy'):
        """
        Save MRPy object.
 
        Parameters:  filename: file to be saved, including path
                     form:     data formatting. Options are
                               'mrpy'   - default gzip pickle saving
                               'excel ' - excel through pandas 
        """

        if (form.lower() == 'mrpy'):
            with gz.GzipFile(filename+'.csv.gz', 'wb') as target:
                pk.dump((self, self.fs), target)

        elif (form.lower() == 'excel'):

            data  = pd.DataFrame(data    = self.T,
                                 index   = self.t_axis(),
                                 columns = np.arange(self.NX))

            excel = pd.ExcelWriter(filename+'.xlsx')
            data.to_excel(excel,'MRPy')
            excel.save()
            
        else:
            sys.exit('Data formatting not available!')
            
        return None

#=============================================================================
# 7. Helpers
#=============================================================================
 
    def resampling(ti, Xi):
        """
        Resampling irregular time step to fixed time step. The last
        element of ti is taken as total series duration. Series length
        is kept unchanged. Returns a MRPy instance.
 
        Parameters:  ti:    irregular time where samples are avaible
                     Xi:    time series samples, taken at ti
        """
        
        sh =  Xi.shape
        if (len(sh) == 1): 
            Xi  = np.reshape(Xi,(1,sh[0]))
        elif (sh[0] > sh[1]):
            Xi = Xi.T
            
        sh =  Xi.shape
        NX =  sh[0]
        N  =  sh[1]

        if (N < 2):
            sys.exit('Come on!!! Start with at least 2 elements!')
        
        tsh =  ti.shape
        if (len(tsh) > 1): 
            sys.exit('Time markers must be a 1D vector!')

        t0 =  ti[0]
        t1 =  ti[-1]

        fs =  N/(t1 - t0)               # average sampling rate
        t  =  np.linspace(t0, t1, N)    # regularly spaced time markers
        X  =  np.empty((NX,N))          # interpolated series
        
        for k in range(NX):
            resX   =  interp1d(ti, Xi[k,:], kind='linear')
            X[k,:] =  resX(t)

        return MRPy(X, fs)

#-----------------------------------------------------------------------------

    def check_fs(N, fs, Td):
        """
        Verifies if either fs or Td are given, and returns both
        properties verifyed. Observe that N is not verified to be
        even, for this will be done later on one MRPy constructor 
        is called. This means that Td may be eventually modified.
        """

        if ((fs is not None) & (Td is None)):    # if fs is available...
            Td = N/fs

        elif ((fs is None) & (Td is not None)):  # if Td is available
            fs = N/Td

        else: 
            sys.exit('Either fs or Td must be specified!')
        
        return fs, Td

#-----------------------------------------------------------------------------

    def check_band(fs, band):
        """
        Verifies if provided frequency band is consistent.
        """
        
        if (band is None):
            b0 = 0.
            b1 = fs/2

        else:
            b0 = band[0]
            b1 = band[1]

        if (b0 < 0):
            warn('Lower band limit truncated to 0!')
            b0 = 0

        if (b1 > fs/2):
            warn('Upper band limit truncated to fs/2!')
            b1 = fs/2
        
        return b0, b1
    
#-----------------------------------------------------------------------------

    def Cx2Sx(Cx, Tmax):
        """
        Returns the spectral density corresponding to a given
        autocovariance function.
 
        Parameters:  Cx:   autocovariances as ndarray (must have odd
                           length, otherwise it will be truncated by 1 and
                           the length of simulation will not be as expected!)
                           The largest dimension of Cx is assumed to be the
                           time gap axis.
                     Tmax: largest time gap, associated to the last element
                           in array Cx. Defines process duration, which 
                           will be approximately 2Tmax.
        """

        sh  =  Cx.shape

        if (len(sh) == 1): 
            Cx = np.reshape(Cx,(1,sh[0]))
        else:
            if (sh[0] > sh[1]):
                Cx = Cx.T

        sh  =  Cx.shape
        NX  =  sh[0]
        M0  =  sh[1]
        M   =  M0 - (np.mod(M0,2) == 0)     # ensure M is odd

        Cx  =  Cx[:, 0:M]
        
        err =  M/M0             
        fs  = (M - 1)/(err*Tmax)            # eventually corrects for Tmax
        
        Sx  =  np.empty((NX, M))

        for k in range(NX): 
            
            C       =  np.hstack((Cx[k,:], Cx[k,-2:0:-1]))
            C       =  np.fft.fft(C)*2/fs
            Sx[k,:] =  np.real(C[0:M])

        return Sx, fs

#-----------------------------------------------------------------------------

    def Sx2Cx(Sx, fs):
        """
        Returns the autocovariance corresponding to a given
        spectral density.
 
        Parameters:  Sx:  spectral density as ndarray (must have odd
                          length, otherwise it will be truncated by 1 and
                          the length of simulation will not be as expected!)
                          The largest dimension of Sx is assumed to be the
                          frequency axis.
                     fs:  sampling frequency in Hz
        """
        
        sh  =  Sx.shape

        if (len(sh) == 1): 
            Sx = np.reshape(Sx,(1,sh[0]))
        else:
            if (sh[0] > sh[1]):
                Sx = Sx.T

        sh  =  Sx.shape
        NX  =  sh[0]
        M0  =  sh[1]
        M   =  M0 - (np.mod(M0,2) == 0)     # ensure M is odd

        Tmax = (M - 1)/fs
        Cx   =  np.empty((NX, M))

        for kX in range(NX):
            
            Sxk      =  np.hstack((Sx[kX,:], Sx[kX,-2:0:-1]))
            Cxk      =  np.fft.ifft(Sxk)*fs/2
            Cx[kX,:] =  np.real(Cxk[:M])

        return Cx, Tmax

#=============================================================================
#=============================================================================
