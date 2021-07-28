import numpy
import numpy.fft
import numpy.linalg
import copy
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from scipy.signal import convolve
import offset_index

# some basic definitions
psSize = 9 # psSize x psSize postage stamps of stars

# zero padded RectBivariateSpline, if on
def RectBivariateSplineZero(y1,x1,map1,kx=1,ky=1):
  return RectBivariateSpline(y1, x1, map1, kx=kx, ky=ky)
  y2 = numpy.zeros(numpy.size(y1)+2)
  y2[1:-1] = y1
  y2[0] = 2*y2[1]-y2[2]
  y2[-1] = 2*y2[-2]-y2[-3]
  x2 = numpy.zeros(numpy.size(x1)+2)
  x2[1:-1] = x1
  x2[0] = 2*x2[1]-x2[2]
  x2[-1] = 2*x2[-2]-x2[-3]
  map2 = numpy.zeros((numpy.size(y1)+2, numpy.size(x1)+2))
  map2[1:-1,1:-1] = map1
  return RectBivariateSpline(y2, x2, map2, kx=kx, ky=ky)

class EmptyClass():
  pass

# spectral energy distribution class
class SpectralEnergyDistribution():

  # make an SED -- several options for type
  def __init__(self, type, info):
    self.type = type
    self.info = copy.deepcopy(info)

  # get Nlambda (photons/m^2/s/um) at lambda_ (um)
  def Nlambda(self, lambda_):

    # blackbody, info = [T (K), solidangle]
    if self.type=='BB':
      T = self.info[0]
      x = 14387.769/lambda_/T # hc/(kTlambda)
      return(2/lambda_**4*2.99792458e14*1e12*numpy.exp(-x)/(1.-numpy.exp(-x))*self.info[1])
      # the 1e12 is the conversion from um^2 -> m^2
    else:
      print('ERROR: Invalid SED type')
      exit()

# filter class
class Filter():

  # make a filter -- several options for type
  def __init__(self, type, info):
    self.type = type
    self.info = copy.deepcopy(info)

  # get transmission
  def Tlambda(self, lambda_):

    # smoothed tophat
    if self.type=='STH':
      lmin = self.info[0]; dlmin = lmin*.02
      lmax = self.info[1]; dlmax = lmax*.02
      return((numpy.tanh((lambda_-lmin)/dlmin)-numpy.tanh((lambda_-lmax)/dlmax))/2.)
    # interpolated file
    # info shape (N,2) -- info[:,0] = wavelength, info[:,1] = throughput
    elif self.type=='interp':
      return(numpy.interp(lambda_, self.info[:,0], self.info[:,1]))
    else:
      print('ERROR: Invalid filter type')
      exit()

# load mask files
maskfiles = EmptyClass()
maskfiles.D = 2292981.05344 # um
maskfiles.rim = []
maskfiles.full = []
maskfiles.i_rim = []
maskfiles.i_full = []
maskfiles.nSCA = 18
for k in range(18):
  inFile = fits.open('pupils/SCA{:d}_rim_mask.fits'.format(k+1))
  maskfiles.rim += [numpy.copy(inFile[0].data[::-1,:])]
  inFile.close()
  inFile = fits.open('pupils/SCA{:d}_full_mask.fits'.format(k+1))
  maskfiles.full += [numpy.copy(inFile[0].data[::-1,:])]
  inFile.close()

  # normalize
  maskfiles.rim[k] /= numpy.amax(maskfiles.rim[k])
  maskfiles.full[k] /= numpy.amax(maskfiles.full[k])

  N_in = maskfiles.N_in = 2048
  x_in = numpy.linspace(-1+1/N_in,1-1/N_in,N_in)
  y_in = numpy.copy(x_in)
  interp_spline = RectBivariateSplineZero(y_in, x_in, maskfiles.rim[k], kx=1, ky=1)
  maskfiles.i_rim += [interp_spline]
  interp_spline = RectBivariateSplineZero(y_in, x_in, maskfiles.full[k], kx=1, ky=1)
  maskfiles.i_full += [interp_spline]

  # lower resolution masks
  maskfiles.n_lores = 7
  for ku in range(1,maskfiles.n_lores):
    N2 = N_in//2**ku
    x_in = numpy.linspace(-1+1/N2,1-1/N2,N2)
    y_in = numpy.copy(x_in)
    interp_spline = RectBivariateSplineZero(y_in, x_in, numpy.mean(maskfiles.rim[k].reshape(N2,2**ku,N2,2**ku), axis=(1,3)), kx=1, ky=1)
    maskfiles.i_rim += [interp_spline]
    interp_spline = RectBivariateSplineZero(y_in, x_in, numpy.mean(maskfiles.full[k].reshape(N2,2**ku,N2,2**ku), axis=(1,3)), kx=1, ky=1)
    maskfiles.i_full += [interp_spline]

# SCA locations
sca = EmptyClass()
sca.size = 40.88 # mm
sca.x = numpy.asarray([-22.14, -22.29, -22.44, -66.42, -66.92, -67.42, -110.70, -111.48, -112.64,
  22.14, 22.29, 22.44, 66.42, 66.92, 67.42, 110.70, 111.48, 112.64])
sca.y = numpy.asarray([12.15, -37.03, -82.06, 20.90, -28.28, -73.06, 42.20, 13.46, -51.06,
  12.15, -37.03, -82.06, 20.90, -28.28, -73.06, 42.20, 13.46, -51.06])
sca.scale = 133.08

# reference Zernikes
ZernRef = EmptyClass()
ZernRef.data = numpy.loadtxt('pupils/zernike_ref.txt')[:,-22:] * 1.38

# filter data
FilterData = numpy.loadtxt('pupils/filter.dat')
FilterData[:,1:] /= numpy.pi/4.*(maskfiles.D/1e6)**2

# makes map of Zernikes of a given amplitude
# amp[0:Namp] = Z1 ... ZNamp
# on a spacing Ngrid (x, y = -(1-1/Ngrid) .. +(1-1/Ngrid) multiplied by scale)
#
def zernike_map_noll(amp, Ngrid, scale):
  xx = numpy.tile(numpy.linspace(-1+1/Ngrid,1-1/Ngrid,Ngrid), (Ngrid,1))
  yy = numpy.copy(xx.T)
  rho = numpy.sqrt(xx**2+yy**2)*scale
  phi = numpy.arctan2(yy,xx)
  output = numpy.zeros((Ngrid,Ngrid))
  nmax = 0
  namp = numpy.size(amp)
  while namp>(nmax+1)*(nmax+2)//2: nmax+=1
  rpows = numpy.ones((nmax+1,Ngrid,Ngrid))
  trigphi = numpy.ones((2*nmax+1,Ngrid,Ngrid))
  for i in range(1,nmax+1): rpows[i,:,:] = rho**i
  for i in range(0,nmax+1): trigphi[i,:,:] = numpy.cos(i*phi)
  for i in range(1,nmax+1): trigphi[-i,:,:] = numpy.sin(i*phi)
  # loop over Zernikes
  for n in range(nmax+1):
    for m in range(-n,n+1,2):
      Z = numpy.zeros((Ngrid,Ngrid))
      for k in range((n-abs(m))//2+1):
        coef = (-1)**k * numpy.math.factorial(n-k)/numpy.math.factorial(k) \
             /numpy.math.factorial((n-m)//2-k)/numpy.math.factorial((n+m)//2-k) 
        Z += coef * rpows[n-2*k,:,:]
      #if m>=0:
      #  Z *= numpy.cos(m*phi)
      #else:
      #  Z *= numpy.sin(-m*phi)
      Z *= trigphi[m,:,:]
      j = n*(n+1)//2 + abs(m)
      if (-1)**j*(m+.5)<0 or m==0: j += 1
      #print(n,m,j)
      factor = numpy.sqrt(n+1)
      if m!=0: factor *= numpy.sqrt(2)
      if j<=namp: output += factor * amp[j-1] * Z
  return(output)

# make annular mask of given obstruction (fraction) and scale
def make_mask_annulus(obs, Nstep, scale):
  xx = numpy.tile(numpy.linspace(-1+1/Nstep,1-1/Nstep,Nstep), (Nstep,1))
  yy = numpy.copy(xx.T)
  rho = numpy.sqrt(xx**2+yy**2)*scale
  return(numpy.where(numpy.logical_and(rho>=obs,rho<1),numpy.ones((Nstep,Nstep)),numpy.zeros((Nstep,Nstep))))

def test_zernike():
  for k in range(36):
    psi = numpy.zeros(36)
    psi[k] = 1
    N=5
    M = zernike_map_noll(psi, N, N/(N-1))
    print(' *** Zernike {:2d} ***'.format(k+1))
    for j in range(N):
      out = ''
      for i in range(N):
        out = out + ' {:10.5f}'.format(M[j,i])
      print(out)
    print('')

# psi is a vector of Zernikes, in wavelengths
# mask information: (currently none)
# scale = sampling (points per lambda/D)
# Nstep = # grid points
# output normalized to sum to 1
def mono_psf(psi, mask, scale, Nstep):
  if hasattr(mask, 'N'):
    if hasattr(mask, 'spline'):
      interp_spline = mask.spline
    else:
      N_in = 2048
      x_in = numpy.linspace(-1+1/N_in,1-1/N_in,N_in)
      y_in = numpy.copy(x_in)
      interp_spline = RectBivariateSplineZero(y_in, x_in, mask.array, kx=1, ky=1)
    x2 = numpy.linspace(-1+1/Nstep,1-1/Nstep,Nstep)*scale
    y2 = numpy.copy(x2)
    amplitude = interp_spline(y2,x2).astype(numpy.complex128) * make_mask_annulus(0, Nstep, scale)
  else:
    amplitude = make_mask_annulus(.32, Nstep, scale).astype(numpy.complex128)
  amplitude *= numpy.exp(2j * numpy.pi * zernike_map_noll(psi, Nstep, scale))
  amplitude = numpy.fft.ifft2(amplitude)
  power = numpy.abs(amplitude)**2
  # shift to center
  newpower = numpy.zeros_like(power)
  newpower[Nstep//2:Nstep,Nstep//2:Nstep] = power[0:Nstep//2,0:Nstep//2]
  newpower[Nstep//2:Nstep,0:Nstep//2] = power[0:Nstep//2,Nstep//2:Nstep]
  newpower[0:Nstep//2,Nstep//2:Nstep] = power[Nstep//2:Nstep,0:Nstep//2]
  newpower[0:Nstep//2,0:Nstep//2] = power[Nstep//2:Nstep,Nstep//2:Nstep]
  return(newpower/numpy.sum(newpower))

# helper function
def onescut(n):
  array = numpy.ones((n+1))
  array[0] = array[-1] = .5
  return(array/n)

# Gaussian quadrature weights across a filter
# sed = spectral energy distribution
# filter = filter information (incl. bandpass)
# nOrder = order of polynomial (number of nodes)
# wlrange = [lmin,lmax,npts] in um
#
# returns wavelengths, weights
def gq_weights(sed, filter, nOrder, wlrange):
  # unpack info
  lmin = wlrange[0]; lmax = wlrange[1]; npts = wlrange[2]

  # build integrals I_k = int x^k S(x) F(x) dx
  x = numpy.linspace(lmin,lmax,npts)
  c = numpy.zeros((npts))
  for i in range(npts):
    c[i] = sed.Nlambda(x[i]) * filter.Tlambda(x[i])
  o = numpy.ones((npts))
  I = numpy.zeros((2*nOrder))
  lctr = numpy.mean(x)
  for k in range(2*nOrder):
    I[k] = numpy.sum(o*(x-lctr)**k*c)
  # orthogonal polynomial p_n
  # require sum_{j=0}^n coef_{n-j} I_{j+k} = 0    or
  # sum_{j=0}^{n-1} coef_{n-j} I_{j+k} = -I_{n+k}    for k = 0 .. n-1
  coef = numpy.zeros((nOrder+1))
  coef[0] = 1.
  A = numpy.zeros((nOrder,nOrder))
  for k in range(nOrder):
    for j in range(nOrder):
      A[k,j] = I[j+k]
  coef[1:] = numpy.linalg.solve(A, -I[nOrder:])[::-1]
  p = numpy.poly1d(coef)
  xroot = numpy.sort(numpy.real(p.r))
  wroot = numpy.zeros_like(xroot)
  pprime = numpy.polyder(p)
  for i in range(nOrder):
    px = numpy.poly1d(numpy.concatenate((xroot[:i], xroot[i+1:])), r=True)
    wroot[i] = numpy.sum(px.c[::-1]*I[:nOrder]) / pprime(xroot[i])
  xroot = xroot + lctr
  return xroot,wroot

# psi is a vector of Zernikes, in microns
# mask information: (currently none)
# sed = spectral energy distribution
# scale = sampling (points per lambda/D @ 1 um)
# Nstep = # grid points
# filter = filter information (incl. bandpass)
# addInfo = class for general additional information
# output normalized to sum to 1
def poly_psf(psi, mask, sed, scale_1um, Nstep, filter, addInfo):

  # integration steps
  hard_lmin = 0.4
  hard_lmax = 2.5
  hard_Nl = 420

  ilmin = hard_Nl-1; ilmax = 0
  for il in range(1,hard_Nl):
    wl = hard_lmin + il/hard_Nl*(hard_lmax-hard_lmin)
    if filter.Tlambda(wl)>1e-4:
      if il<ilmin:
        ilmin=il
        wlmin=wl
      if il>ilmax:
        ilmax=il
        wlmax=wl
  na = ilmin//6 + 1
  nb = (hard_Nl-ilmax)//6 + 1
  wl = numpy.concatenate((numpy.linspace(hard_lmin,wlmin,na+1), numpy.linspace(wlmin,wlmax,ilmax-ilmin+1), numpy.linspace(wlmax,hard_lmax,nb+1)))
  dwl = numpy.concatenate(((wlmin-hard_lmin)*onescut(na), (wlmax-wlmin)*onescut(ilmax-ilmin), (hard_lmax-wlmax)*onescut(nb)))
  #print(wl,dwl,numpy.size(wl),numpy.size(dwl))

  # reduced coverage
  if hasattr(addInfo,'FastMode'):
    if addInfo.FastMode:
      wl, dwl = gq_weights(sed, filter, 10, [wlmin,wlmax,ilmax-ilmin+1])

  # make output PSF
  sumc = 0.
  output = numpy.zeros((Nstep,Nstep))
  for i in range(numpy.size(wl)):
    c = sed.Nlambda(wl[i]) * filter.Tlambda(wl[i]) * dwl[i]
    if hasattr(addInfo,'FastMode'):
      if addInfo.FastMode: c = dwl[i]
    this_psi = numpy.copy(psi)/wl[i] # convert from um -> wavelengths of wavefront
    sumc += c
    output += c * mono_psf(this_psi, mask, scale_1um*wl[i], Nstep)
    #print('{:6.4f} {:11.5E}'.format(wl[i],filter.Tlambda(wl[i])))
  output /= sumc

  return(output)

# make oversampled PSF at given SCA, position
#
# sed = source SED
# filt = filter (letter: RZYJHFK)
# ovsamp = oversampling factor
# Nstep = number of samples in each axis
# scanum = SCA number (1..18)
# pos = (x,y) position on SCA in mm (0,0)=center
# offsets = adjustment parameters
#   .par  -> offset parameters
# addInfo = additional information class:
#   .ctr  -> centroid (dx,dy) 
def oversamp_psf(sed, filt, ovsamp, Nstep, scanum, pos, offsets, addInfo):

  # get information
  parOn = False
  if hasattr(offsets, 'par'): parOn = True

  # get Zernikes in microns
  ZR = ZernRef.data[4*(scanum-1):4*scanum,:]
  wt_L = .5 - pos[0]/sca.size
  wt_R = .5 + pos[0]/sca.size
  wt_B = .5 - pos[1]/sca.size
  wt_T = .5 + pos[1]/sca.size
  psi = wt_T*wt_L*ZR[0,:] + wt_B*wt_L*ZR[1,:] + wt_B*wt_R*ZR[2,:] + wt_T*wt_R*ZR[3,:]

  xf = sca.x[scanum-1] + pos[0]
  yf = sca.y[scanum-1] + pos[1]

  # Zernike offsets
  if parOn:
    psi[3] += offsets.par[offset_index.foc   ]
    psi[4] += offsets.par[offset_index.astig2]
    psi[5] += offsets.par[offset_index.astig1]

    psi[6] += offsets.par[offset_index.coma2]
    psi[7] += offsets.par[offset_index.coma1]

    psi[3] += (offsets.par[offset_index.focg1]*xf + offsets.par[offset_index.focg2]*yf)/sca.scale

  scale_1um = ovsamp / (.11*numpy.pi/648000) / maskfiles.D
  #print(scale_1um)

  # filter curves
  if filt=='K':
    filter = Filter('STH', [1.95,2.30])
  elif filt=='F':
    filter = Filter('interp', FilterData[:,(0,7)])
  elif filt=='H':
    filter = Filter('interp', FilterData[:,(0,6)])
  elif filt=='W':
    filter = Filter('interp', FilterData[:,(0,5)])
  elif filt=='J':
    filter = Filter('interp', FilterData[:,(0,4)])
  elif filt=='Y':
    filter = Filter('interp', FilterData[:,(0,3)])
  elif filt=='Z':
    filter = Filter('interp', FilterData[:,(0,2)])
  elif filt=='R':
    filter = Filter('interp', FilterData[:,(0,1)])
  else:
    print('Error: unknown filter')
    exit()

  la = numpy.linspace(.4, 2.5, 2101)
  fla = numpy.zeros(2101)
  for i in range(2101): fla[i] = filter.Tlambda(la[i])
  scale = scale_1um*numpy.sum(la*fla)/numpy.sum(fla)

  # get the mask
  mask = EmptyClass(); mask.N=1
  imk = 0
  while imk<maskfiles.n_lores-1 and Nstep/scale<maskfiles.N_in/2**(imk+1): imk+=1
  #print(' *** ', Nstep, scale, scale/scale_1um, imk)
  if filt=='F' or filt=='K':
    mask.spline = maskfiles.i_full[scanum-1 + maskfiles.nSCA*imk]
  else:
    mask.spline = maskfiles.i_rim[scanum-1 + maskfiles.nSCA*imk]

  # x & y offsets
  if hasattr(addInfo, 'ctr'):
    d = .5*(1-1/ovsamp)
    psi[1:3] -= (addInfo.ctr+d) * ovsamp / scale_1um / 4.

  output = poly_psf(psi, mask, sed, scale_1um, Nstep, filter, addInfo)

  # smooth
  Cxx = Cyy = .09; Cxy = 0.
  if parOn:
    Cxx = .09  + offsets.par[offset_index.jxx   ]
    Cxy =        offsets.par[offset_index.jxy   ]
    Cyy = .09  + offsets.par[offset_index.jyy   ]

  output_fft = numpy.fft.fft2(output)
  kx = numpy.zeros((Nstep,Nstep))
  ky = numpy.zeros((Nstep,Nstep))
  for i in range(-Nstep//2, Nstep//2):
    kx[:,i] = abs(i)
    ky[i,:] = abs(i)
  kx *= 2.*numpy.pi*ovsamp/Nstep
  ky *= 2.*numpy.pi*ovsamp/Nstep
  output_fft = output_fft * numpy.exp(-Cxx*kx**2/2. - Cyy*ky**2/2. - Cxy*kx*ky)
  output = numpy.real(numpy.fft.ifft2(output_fft))

  return(output)

# parameters for next couple of functions
N_STD = 1024  # must be a multiple of 4
OV_STD = 8

# make oversampled PSF at given SCA, position
#
# sed = source SED
# filt = filter (letter: RZYJHFK)
# scanum = SCA number (1..18)
# pos = (x,y) position on SCA in mm (0,0)=center
# offsets = adjustment parameters (placeholder)
# addInfo = additional information class:
#   .F    -> total counts (in e)
#   .ctr  -> centroid (dx,dy)
#   .many -> @ 5x5 grid of offsets
#
#   .bfe = add bfe (can include .bfe_a, .bfe_aplus)
#
#   .bfe_overwrite => special mode to compute BFE with time dependent PSF
#   .stamp_in = input stamp (so compute BFE from stamp_in *acting on* this PSF)
def postage_stamp(sed, filt, scanum, pos, offsets, addInfo):
  N = N_STD # must be even
  ov = OV_STD
  if hasattr(addInfo,'many'):
    ov = addInfo.force_ov
  if hasattr(addInfo,'FastMode'):
    if addInfo.FastMode:
      N = N//2
  bigStamp = oversamp_psf(sed, filt, ov, N, scanum, pos, offsets, addInfo) * addInfo.F
  out = numpy.zeros((psSize, psSize))
  for i in range(psSize):
    x = N//2+(i-psSize//2)*ov
    for j in range(psSize):
      y = N//2+(j-psSize//2)*ov
      out[j,i] += numpy.sum(bigStamp[y:y+ov,x:x+ov])
      if hasattr(addInfo, 'vtpe'):
        out[j,i] += addInfo.vtpe * numpy.sum(bigStamp[y+ov:y+2*ov,x:x+ov])
  if hasattr(addInfo,'many'):
    out = numpy.zeros((25, psSize, psSize))
    for i in range(psSize):
      x = N//2+(i-psSize//2)*ov
      for j in range(psSize):
        y = N//2+(j-psSize//2)*ov
        for k in range(25):
          dy = k%5 - 2; dx = k//5 - 2
          out[k,j,i] += numpy.sum(bigStamp[y+dy:y+dy+ov,x+dx:x+dx+ov])

  # BFE?
  if hasattr(addInfo, 'bfe'):
    if hasattr(addInfo,'many'):
      print('Error -- cannot do both bfe and many in postage_stamp')
      exit()
    dout = numpy.zeros_like(out)
    # horizontal BFE
    ah = 0
    if hasattr(addInfo, 'bfe_a'): ah += addInfo.bfe_a
    if hasattr(addInfo, 'bfe_aplus'): ah += addInfo.bfe_aplus
    for i in range(psSize-1):
      x = N//2+(i-psSize//2)*ov
      for j in range(psSize):
        y = N//2+(j-psSize//2)*ov
        shift = ov * ah * (out[j,i+1]-out[j,i]) / 2. # in sub-pixels, average over exposure
        if hasattr(addInfo, 'bfe_overwrite'): shift = ov * ah * (addInfo.stamp_in[j,i+1]-addInfo.stamp_in[j,i]) / 2.
        mflux = numpy.sum(bigStamp[y:y+ov,x+ov-1:x+ov+1])/2.
        dout[j,i] += shift*mflux
        dout[j,i+1] -= shift*mflux
    # vertical BFE
    av = 0
    if hasattr(addInfo, 'bfe_a'): av += addInfo.bfe_a
    if hasattr(addInfo, 'bfe_aplus'): av -= addInfo.bfe_aplus
    for i in range(psSize):
      x = N//2+(i-psSize//2)*ov
      for j in range(psSize-1):
        y = N//2+(j-psSize//2)*ov
        shift = ov * av * (out[j+1,i]-out[j,i]) / 2. # in sub-pixels, average over exposure
        if hasattr(addInfo, 'bfe_overwrite'): shift = ov * av * (addInfo.stamp_in[j+1,i]-addInfo.stamp_in[j,i]) / 2.
        mflux = numpy.sum(bigStamp[y+ov-1:y+ov+1,x:x+ov])/2.
        dout[j,i] += shift*mflux
        dout[j+1,i] -= shift*mflux
    out+=dout

    if hasattr(addInfo, 'bfe_overwrite'): out=dout

  return(out)

#
# same input format but returns moments of the PSF
# A, xc, yc, T, e1, e2
def psfmoments(sed, filt, scanum, pos, offsets, addInfo):
  N = N_STD # must be even
  ov = OV_STD
  if hasattr(addInfo,'many'):
    ov = addInfo.force_ov
  if hasattr(addInfo,'FastMode'):
    if addInfo.FastMode:
      N = N//2
  addInfoX = copy.deepcopy(addInfo); addInfoX.ctr = numpy.zeros((2)); addInfoX.F = 1.
  bigStamp = oversamp_psf(sed, filt, ov, N, scanum, pos, offsets, addInfoX)
  bigStamp = convolve(bigStamp, numpy.ones((ov,ov)), mode='full', method='direct')/ov**2
  Np = N+ov-1
  # moment format: A,x,y,Cxx,Cxy,Cyy
  mom = numpy.asarray([1,0,0,4*ov**2,0,4*ov**2]).astype(numpy.float64)
  newmom = numpy.zeros_like(mom)
  con = .5 # convergence factor
  xx1 = numpy.tile(numpy.linspace(-(Np-1)/2., (Np-1)/2., Np), (Np,1))
  yy1 = numpy.copy(xx1.T)
  for iter in range(256):
    det = mom[3]*mom[5]-mom[4]**2
    xx = xx1-mom[1]
    yy = yy1-mom[2]
    G = numpy.exp((-mom[5]*xx**2 + 2*mom[4]*xx*yy - mom[3]*yy**2)/2./det) * bigStamp
    newmom[0] = numpy.sum(G)
    newmom[1] = numpy.sum(G*xx)
    newmom[2] = numpy.sum(G*yy)
    newmom[3] = numpy.sum(G*xx**2)
    newmom[4] = numpy.sum(G*xx*yy)
    newmom[5] = numpy.sum(G*yy**2)
    mom[0] = 2*newmom[0]
    err = newmom[1:]/newmom[0]; err[-3:] -= mom[-3:]/2.
    mom[1:] += err*con
  return(numpy.array([mom[0], mom[1]/ov, mom[2]/ov, (mom[3]+mom[5])/ov**2, (mom[3]-mom[5])/(mom[3]+mom[5]), 2*mom[4]/(mom[3]+mom[5])]))

# returns chi^2
# var = read noise variance
def chi2_postage_stamp(obs, theory, var):
  obs2 = numpy.maximum(obs+var, 1e-24)
  return(numpy.sum(theory+var-obs2-obs2*numpy.log((theory+var)/obs2))*2)
