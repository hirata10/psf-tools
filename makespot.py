import sys
import numpy
import spotutils
from astropy.io import fits
import time
import copy
from numpy.random import default_rng
import offset_index

# seed random number generator here
rng = default_rng(int(sys.argv[2]))

#spotutils.test_zernike()

# SED
sed = spotutils.SpectralEnergyDistribution('BB', [5800, 6.11e-5])
filter = spotutils.Filter('STH', [1.131, 1.454])
#filterK = spotutils.Filter('STH', [1.95, 2.30])

mask = spotutils.EmptyClass()
mask.N = 1
mask.array = spotutils.maskfiles.rim[5]

offsets = spotutils.EmptyClass()
offsets.par = numpy.zeros((offset_index.parNum))

# get info

# in file format: number pars used, pars[0:npUse], flags (prefer hex)

with open(sys.argv[3], 'r') as inFile: lines = inFile.readlines()
npUse = int(lines[0])
for i in range(npUse): offsets.par[i] = float(lines[i+1])
flags1 = int(lines[npUse+1])

addInfo = spotutils.EmptyClass()

# filter choice
filt = 'J'

print('initialized', time.asctime(time.localtime(time.time())))

Nsample = 50
pmom = numpy.zeros((Nsample, 9))
for k in range(Nsample):
  pmom[k,0] = rng.integers(1,19) # 19 exclusive
  pmom[k,1] = spotutils.sca.size*rng.uniform(-.5,.5)
  pmom[k,2] = spotutils.sca.size*rng.uniform(-.5,.5)
  addInfo.ctr = numpy.zeros((2))
  pmom[k,-6:] = spotutils.psfmoments(sed, filt, int(pmom[k,0]), pmom[k,1:3], offsets, addInfo)
  print('{:2d} {:6.2f} {:6.2f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}'.format(
    int(pmom[k,0]), pmom[k,1], pmom[k,2], pmom[k,3], pmom[k,4], pmom[k,5], pmom[k,6], pmom[k,7], pmom[k,8]))
numpy.savetxt(sys.argv[1]+'_samplemoms.txt', pmom) # sample moments

Nstar = 100
starspots = numpy.zeros((Nstar,spotutils.psSize,spotutils.psSize))
pos = numpy.zeros((Nstar,3))
fc = numpy.zeros((Nstar,3))

# BFE flag
if flags1%2==1:
  addInfo.bfe = True
  addInfo.bfe_a = 3e-7
if (flags1>>4)%2==1:
  addInfo.bfe = True
  addInfo.bfe_aplus = 3e-7

# vtpe flag
if (flags1>>1)%2==1:
  addInfo.vtpe = -.004

for k in range(Nstar):

  print(k, time.asctime(time.localtime(time.time())))
  pos[k,0] = rng.integers(1,19) # 19 exclusive
  pos[k,1] = spotutils.sca.size*rng.uniform(-.5,.5)
  pos[k,2] = spotutils.sca.size*rng.uniform(-.5,.5)
  addInfo.ctr = numpy.zeros((2))
  fc[k,1] = addInfo.ctr[0] = rng.uniform(0,1)
  fc[k,2] = addInfo.ctr[1] = rng.uniform(0,1)
  fc[k,0] = addInfo.F = rng.uniform(5e4,1e5)
  print(pos[k,:])
  starspots[k,:,:] = spotutils.postage_stamp(sed, filt, int(pos[k,0]), pos[k,-2:], offsets, addInfo)

  # special time dependent BFE mode
  if (flags1>>5)%2==1:
    offsetsY = copy.deepcopy(offsets)
    offsetsY.par[offset_index.foc] += .09
    starspots1 = spotutils.postage_stamp(sed, filt, int(pos[k,0]), pos[k,-2:], offsetsY, addInfo)
    mom_orig = spotutils.psfmoments(sed, filt, int(pos[k,0]), pos[k,-2:], offsets, addInfo)
    mom_1 = spotutils.psfmoments(sed, filt, int(pos[k,0]), pos[k,-2:], offsetsY, addInfo)
    print('change in ln T:  {:9.6f}'.format(numpy.log(mom_1[3]/mom_orig[3])))
    addInfoY = copy.deepcopy(addInfo)
    addInfoY.bfe_overwrite = True
    addInfoY.stamp_in = starspots[k,:,:]
    dstar = starspots1 - starspots[k,:,:]
    K1 = spotutils.postage_stamp(sed, filt, int(pos[k,0]), pos[k,-2:], offsetsY, addInfoY)\
         - spotutils.postage_stamp(sed, filt, int(pos[k,0]), pos[k,-2:], offsets, addInfoY)
    addInfoY.stamp_in = dstar
    K2 = spotutils.postage_stamp(sed, filt, int(pos[k,0]), pos[k,-2:], offsets, addInfoY)
    starspots[k,:,:] += (K1-K2)/6.

# classical non-linearity
if (flags1>>2)%2==1:
  starspots = starspots - 1e-6*starspots**2

# count rate non-linearity, exponent 1.01 at ref = 1e4 e
if (flags1>>3)%2==1:
  starspots = starspots*(starspots/1e4)**0.01

hdu = fits.PrimaryHDU(starspots.reshape((Nstar*spotutils.psSize,spotutils.psSize)))
hdu.writeto(sys.argv[1]+'_spotsIdeal.fits', overwrite=True)

starspots_obs = rng.poisson(starspots) + rng.normal(numpy.zeros_like(starspots), 7.)

hdu = fits.PrimaryHDU(starspots_obs.reshape((Nstar*spotutils.psSize,spotutils.psSize)))
hdu.header['RNGSEED'] = int(sys.argv[2])
hdu.writeto(sys.argv[1]+'_spotsObs.fits', overwrite=True)

numpy.savetxt(sys.argv[1]+'_starpos.txt', pos) # star sca & positions
numpy.savetxt(sys.argv[1]+'_fctr.txt', fc) # star flux & centroid

