import sys
import numpy
import copy
import time
from astropy.io import fits
import spotutils
import offset_index
import multiprocessing

# *** LOAD DATA ***

pos = numpy.loadtxt(sys.argv[1]+'_starpos.txt')
sca = (pos[:,0]+1e-6).astype(numpy.int16)
pos = pos[:,-2:]
Nstar = numpy.size(sca)
print(Nstar, 'stars')

if int(sys.argv[2])==1:
  fh = fits.open(sys.argv[1]+'_spotsIdeal.fits')
else:
  fh = fits.open(sys.argv[1]+'_spotsObs.fits')
spotsObs = numpy.copy(fh[0].data.reshape((Nstar,spotutils.psSize,spotutils.psSize)))
fh.close()

print(sca)
print(pos)

print('start', time.asctime(time.localtime(time.time())))

fit_noise = 6.5

# Initial conditions
sF = numpy.ones((Nstar))*3e4
sCtr = numpy.ones((Nstar,2))/2

# choose filter and SED we are fitting
filt = 'J'
sed = spotutils.SpectralEnergyDistribution('BB', [5800, 6.11e-5])

offsets = spotutils.EmptyClass(); offsets.par = numpy.zeros((offset_index.parNum))

FastMode = True

# quartic minimum
def quartmin(ar):
  p = numpy.polyfit(numpy.linspace(-2,2,5), ar, 4)
  n = 10000
  x = numpy.linspace(-2,2,n+1)
  px = p[0]*x**4+p[1]*x**3+p[2]*x**2+p[3]*x+p[4]
  newx = numpy.argmin(px)*4./n-2.
  x = newx + .01*numpy.linspace(-1,1,n+1)
  px = p[0]*x**4+p[1]*x**3+p[2]*x**2+p[3]*x+p[4]
  newx2 = newx + .01*(numpy.argmin(px)*2./n-1)
  return(newx2)

# iterate positions, fluxes
def iterStar(nloop, vern):
  for iloop in range(nloop):

    for k in range(Nstar):
      addInfo = spotutils.EmptyClass()
      addInfo.FastMode = FastMode
      addInfo.F = sF[k]; addInfo.ctr = sCtr[k,:]
      theory = spotutils.postage_stamp(sed, filt, sca[k], pos[k,:], offsets, addInfo)
      qmin = 1
      if vern: qmin = 3
      for q in range(qmin,6):
        nc2 = numpy.zeros((5))
        factor = 2**(.5**q)
        theory /= factor**2
        for j in range(5):
          nc2[j] = spotutils.chi2_postage_stamp(spotsObs[k,:,:], theory, fit_noise**2)
          theory *= factor
        theory /= factor**3
        x = factor**quartmin(nc2)
        theory *= x; sF[k] *= x; addInfo.F = sF[k]

      # on a grid of points
      if not vern:
        for a in range(3):
          addInfoX = copy.deepcopy(addInfo)
          addInfoX.many = True
          addInfoX.force_ov=8
          theoryX = spotutils.postage_stamp(sed, filt, sca[k], pos[k,:], offsets, addInfoX)
          c2 = numpy.zeros((25))
          for l in range(25):
            c2[l] = spotutils.chi2_postage_stamp(spotsObs[k,:,:], theoryX[l,:,:], fit_noise**2)
          if a==0:
            l = numpy.argmin(c2)
            sCtr[k,0] -= (l//5 -2)/8.
            sCtr[k,1] -= (l% 5 -2)/8.
          if a==1 or a==2:
            c2x = c2[2::5]
            if a==2: c2x = c2[20:25]
            xmin = quartmin(c2x)
            sCtr[k,a-1] -= xmin/8.
          addInfo.ctr = sCtr[k,:]
      else:
        # scan spot over a smaller range
        for ax in range(4):
          d = .01
          if ax%2==0: d = .05
          addInfoX = copy.deepcopy(addInfo)
          nc2 = numpy.zeros((5))
          addInfoX.ctr[ax//2] -= 3*d
          for kdp in range(5):
            addInfoX.ctr[ax//2] += d
            theory = spotutils.postage_stamp(sed, filt, sca[k], pos[k,:], offsets, addInfoX)
            nc2[kdp] = spotutils.chi2_postage_stamp(spotsObs[k,:,:], theory, fit_noise**2)
          scale1 = quartmin(nc2)
          addInfoX.ctr[ax//2] += d*(scale1-2)
          sCtr[k,:] = addInfoX.ctr

#
# -- end

# use scipy optimizer. first the target function

def chi2tot(parvec):
  offsets.par[:] = parvec
  iterStar(3, False)
  res = 0.
  for k in range(Nstar):
    addInfo = spotutils.EmptyClass()
    addInfo.FastMode = FastMode
    addInfo.F = sF[k]; addInfo.ctr = sCtr[k,:]
    theory = spotutils.postage_stamp(sed, filt, sca[k], pos[k,:], offsets, addInfo)
    res += spotutils.chi2_postage_stamp(spotsObs[k,:,:], theory, fit_noise**2)
  return(res)

#x_init = numpy.zeros((offset_index.parNum))
#myBounds = ()
#for j in range(offset_index.parNum):
#  x_max = .1
#  if offset_index.amask[j]: x_max = .045
#  x_min = -x_max
#  myBounds = myBounds + ((x_min,x_max),)
#print('myBounds =', myBounds)
#p_opt = minimize(chi2tot, x_init, method='Powell', bounds=myBounds, options={'return_all':True, 'xtol':1e-5, 'disp':True})
#offsets.par[:] = p_opt

dstep = numpy.zeros((offset_index.parNum))
for j in range(offset_index.parNum):
  dstep[j] = .04
  if offset_index.amask[j]: dstep[j]=.01

for k_it in range(12):
  print('iteration', k_it, time.asctime(time.localtime(time.time())))
  if k_it==4: dstep = dstep/4.
  #if k_it==10: FastMode=False

  iterStar(1, k_it>=2)
  #print(sF)
  #print(sCtr)
  old_offsets = numpy.copy(offsets.par)
  for j in range(2*offset_index.parNum+1):
    if j<2*offset_index.parNum:
      sdir = numpy.zeros((offset_index.parNum))
      sdir[j//2] = dstep[j//2]
      if j%2==1 and abs(scale1)<1: sdir[j//2] *= abs(scale1)
    else:
      sdir = offsets.par - old_offsets
    nc2 = numpy.zeros((5))
    offsets.par -= 3*sdir
    for kdp in range(5):
      offsets.par += sdir
      #
      # this is split into 2 processes for speed
      #
      mysum = multiprocessing.Value('d', 0)
      lock = multiprocessing.Lock()
      def wrapfunc(k):
        addInfo = spotutils.EmptyClass()
        addInfo.FastMode = FastMode
        addInfo.F = sF[k]; addInfo.ctr = sCtr[k,:]
        theory = spotutils.postage_stamp(sed, filt, sca[k], pos[k,:], offsets, addInfo)
        thisterm = spotutils.chi2_postage_stamp(spotsObs[k,:,:], theory, fit_noise**2)
        lock.acquire()
        try: mysum.value += thisterm
        finally: lock.release()
      with multiprocessing.Pool(processes=2) as pool: pool.map(wrapfunc, range(Nstar))
      nc2[kdp] = mysum.value
      #
      # multiprocessing part ends here
      #
    scale1 = quartmin(nc2)
    print('    ->', nc2, '{:9.6f}'.format(scale1))
    offsets.par += sdir*(scale1-2)
    print(k_it,j,offsets.par); sys.stdout.flush()
  print('')

print('fit complete', time.asctime(time.localtime(time.time())))

# now get information from sample moments
pmom = numpy.loadtxt(sys.argv[1]+'_samplemoms.txt')
Nsample = numpy.shape(pmom)[0]
pmom_all = numpy.zeros((Nsample, 15))
pmom_all[:,:9] = pmom
addInfoM = spotutils.EmptyClass()
for k in range(Nsample):
  addInfoM.ctr = numpy.zeros((2))
  pmom_all[k,-6:] = spotutils.psfmoments(sed, filt, int(pmom[k,0]+1e-8), pmom[k,1:3], offsets, addInfoM)
  print('{:2d} {:6.2f} {:6.2f} {:9.6f} {:9.6f} {:9.6f}'.format(
    int(pmom[k,0]+1e-8), pmom[k,1], pmom[k,2], numpy.log(pmom_all[k,-3]/pmom[k,-3]), pmom_all[k,-2]-pmom[k,-2], pmom_all[k,-1]-pmom[k,-1]))
numpy.savetxt(sys.argv[1]+'_samplemoms_compare.txt', pmom_all) # sample moments

print('')
err_tr = numpy.log(pmom_all[:,-3]/pmom[:,-3])
err_e = pmom_all[:,-2:]-pmom[:,-2:]
eres = numpy.copy(err_e)
for i in range(2): eres[:,i] -= numpy.mean(err_e[:,i])
print('trace: {:9.6f} {:9.6f}'.format(numpy.mean(err_tr), numpy.std(err_tr)))
print('ellip: {:9.6f} {:9.6f} {:9.6f}'.format(numpy.mean(err_e[:,0]), numpy.mean(err_e[:,1]), numpy.std(eres)))

print('end', time.asctime(time.localtime(time.time())))
