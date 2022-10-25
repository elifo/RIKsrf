import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def set_style(whitegrid=False, scale=0.85):
  sns.set(font_scale = scale)
  sns.set_style('white')
  if whitegrid: sns.set_style('whitegrid')
  sns.set_context('talk', font_scale=scale)
#

# Konno-Ohmachi smoothening
def ko(y,dx,bexp):
  nx      = len(y)
  fratio  = 10.0**(2.5/bexp)
  ylis    = np.arange( nx )
  ylis[0] = y[0]

  for ix in np.arange( 1,nx ):
     fc  = float(ix)*dx
     fc1 = fc/fratio
     fc2 = fc*fratio
     ix1 = int(fc1/dx)
     ix2 = int(fc2/dx) + 1
     if ix1 <= 0:  ix1 = 1
     if ix2 >= nx: ix2 = nx
     a1 = 0.0
     a2 = 0.0
     for j in np.arange( ix1,ix2 ):
        if j != ix:
           c1 = bexp* np.log10(float(j)* dx/ fc)
           c1 = (np.sin(c1)/c1)**4
           a2 = a2+c1
           a1 = a1+c1*y[j]
        else:
           a2 = a2+1.0
           a1 = a1+y[ix]
     ylis[ix] = a1 / a2

  for ix in np.arange( nx ):
     y[ix] = ylis[ix]
  return y
#

def get_stf (self):
   fname = self.directory+ '/stf.dat'
   stf = np.loadtxt(fname)
   dt = stf[1,0]
   N = stf[:,0].size
   T = stf[-1,0]
   df = 1.0/ T
   f = np.arange(N/ 2- 1)* df
   M0 = self.M0

   # spectra
   spec = abs(np.fft.fft(stf[:,1]))* dt
   specD = spec[:int(N/2)-1]
   # specA = (2* pi* f)**2* specD
   #
   dsigma = self.dsigma
   dsigma [dsigma == 0] = -1.e99 
   dsigma = -(max(dsigma)/ 1.e6)
   #
   time = np.arange(len(stf[:,1]))* dt
   fc = (dsigma* 1.e6/ M0* 16.0/ 7.0)** (1./3.)
   fc *= 0.37* 4550.0
   print ('* fc (Hz): ', fc)
   print ('* Stress drop (MPa): ', dsigma)
   # Theoretical spectrum
   spectrum_theo = 1.0/ (1.0 + np.power((f/ fc), 2))
   spectrum_theo *= M0

   # smoothen by Kohno-Ohmachi
   dum = np.where( f > 0.5)[0]
   specD[dum]  = ko(specD[dum], df, 40.0)  

   return time, stf[:,1], f, specD, spectrum_theo, fc, dsigma 
#


class RIKsrf(object):
   """
   add explanation here ...
   """

   def __init__(self, directory=''):
      self.directory = directory+ '/'

      # read input file
      self.__read_inputfile()

      # read slip
      self.__read_final_slip()

      # read sub-sources
      self.__read_subsources()

      print('***')
   ###

   def __read_inputfile (self):

      fname = self.directory+ 'RIKsrf.in'
      data = pd.read_csv(fname, names=('Lf_all','Wf_all'), delim_whitespace=True, header=0, nrows=1, usecols=[0,1])
      self.Lf_all, self.Wf_all = float(data['Lf_all']), float(data['Wf_all'])
      print ('Fault length (km) and width (km): ', self.Lf_all, self.Wf_all)

      data = pd.read_csv(fname, names=('Lf','Wf'), delim_whitespace=True, header=2, nrows=1, usecols=[0,1])
      self.Lf, self.Wf = float(data['Lf']), float(data['Wf'])
      print ('Seismogeneic fault length (km) and width (km): ', self.Lf, self.Wf)

      data = pd.read_csv(fname, delim_whitespace=True, header=4, nrows=1)
      self.M0 = float(data.values[0][0])
      print ('Seismic moment (Nm): ', self.M0 )

      # only regular grid
      data = pd.read_csv(fname, delim_whitespace=True, header=6, nrows=1)
      is_regular_grid = int(data.values[0][0])
      if is_regular_grid != 1: print('Modify the script for irregular grids!!!')
      data = pd.read_csv(fname, delim_whitespace=True, header=6, nrows=2)
      self.nx, self.nz = int(data.values[1][0]), int(data.values[1][1])
      print ('Number of grid points (x,z): ', self.nx, self.nz)

      data = pd.read_csv(fname, names=('hypo_x','hypo_z'), delim_whitespace=True, header=9, nrows=1, usecols=[0,1])
      self.hypo_x, self.hypo_z = float(data['hypo_x']), float(data['hypo_z'])
      print ('Hypocenter location (x,z): ', self.hypo_x, self.hypo_z)

      data = pd.read_csv(fname, names=('smin','smax'), delim_whitespace=True, header=15, nrows=1, usecols=[0,1])
      self.SUBMIN, self.SUBMAX = int(data['smin']), int(data['smax'])
      print ('SUBMIN and SUBMAX: ', self.SUBMIN, self.SUBMAX)

      # total number of grid points
      self.npts = self.nx* self.nz
   ###

   def __read_final_slip (self):

      fname = self.directory+ '/slipdistribution.dat'
      self.x = np.genfromtxt(fname, usecols=0)
      self.z = np.genfromtxt(fname, usecols=1)
      self.slip = np.genfromtxt(fname, usecols=2)
      self.ruptime = np.genfromtxt(fname, usecols=4)
      self.dsigma = np.genfromtxt(fname, usecols=5)

   ###

   def __read_subsources(self):

      fname = self.directory+ '/subsources.dat'
      self.x_circle = np.genfromtxt(fname, usecols=0)
      self.z_circle = np.genfromtxt(fname, usecols=1)
      self.R_circle = np.genfromtxt(fname, usecols=2)
   ###


   def plot_grid(self):
      # Figure parameters
      fig = plt.figure(figsize=(8,6)); set_style(whitegrid=False)
      ax = fig.add_subplot(111)
      ax.set_xlabel('Along strike (km)')
      ax.set_ylabel('Along up-dip (km)')
      ax.set_xlim([-0.2, 1.01* self.Lf])
      ax.set_ylim([-0.2, 1.01* self.Wf])

      ax.scatter(self.x, self.z, color='black', alpha=0.8, marker='.')
      # ax.plot(self.xhypo,self.yhypo, marker='*', color='red',markersize=20)
      plt.show()
  ###


   def plot_final_slip(self,cmap='magma'):

      fig = plt.figure(figsize=(10,4)); set_style(whitegrid=False)
      ax = fig.add_subplot(111)
      ax.set_xlabel('Along strike (km)')
      ax.set_ylabel('Along up-dip (km)')
      ax.set_xlim([0,self.Lf])
      ax.set_ylim([-0.5, 1.01* self.Wf])
      _ext = [0.0, self.Lf, 0.0, self.Wf]

      ax.set_title('Normalised by '+ '%.2f' % self.slip.max())
      slip = self.slip.copy()
      slip /= slip.max()

      vmin = slip.min(); vmax = slip.max()
      print ('min and max values:', vmin, vmax)
      grid = slip.reshape((self.nx, self.nz))
      im = plt.imshow(grid, extent=_ext, interpolation='bilinear', cmap=cmap, origin='lower')

      cb = fig.colorbar(im, shrink=0.25, aspect=10, pad=0.01, ticks=np.linspace(vmin, vmax, 5))
      cb.set_label('Normalised slip (m)', labelpad=20, y=0.5, rotation=90)

      ax.plot(self.hypo_x, self.hypo_z, marker='*', color='k',markersize=20, alpha=0.75)   
      plt.show()
   ###

   def plot_subsources(self):

      asp=max(1.0, self.Lf/self.Wf)

      from matplotlib.patches import Circle
      fig = plt.figure(figsize=(6*asp,6)); set_style(whitegrid=False)
      ax = fig.add_subplot(111)

      # ax.set_xlabel('Along strike (km)')
      # ax.set_ylabel('Along up-dip (km)')
      ax.set_xlim([0,self.Lf])
      ax.set_ylim([-1.0, 1.01* self.Wf])

      alphas = [0.9* rad/max(self.R_circle)+ 0.1 for rad in self.R_circle]
      for _alpha, x, z, R in zip(alphas, self.x_circle, self.z_circle, self.R_circle):
         circ = Circle((x,z), R, fill=False, lw=1, color='k', alpha=_alpha)
         ax.add_patch(circ)     
      ###
      tit  = 'Total number of subsources = '+ '%d' % len(self.R_circle)
      # ax.set_title(tit+ '\n')
      # ax.plot(self.hypo_x, self.hypo_z, marker='*', color='red', markersize=20)   
      plt.show()
   ###

   def plot_slip_rates(self, gfortran=True, dt=0.01):


      fname = self.directory+ 'sr.dat'
      num_lines = sum(1 for line in open(fname))

      NSR = self.npts      
      NT = int(num_lines/(NSR))- 2

      print ('***', NSR, NT)
      srate = np.zeros((NSR,NT))



      if gfortran:
         with open (fname, 'r') as f:
            lines = f.read().splitlines()
            for ii in range(NSR):
               # print (ii, ii*(NT+2),(ii+1)*(NT+2) )
               data = lines[ii*(NT+2): (ii+1)*(NT+2)-2]
               rate = np.array([float(dum) for dum in data])
               srate[ii,:] = rate
         print('File read!')
      ###
      else:
         print ('Here reading file with only a single column')
         print ('Check your file format!')
         exit
      #             # ifort
      #             # 2 columns: time, slip rate  


      # PLOT
      asp=0.5*self.Lf/self.Wf
      fig = plt.figure(figsize=(8,6)); set_style(whitegrid=False)
      ax = fig.add_subplot(111)

      ax.set_xlabel('Along strike (km)')
      ax.set_ylabel('Along up-dip (km)')
      ax.set_xlim([0,self.Lf])
      ax.set_ylim([-0.5, 1.01* self.Wf])
      _ext = [0.0, self.Lf, 0.0, self.Wf]

      ax.plot(self.hypo_x,self.hypo_z, marker='*', color='black',markersize=15)

      xstep = self.Lf/ self.Wf
      ystep = self.Wf/ self.nz

      time = np.arange(NT)* dt

      k = -1
      for j in range(self.nz):
         for i in range(self.nx):
            k += 1
            # print x[k],y[k]
            if i%2 == 0  and j%2 == 0:
               ax.plot(time* 0.15+ i* xstep  ,srate[k,:]* 1.0+ j* ystep, color='k',lw=0.5)

      ###
      plt.show()
   ###

   def plot_source_function(self):
       #
       t1, STF1, f1, spec1, Brune1, fc1, dsigma1 = get_stf(self)
       # Plot
       fig = plt.figure(figsize=(8,4))
       set_style(whitegrid=True, scale=1.0)
       # STF
       ax = fig.add_subplot(121)
       ax.plot(t1, STF1, color='k', lw=1.2)
       ax.set_xlabel('Time (s)'); ax.set_ylabel('Moment rate (Nm/s)')
       # Moment spectra
       ax = fig.add_subplot(122)
       ax.set_xscale('log'); ax.set_yscale('log')   
       plt.minorticks_on()
       plt.grid(which='minor', linestyle='--', lw=0.2, color='gray')
       ax.plot(f1, Brune1, color='k', lw=1, linestyle='--', alpha=0.7, label='Brune')
       ax.plot(f1, spec1, color='k', lw=1.2, label='RIKsrf')
       ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Moment spectra')
       ax.legend(prop={'size':11}, loc='upper right')

       plt.tight_layout(); plt.show()
   ###
   
### end class