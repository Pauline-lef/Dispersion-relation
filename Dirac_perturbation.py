import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

cm = 1/2.54
g  = 9.81 #m.s-2

def interp(field_fft, k, factor):
    """
    Returns the spectrum for which zeros have been added in space-time  (0-padding)
    The spectrum has been interpolated by multiplying the number of points by a factor
    """
    field_fft = np.transpose(field_fft)
    field = np.fft.ifft(np.fft.fftshift(field_fft, axes=1), axis=1)
    N = len(k)
    N_z = len(field_fft)
    zeros = np.zeros((N_z, int(N*factor/2)))
    new_field = np.concatenate((zeros, field, zeros), axis=1)
    new_field_fft = np.transpose(np.fft.fftshift(np.fft.fft(new_field, axis=1), axes=1))
    return (factor+1)*abs(new_field_fft)


#%% Data extraction

fichier_path = 'out_Dirac_8cm.data'

h  = 0.08 # m
dt = 1/25 # s
data = pd.read_csv(fichier_path, delim_whitespace=True, header=None)
    
# Convertion to numpy array
L = 0.4  # m 
l = 0.04 # m
T = data[0][len(data[0])-1] * 30/25 # ! problème de dt au niveau du fichier
N_x = max(data[1])+1 
N_t = int(len(data[0])/N_x)
    
interface = np.transpose(np.reshape(data[3], (N_t, N_x))) 


x, dx = np.linspace(0, L, N_x, retstep=True, endpoint=False)
t = np.linspace(0, T, N_t)


# Correction on camera slope
droite = lambda x, A, B : A*x + B
droite_fit, pcov = curve_fit(droite, x, interface[:, 0], maxfev=1000000000)
A_fit = droite_fit[0]
B_fit = droite_fit[1]
for i in range(len(t)):
    interface[:, i] = interface[:, i] - A_fit*x - B_fit

# Height scaling
moy = np.mean(interface[:, 0])
interface = (interface-moy)*L/N_x

plt.figure('Spatio-temporel', figsize=(25*cm, 20*cm))
plt.title('Spatio-temporal', fontsize=18)
pc = plt.pcolormesh(t, x, interface*100, cmap='turbo')
plt.xlabel('t (s)', fontsize=18)
plt.ylabel('x (m)', fontsize=18)
plt.xlim([0, 5])
cb = plt.colorbar(pc, orientation='vertical') 
cb.set_label(label=r'$\eta$ (cm)', size=18)
plt.tick_params(length=6, width=1, colors='k', grid_color='k', labelsize=16)
pc.figure.axes[1].tick_params(axis="y", labelsize=16)


#%% FFT2

# Space frequency
f = np.fft.fftfreq(N_x, dx)
k = np.pi*2 * f

# Time frequency
nu = np.fft.fftfreq(N_t, dt)
omega = np.pi*2 * nu

Interface = np.fft.fft2(interface) / (N_t*N_x) 
Interface[:, omega==0] = 0 # Average removed

plt.figure('DTF2D', figsize=(25*cm, 20*cm))
plt.title('Spectrum', fontsize=18)
pcolor = plt.pcolormesh(np.fft.fftshift(k), np.fft.fftshift(omega), np.transpose(np.fft.fftshift(np.abs(Interface)))*1e4, cmap='turbo')
plt.xlabel('k $(m^{-1})$', fontsize=18)
plt.ylabel('$\omega (s^{-1})$', fontsize=18)
plt.tick_params(length=6, width=1, colors='k', grid_color='k', labelsize=16)
cb = plt.colorbar(pcolor, orientation='vertical') 
cb.set_label(label=r'E ($cm^2$s)', size=18)
pcolor.figure.axes[1].tick_params(axis="y", labelsize=16)


#%% Transverses
Nmax = 3
legend = False
for N in range(1, Nmax+1):
    if legend == False:
        omega_tr = np.sqrt(g*np.sqrt(k**2 + (N*np.pi/l)**2)*np.tanh(h*np.sqrt(k**2 + (N*np.pi/l)**2)))
        plt.plot(k, omega_tr, 'w', label='Transversal modes')
        legend = True
    else:
        omega_tr = np.sqrt(g*np.sqrt(k**2 + (N*np.pi/l)**2)*np.tanh(h*np.sqrt(k**2 + (N*np.pi/l)**2)))
        plt.plot(k, omega_tr, 'w')
    
#%% Theorique

omega_th = np.sqrt(g*k*np.tanh(h*k))


plt.figure('DTF2D')
plt.plot(k, omega_th, 'darkorange', label='Theory')
plt.xlim([-800, 800])
plt.ylim([0, max(omega)])
plt.tick_params(length=6, width=1, colors='k', grid_color='k', labelsize=16)
plt.legend(fontsize=14, loc=2)

#%% Interpolation
I = 15
if I > 0 :
    Interface_interp = interp(Interface, t, I)
    k_interp = 2*np.pi* np.fft.fftfreq(len(Interface_interp), dx)
    
plt.figure('DTF2D_interp', figsize=(25*cm, 20*cm))
plt.title('Interpolated spectrum', fontsize=18)
pcolor = plt.pcolormesh(np.fft.fftshift(k_interp), np.fft.fftshift(omega), np.transpose(np.fft.fftshift(np.abs(Interface_interp))), cmap='turbo')
plt.xlabel('k $(m^{-1})$', fontsize=16)
plt.ylabel('$\omega (s^{-1})$', fontsize=16)
plt.tick_params(length=6, width=1, colors='k', grid_color='k', labelsize=14)
plt.xlim([-800, 800])
plt.ylim([0, max(omega)])
cb = plt.colorbar(pcolor, orientation='vertical') 
cb.set_label(label=r'E ($cm^2$s)', size=18)
pcolor.figure.axes[1].tick_params(axis="y", labelsize=16)



#%% Data filters
oo, kk = np.meshgrid(omega, k)
oo_th = np.sqrt(g*kk*np.tanh(h*kk))*1.5
pas_good_o = ((abs(oo)>oo_th) & (abs(oo)>-oo_th)) 
Interface_filt  = np.copy(Interface) * (N_t*N_x)
Interface_filt2 = np.copy(Interface) * (N_t*N_x)
Interface_filt[np.logical_not(pas_good_o)] = 0
Interface_filt2[pas_good_o] = 0

plt.figure('Tranversal modes', figsize=(25*cm, 20*cm))
plt.title('Tranversal modes filter', fontsize=18)
pcolor = plt.pcolormesh(np.fft.fftshift(k), np.fft.fftshift(omega), np.transpose(np.fft.fftshift(np.abs(Interface_filt))), cmap='turbo')
plt.xlabel('k $(m^{-1})$', fontsize=18)
plt.ylabel('$\omega (s^{-1})$', fontsize=18)
plt.xlim([-800, 800])
plt.ylim([0, max(omega)])
plt.tick_params(length=6, width=1, colors='k', grid_color='k', labelsize=16)
cb = plt.colorbar(pcolor, orientation='vertical') 
cb.set_label(label=r'E ($cm^2$s)', size=18)
pcolor.figure.axes[1].tick_params(axis="y", labelsize=16)


# Inverse Fourier Transform for both filtered spectra

new_interface1 = np.real(np.fft.ifft2(Interface_filt))
plt.figure('Tranversal modes', figsize=(25*cm, 20*cm))
plt.title('Spatio-temporal of the tranversal modes', fontsize=18)
pc = plt.pcolormesh(t, x, new_interface1*100, cmap='turbo')
plt.xlabel('t (s)', fontsize=18)
plt.ylabel('x (m)', fontsize=18)
cb = plt.colorbar(pc, orientation='vertical') 
cb.set_label(label=r'$\eta$ (cm)', size=18)
plt.xlim([0, 5])
plt.tick_params(length=6, width=1, colors='k', grid_color='k', labelsize=16)
pc.figure.axes[1].tick_params(axis="y", labelsize=16)

new_interface2 = np.real(np.fft.ifft2(Interface_filt2))
plt.figure('Dispersion relation', figsize=(25*cm, 20*cm))
plt.title('Spatio-temporal of the dispersion relation', fontsize=18)
pc = plt.pcolormesh(t, x, new_interface2*100, cmap='turbo')
plt.xlabel('t (s)', fontsize=18)
plt.ylabel('x (m)', fontsize=18)
cb = plt.colorbar(pc, orientation='vertical') 
plt.xlim([0, 5])
cb.set_label(label=r'$\eta$ (cm)', size=18)
plt.tick_params(length=6, width=1, colors='k', grid_color='k', labelsize=16)
pc.figure.axes[1].tick_params(axis="y", labelsize=16)
