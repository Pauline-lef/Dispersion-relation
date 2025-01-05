import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cm = 1/2.54
g  = 9.81 #m.s-2

#%% Data extraction

fichier_path = 'out_Wave_8cm_10V.data'
h  = 0.08 # m
dt = 1/50 # s
data = pd.read_csv(fichier_path, delim_whitespace=True, header=None)
    
# Convertion to numpy array
L   = 0.8 # m
T   = data[0][len(data[0])-1] * 30/50 # The data were created using a resolution of 30fps.
N_x = max(data[1])+1 
N_t = int(len(data[0])/N_x)
    
interface = np.transpose(np.reshape(data[3], (N_t, N_x)))

x, dx = np.linspace(0, L, N_x, retstep=True, endpoint=False)
t = np.linspace(0, T, N_t)
    
# Height scaling
moy = np.mean(interface[:, 0])
interface = (interface-moy)*L/N_x

plt.figure('Spatio-temporal', figsize=(25*cm, 20*cm))
plt.title('Spatio-temporal', fontsize=18)
pc = plt.pcolormesh(t, x, interface*100, cmap='turbo')
plt.xlabel('t (s)', fontsize=18)
plt.ylabel('x (m)', fontsize=18)
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
plt.xlim([-150, 150])
plt.ylim([-100, 100])
plt.tick_params(length=6, width=1, colors='k', grid_color='k', labelsize=16)
cb = plt.colorbar(pcolor, orientation='vertical') 
cb.set_label(label=r'E ($cm^2$s)', size=18)
pcolor.figure.axes[1].tick_params(axis="y", labelsize=16)

#%% Theory

k_th = np.linspace(-150, 150, 1000)
omega_th = np.sqrt(g*k_th*np.tanh(h*k_th))

plt.figure('DTF2D')
plt.plot(k_th, omega_th, 'darkorange', label='h=0.08m')
plt.tick_params(length=6, width=1, colors='k', grid_color='k', labelsize=14)
plt.legend(fontsize=14)
