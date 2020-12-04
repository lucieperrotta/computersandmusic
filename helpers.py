import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
from scipy import signal
from scipy.signal import butter, sosfilt, sosfreqz


# Moving average
def moving_average(X, N=10):
    '''
    Helper function to compute the smoothed sequence X using a moving average.
    '''
    augmented_X = np.concatenate((tuple([X[0] for i in range(int(N/2))]) , X , tuple([X[-1] for i in range(int(N/2)-1)])))
    res = np.convolve(augmented_X, np.ones((N,))/N, mode='valid')
    return res


# Butterworth filter
def butter_pass(lowhighcut, fs, btype, order=5):
        nyq = 0.5 * fs
        lowhigh =  lowhighcut / nyq
        sos = butter(order, lowhigh, analog=False, btype=btype, output='sos')
        return sos

def butter_pass_filter(data, lowhighcut, fs, btype, order=5):
        sos = butter_pass(lowhighcut, fs, btype, order=order)
        y = sosfilt(sos, data)
        return y

"""
Oversampling utilities
"""
def oversample(x, oversampling_rate=8):
    return signal.resample(x, x.size*oversampling_rate)

def undersample(x, oversampling_rate=8):
    down_x = butter_pass_filter(x, 1, oversampling_rate*2, "lowpass", order=10)
    return signal.resample(down_x, int(x.size / oversampling_rate))


# Copyright (c) 2011 Christopher Felton
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# The following is derived from the slides presented by
# Alexander Kain for CS506/606 "Special Topics: Speech Signal Processing"
# CSLU / OHSU, Spring Term 2011.
    
def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    

    return z, p, k


def freq_response(b, a):
    w, h = signal.freqz(b, a, whole=True)
    fig, ax1 = plt.subplots()

    ax1.plot(w, abs(h), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')
    plt.show()