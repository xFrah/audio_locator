"""Implementation of a 3D Audio Panner using the CIPIC HRTF Database.

Usage:

Select a subject from the CIPIC database. You should select a subject 
with similar anthropometric measurements as yourself for the best
experience.

    - Note: Due to storage limitations, the repository has only 4
            subjects of the database to choose from. The full database
            is ~170MB and has 45 subjects. It can be downloaded for free
            at:

            https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/
            
            In order to make it work, you should simply replace the 
            folder ´CIPIC_hrtf_database´ with the one you downloaded.

Press 'Play' to start playing the default sound file. Move the Azimuth
and Elevation sliders to position the sound in the 3D space.

You can load your own audio file in File/Load audio file. Also, there
are other sound samples in the folder resources/sound.

*IMPORTANT:* For now, the only working format is a mono WAV file at
44100 Hz sample rate and 16 bit depth. 

You can save the file at the specified pair of Azimuth/Elevation in
File/Save audio file.

Lastly, you can choose to use a crossover in order not to spatialize low 
frequencies, since low frequencies are non-directional in nature. Go to 
Settings/Change cutoff frequency to set the desired frequency. By
default, crossover is set at 200 Hz.

Author:         Francisco Rotea
                (Buenos Aires, Argentina)
Repository:     https://github.com/franciscorotea
Email:          francisco.rotea@gmail.com

"""

import os
import wave
import itertools

import scipy.io
from scipy.io import wavfile
import scipy.spatial
import pyaudio
import scipy.signal
import librosa

import numpy as np
import numpy.linalg

# Values of azimuth and elevation angles measured in the CIPIC database. 
# See ´CIPIC_hrtf_database/doc/hrir_data_documentation.pdf´ for
# information about the coordinate system and measurement procedure.

AZIMUTH_ANGLES = [
    -80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15,
    20, 25, 30, 35, 40, 45, 55, 65, 80,
    ]

ELEVATION_ANGLES = -45 + 5.625*np.arange(0, 50)

POINTS = np.array(list(itertools.product(AZIMUTH_ANGLES, ELEVATION_ANGLES)))

# Get indexes from angles.

AZ = dict(zip(AZIMUTH_ANGLES, np.arange(len(AZIMUTH_ANGLES))))
EL = dict(zip(ELEVATION_ANGLES, np.arange(len(ELEVATION_ANGLES))))

# Load anthropometric measurements data from the CIPIC database.

# See ´CIPIC_hrtf_database/doc/anthropometry.pdf´ for information about 
# the parameters definition.

anthro_data = scipy.io.loadmat('CIPIC_hrtf_database/anthropometry/anthro.mat')

PARAMETERS = {
    'info': ['Age:', 'Sex:', 'Weight:'],
    'X': ['Head width:', 'Head height:', 'Head depth:', 'Pinna offset down:', 
          'Pinna offset back:', 'Neck width:', 'Neck height:', 'Neck depth:', 
          'Torso top width:', 'Torso top height:', 'Torso top depth:', 
          'Shoulder width:', 'Head offset forward:', 'Height:',
          'Seated height:', 'Head circumference:', 'Shoulder circumference:'],
    'D': ['Cavum concha height', 'Cymba concha height', 'Cavum concha width', 
          'Fossa height', 'Pinna height', 'Pinna width', 
          'Intertragal incisure width', 'Cavum concha depth'],
    'theta': ['Pinna rotation angle:', 'Pinna flare angle:']
}

L_R = [' (left):', ' (right):']   # To use with 'D' and 'theta' parameters.

# Clean anthropometric data for display.

for key, value in anthro_data.items():
    if key not in ['__header__', '__version__', '__globals__', 'id', 'sex']:
        if key == 'age':
            anthro_data[key][np.isnan(anthro_data[key])] = 0
            anthro_data[key] = np.squeeze(value.astype('int')).astype('str')
            anthro_data[key][anthro_data[key] == '0'] = '-'
        else:
            anthro_data[key] = np.around(np.squeeze(value), 1).astype('str')
            anthro_data[key][anthro_data[key] == 'nan'] = '-'

# Get indexes from ID's.

ANTHRO_ID = anthro_data['id'].flatten().tolist()
ID_TO_IDX = dict(zip(ANTHRO_ID, range(len(ANTHRO_ID))))

# Generate a list with all subject's ID present in the database.

FOLDERS = os.listdir('CIPIC_hrtf_database/standard_hrir_database')
SUBJECT_ID = [id_.strip('subject_') for id_ in FOLDERS if id_ != 'show_data']

# Initialization variables for audio stream.

SAMPLE_RATE = 44100

# Initialization variables for overlap-save algorithm.

# L = Window size.
# M = Length of impulse response.
# N = Size of the DFT. Since the length of the convolved signal will be
#     L+M-1, it is rounded to the nearest power of 2 for efficient fft
#     calculation.

L = 2048
M = 200
N = int(2**np.ceil(np.log2(np.abs(L+M-1))))

L = N - M + 1

# Preallocate interpolated impulse responses.

interp_hrir_l = np.zeros(M)
interp_hrir_r = np.zeros(M)


def load_and_resample(file_path, target_sr=SAMPLE_RATE):
    """Load an audio file, convert it to mono, and resample if necessary."""
    try:
        import soundfile as sf
        data, sr = sf.read(file_path)
    except Exception as e:
        # Fall back to scipy wavfile if soundfile fails
        sr, data = scipy.io.wavfile.read(file_path)
        data = data.astype(np.float32) / 32768.0  # normalize
        
    if data.ndim > 1:
        data = data[:, 0]
        
    if sr != target_sr:
        print(f"Resampling from {sr} to {target_sr}...")
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        
    return data

def butter_lp(cutoff, fs, order):
    """Design of a digital Butterworth low pass filter with a
    second-order section format for numerical stability."""

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    sos = scipy.signal.butter(N=order,
                              Wn=normal_cutoff,
                              btype='lowpass',
                              output='sos')

    return sos


def butter_lp_filter(signal, cutoff, fs=SAMPLE_RATE, order=1):
    """Filter a signal with the filter designed in ´butter_lp´.

    Filfilt applies the linear filter twice, once forward and once
    backwards, so that he combined filter has zero phase delay."""

    sos = butter_lp(cutoff=cutoff, fs=fs, order=order)
    out = scipy.signal.sosfiltfilt(sos, signal)

    return out


def butter_hp(cutoff, fs, order):
    """Design of a digital Butterworth high pass filter with a
    second-order section format for numerical stability."""

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    sos = scipy.signal.butter(N=order,
                              Wn=normal_cutoff,
                              btype='highpass',
                              output='sos')

    return sos


def butter_hp_filter(signal, cutoff, fs=SAMPLE_RATE, order=1):
    """Filter a signal with the filter designed in ´butter_hp´.

    Filfilt applies the linear filter twice, once forward and once
    backwards, so that he combined filter has zero phase delay."""

    sos = butter_hp(cutoff=cutoff, fs=fs, order=order)
    out = scipy.signal.sosfiltfilt(sos, signal)

    return out


def create_triangulation(points):
    """Generate a triangular mesh from HRTF measurement points (azimuth,
    elevation) using the Delaunay triangulation algorithm."""

    triangulation = scipy.spatial.Delaunay(points)

    return triangulation


def calculate_T_inv(triang, points):
    """Performs the calculation of the inverse of matrix T for all
    triangles in the triangulation and stores it in an array.

    Matrix T is defined as:

        T = [[A - C],
             [B - C]]

    where A, B and C are vertices of the triangle.

    Since T is independent of source position X, the precalculation of T
    allows to reduce the operational counts for finding the
    interpolation weights.
    
    For a more comprehensive explanation of this procedure, refer to:

    Gamper, H., Head-related transfer function interpolation in azimuth,
    elevation, and distance, J. Acoust. Soc. Am. 134 (6), December 2013.

    """

    A = points[triang.simplices][:,0,:]
    B = points[triang.simplices][:,1,:]
    C = points[triang.simplices][:,2,:]

    T = np.empty((2*A.shape[0], A.shape[1]))

    T[::2,:] = A - C
    T[1::2,:] = B - C

    T = T.reshape(-1, 2, 2)

    T_inv = np.linalg.inv(T)

    return T_inv


def interp_hrir(triang, points, T_inv, hrir_l, hrir_r, azimuth, elevation):
    """Estimate a HRTF for any point X lying inside the triangular mesh 
    calculated.

    This is done by interpolating the vertices of the triangle enclosing
    X. Given a triangle with vertices A, B and C, any point X inside the
    triangle can be represented as a linear combination of the vertices:

    X = g_1 * A + g_2 * B + g_3 * C

    where g_i are scalar weights. If the sum of the weights is equal to
    1, these are barycentric coordinates of point X. Given a desired
    source position X, barycentric interpolation weights are calculated
    as:

    [g_1, g_2] = (X - C) * T_inv
    g_ 3 = 1 - g_1 - g_2

    Barycentric coordinates are used as interpolation weights for
    estimating the HRTF at point X as the weighted sum of the HRTFs
    measured at A, B and C, respectively.

    One of the main advantages of this interpolation approach is that
    it does not cause discontinuities in the interpolated HRTFs: for a
    source moving smoothly from one triangle to another, the HRTF
    estimate changes smoothly, even at the crossing point.

    For a more comprehensive explanation of the interpolation algorithm,
    please refer to:

    Gamper, H., Head-related transfer function interpolation in azimuth,
    elevation, and distance, J. Acoust. Soc. Am. 134 (6), December 2013.

    """

    position = [azimuth, elevation]
    triangle = triang.find_simplex(position)
    vert = points[triang.simplices[triangle]]

    X = position - vert[2]
    g = np.dot(X, T_inv[triangle])

    g_1 = g[0]
    g_2 = g[1]
    g_3 = 1 - g_1 - g_2

    if g_1 >= 0 and g_2 >= 0 and g_3 >= 0:
        interp_hrir_l[:] = g_1 * hrir_l[AZ[vert[0][0]]][EL[vert[0][1]]][:] + \
                           g_2 * hrir_l[AZ[vert[1][0]]][EL[vert[1][1]]][:] + \
                           g_3 * hrir_l[AZ[vert[2][0]]][EL[vert[2][1]]][:]

        interp_hrir_r[:] = g_1 * hrir_r[AZ[vert[0][0]]][EL[vert[0][1]]][:] + \
                           g_2 * hrir_r[AZ[vert[1][0]]][EL[vert[1][1]]][:] + \
                           g_3 * hrir_r[AZ[vert[2][0]]][EL[vert[2][1]]][:]
    
    return interp_hrir_l, interp_hrir_r



TRI = create_triangulation(points=POINTS)
T_INV = calculate_T_inv(triang=TRI, points=POINTS)
