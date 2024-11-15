"""
Acoustic Preprocessing.Filter Tutorial
==============================================================
"""

######################################################################

# !pip install pysensing

######################################################################
# In this tutorial, we will be implementing a simple acoustic.preprocessing.filtering
# 
import torch
import torchaudio
import matplotlib.pyplot as plt
import pysensing.acoustic.preprocessing.filtering as filtering
import pysensing.acoustic.preprocessing.transform as transform
import matplotlib.gridspec as gridspec

######################################################################
# Load the audio
# -----------------------------------

# First, the example audio is loaded

# Define the plot function
def plot_wave_and_spec(waveform,sample_rate):
    specgram = transform.spectrogram()(waveform)
    specgram = transform.amplitude2db()(specgram)
    n_fft = waveform.size(-1)
    freqs = torch.linspace(0, sample_rate / 2, int(n_fft / 2) + 1)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(specgram[0].squeeze().numpy(), aspect='auto', origin='lower', cmap='inferno', extent=[0, waveform.size(1) / sample_rate, 0, sample_rate / 2])
    plt.title('Spectrogram')
    plt.ylim(0,5000)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.subplot(2, 1, 2)
    plt.plot(waveform.t().numpy())
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

# Load the data
waveform, sample_rate = torchaudio.load('example_data/example_audio.wav')
#plot the original audio data and spectrogram
plot_wave_and_spec(waveform,sample_rate)
######################################################################
# 1. High pass filter
# ------------------------
# Use high pass filter

highpass_trans = filtering.highpass(sample_rate=44100,cutoff_freq=2000.0)
highpass       = highpass_trans(waveform)
plot_wave_and_spec(highpass,sample_rate)

######################################################################
# 2. Low pass filter
# ------------------------
# Use low pass filter

lowpass_trans = filtering.lowpass(sample_rate=44100,cutoff_freq=200.0)
lowpass       = lowpass_trans(waveform)
plot_wave_and_spec(lowpass,sample_rate)
######################################################################
# 3. Bandpass filter
# ------------------------
# Use bandpass filter

bandpass_trans = filtering.bandpass(sample_rate=44100,central_freq=1000)
bandpass       = bandpass_trans(waveform)
plot_wave_and_spec(bandpass,sample_rate)

######################################################################
# 4. Bandreject filter
# ------------------------
# Use bandreject filter

bandreject_trans = filtering.bandreject(sample_rate=44100,central_freq=1000)
bandreject       = bandreject_trans(waveform)
plot_wave_and_spec(bandreject,sample_rate)

######################################################################
# 5. Allpass filter
# ------------------------
# Use allpass filter
allpass_trans = filtering.allpass(sample_rate=44100,central_freq=2000)
allpass     = allpass_trans(waveform)
plot_wave_and_spec(allpass,sample_rate)
######################################################################
# And that's it. We're done with our acoustic preprocessing tutorials. Thanks for reading.
