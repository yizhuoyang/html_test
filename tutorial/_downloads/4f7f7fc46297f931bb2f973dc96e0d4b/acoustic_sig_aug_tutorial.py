"""
Acoustic Augmentation.Signal_aug Tutorial
==============================================================
"""

######################################################################
# !pip install pysensing

######################################################################
# In this tutorial, we will be implementing a simple acoustic.augmentation.sig_aug
# 
import torch
import torchaudio
import matplotlib.pyplot as plt
from pysensing.acoustic.augmentation import signal_aug
from pysensing.acoustic.preprocessing import transform
######################################################################
# Load the audio
# -----------------------------------
# First, the example audio is loaded

# Define the plot function
def plot(audio_data_list, titles,channel_first=True):
    num_audios = len(audio_data_list)
    fig, axes = plt.subplots(num_audios, 1, figsize=(12, 8))
    if num_audios == 1:
        axes = [axes]
    for ax, audio_data, title in zip(axes, audio_data_list, titles):
        if channel_first==False:
            ax.plot(audio_data.numpy())
        else:
            ax.plot(audio_data[0].numpy())
        ax.set_title(title)
    fig.tight_layout()
    plt.show()

# Load the data
# Load the data
waveform, sample_rate = torchaudio.load('example_data/example_audio.wav')

######################################################################
# 1. Add Noise
# ------------------------

noise = torch.randn_like(waveform)
add_noise_0 = signal_aug.add_noise(noise,torch.tensor([0]))
add_noise_5 = signal_aug.add_noise(noise,torch.tensor([20]))

noise_data_0 = add_noise_0(waveform)
noise_data_5 = add_noise_5(waveform)

plot([waveform,noise_data_5,noise_data_0],['Original','SNR=20','SNR=0'])

######################################################################
# 2. Add echo
# ------------------------
waveform, sample_rate = torchaudio.load('example_data/example_audio.wav',channels_first=False)
add_echo_tran = signal_aug.add_echo(sample_rate,in_gain=0.6,out_gain=0.3,delays=[1000],decays=[0.5])
echo_data = add_echo_tran(waveform)

plot([waveform,echo_data],['Original','Add_echo'],False)

######################################################################
# 3. Add atempo
# ------------------------

add_echo_tran = signal_aug.add_echo(sample_rate,in_gain=0.6,out_gain=0.3,delays=[1000],decays=[0.5])
echo_data = add_echo_tran(waveform)

plot([waveform,echo_data],['Original','Add_echo'],False)

######################################################################
# 4. Add chorus
# ------------------------

add_chorus_trans = signal_aug.add_chorus(sample_rate,in_gain=0.6,out_gain=0.3,delays=[1000],decays=[0.5],speeds=[0.25],depths=[2.0])
chorus_data        = add_chorus_trans(waveform)
plot([waveform,chorus_data],['Original','Add_chorus'],False)

######################################################################
# And that's it. We're done with our acoustic augmentation.signal_aug tutorials. Thanks for reading.
