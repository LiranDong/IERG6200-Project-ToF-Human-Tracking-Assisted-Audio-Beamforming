import torch
import torchaudio
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy import signal as sg
import matplotlib.pyplot as plt

sample_file = 'final-tof'
direction = 'h'
enhanced = False

sigma_seconds = 1.2
FFT_LENGTH = 1024
CUTTING_SECONDS = 0.5
SOUND_SPEED = 340 * 1000
lowcut, highcut = 200, 4500
mic_choice = [0, 1, 2, 3, 4, 5, 6, 7]
tof_fps = 30

input_file = f'files/{sample_file}.wav'
output_file = f'output-{sample_file}-{direction}-dlr.wav'

traj_xkc = np.load('3_new_Trajectory_xkc.npy')[:, 1:3]
traj_dlr = np.load('3_new_Trajectory_dlr.npy')[:, 1:3]

traj_start_frame = 0
traj_end_frame = 600
direction_mapping = {'h': 0, 'v': 1}


def array_pos():
    # lse
    pos = torch.tensor(
        [[-61.55, -26.93, 0.0],
         [-13.74, -40.87, 0.0],
         [34.49, -54.0, 0.0],
         [59.49, 26.71, 0.0],
         [11.79, 40.74, 0.0],
         [-36.35, 53.64, 0.0],
         [-25.62, 6.92, 0.0],
         [23.58, -7.18, 0.0]],
        dtype=src_audio.dtype, device=src_audio.device)[mic_choice]
    return pos


def enframe(wav_data, frame_length):
    shift = frame_length // 2
    len_sample, len_channel_vec = np.shape(wav_data)
    number_of_frame = int((len_sample - frame_length) / shift)
    frame_split = torch.stack([wav_data[(shift*i):(shift*i+frame_length)] for i in range(number_of_frame)], dim=0)
    frame_split *= sg.windows.hann(frame_length, False)[..., None]
    return frame_split, number_of_frame


def deframe(data, beta=0.5):
    shift = data.size(1) // 2
    dst_data = torch.zeros(data.size(0) * shift)
    dst_data[:shift] = data[0, :shift]
    for i in range(1, data.size(0)):
        dst_data[i * shift:(i + 1) * shift] = beta * data[i - 1, shift:] + (1 - beta) * data[i, :shift]
    return dst_data


src_audio, sample_rate = sf.read(input_file, dtype='float64')
src_audio = torch.from_numpy(src_audio)
src_audio = src_audio[int(CUTTING_SECONDS * sample_rate):-sample_rate, :]
src_audio = src_audio[:, mic_choice]
sample_length = src_audio.size(0)

if enhanced:
    src_audio = src_audio.numpy(force=True)
    sf.write(f'{sample_file}-enhanced.wav', src_audio / np.max(np.abs(src_audio)), sample_rate)
    exit()

src_audio, n_frames = enframe(src_audio, FFT_LENGTH)
channels = src_audio.size(-1)

# mean power equalization
# src_audio /= torch.mean(src_audio ** 2, dim=0, keepdim=True).sqrt()
# src_audio /= torch.max(src_audio.abs(), dim=0, keepdim=True).values
# src_audio /= torch.quantile(src_audio.abs(), 0.9, dim=0, keepdim=True)

freq_samples = torch.arange(FFT_LENGTH, dtype=src_audio.dtype, device=src_audio.device) / FFT_LENGTH
freq_samples -= torch.heaviside(freq_samples - 0.5, torch.tensor([0], dtype=src_audio.dtype, device=src_audio.device))
freq_samples *= sample_rate

tof_time = np.linspace(traj_start_frame / tof_fps, traj_end_frame / tof_fps, num=np.size(traj_dlr, axis=0))
aud_start_frame = int((traj_start_frame / tof_fps) / (FFT_LENGTH // 2 / sample_rate))

source_vector_dlr = np.zeros([n_frames, 3])
source_vector_xkc = np.zeros([n_frames, 3])

frame_time = np.arange(1, n_frames + 1) * (FFT_LENGTH // 2 / sample_rate)
source_vector_dlr[aud_start_frame:n_frames, 0] = np.interp(x=frame_time, xp=tof_time, fp=traj_dlr[:, 0])
source_vector_dlr[aud_start_frame:n_frames, 1] = np.interp(x=frame_time, xp=tof_time, fp=traj_dlr[:, 1])

source_vector_xkc[aud_start_frame:n_frames, 0] = np.interp(x=frame_time, xp=tof_time, fp=traj_xkc[:, 0])
source_vector_xkc[aud_start_frame:n_frames, 1] = np.interp(x=frame_time, xp=tof_time, fp=traj_xkc[:, 1])

fft_frame = torch.fft.fft(src_audio, dim=1)

sigma = sample_rate / FFT_LENGTH * sigma_seconds
sigma_range = 4
alpha = torch.exp(-torch.linspace(-sigma_range, sigma_range, 1 + int(2 * sigma_range * sigma), dtype=src_audio.dtype) ** 2) * (1 + 0j)
r = torch.einsum('...fi, ...fj -> ...fij', fft_frame, fft_frame.conj())
r = torchaudio.functional.convolve(r.permute(1, 2, 3, 0), alpha[None, None, None, :], mode='same').permute(3, 0, 1, 2)

dst_audio = torch.zeros([n_frames, FFT_LENGTH])
dst_audio[0:aud_start_frame, :] = src_audio[0:aud_start_frame, :, 0]

mic_pos = array_pos()
# f = torch.tensor([1, ], dtype=src_audio.dtype, device=src_audio.device)
f = torch.tensor([1, ] + [0, ], dtype=src_audio.dtype, device=src_audio.device)

for fr in range(aud_start_frame, n_frames):
    source_vector = torch.tensor([
        [source_vector_dlr[fr, 0], source_vector_dlr[fr, 1], 0],
        [source_vector_xkc[fr, 0], source_vector_xkc[fr, 1], 0]
    ])
    tag_vector = source_vector[direction_mapping[direction]]
    inf_vector = source_vector[[i for i in range(len(source_vector)) if i != direction_mapping[direction]]]

    tag_pos = tag_vector.to(dtype=src_audio.dtype, device=src_audio.device)
    inf_pos = inf_vector.to(dtype=src_audio.dtype, device=src_audio.device)

    tag_dis = torch.norm(tag_pos - mic_pos, dim=-1)
    tag_dis_diff = tag_dis - torch.mean(tag_dis, dim=-1, keepdim=True)
    tag_dis_gain = 1 / tag_dis ** 2
    tag_dis_gain /= torch.max(tag_dis_gain)
    tag_guide = tag_dis_gain * torch.exp(- 2 * torch.pi * 1j * freq_samples[:, None] * tag_dis_diff / SOUND_SPEED)

    inf_dis = torch.norm(inf_pos[:, None, :] - mic_pos, dim=-1)
    inf_dis_diff = inf_dis - torch.mean(inf_dis, dim=-1, keepdim=True)
    inf_dis_gain = 1 / inf_dis ** 2
    if inf_dis_gain.numel() > 0:
        inf_dis_gain /= torch.max(inf_dis_gain)
    inf_guide = inf_dis_gain * torch.exp(- 2 * torch.pi * 1j * freq_samples[:, None, None] * inf_dis_diff / SOUND_SPEED)

    # C = tag_guide[:, None, :].transpose(-1, -2)
    # inv_R_C = torch.linalg.solve(r, C)
    # ct_inv = torch.einsum('...ki, ...kj -> ...ij', C.conj(), inv_R_C)
    # w = torch.einsum('...ij, ...j -> ...i', inv_R_C, torch.linalg.solve(ct_inv, f * (1 + 0j)))
    # fft_frame = torch.einsum('...i, ...i -> ...', w.conj(), fft_frame)

    C = torch.concat([tag_guide[:, None, :], inf_guide], dim=-2).transpose(-1, -2)
    inv_R_C = torch.linalg.solve(r[fr, :, :, :], C)
    ct_inv = torch.einsum('fki, fkj -> fij', C.conj(), inv_R_C)
    w = (inv_R_C @ torch.linalg.solve(ct_inv, f * (1 + 0j))[..., None]).squeeze(dim=-1)
    fft_frame_curr = torch.einsum('fi, fi -> f', w.conj(), fft_frame[fr, :, :])
    dst_audio[fr, :] = torch.fft.ifft(fft_frame_curr).real

dst_audio = deframe(dst_audio).numpy(force=True)
b, a = sg.butter(5, [lowcut * 2 / sample_rate, highcut * 2 / sample_rate], btype='bandpass')
dst_audio = sg.lfilter(b, a, dst_audio)
dst_audio = nr.reduce_noise(y=dst_audio, sr=sample_rate, prop_decrease=0.89)
dst_audio /= np.sqrt(np.mean(dst_audio ** 2))
sf.write(output_file, dst_audio / np.max(np.abs(dst_audio)), sample_rate)
