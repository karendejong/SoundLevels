import os

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
from scipy.integrate import simps
import pandas as pd
from scipy.signal import butter, lfilter
import re

# Read first wav file
filepath = "/media/a5541/LaCie: Broadband2018- De Jong/ZoopSeisWP4_approaches/20220506_Saskia/"
# Saskia6may: "/media/a5541/LaCie: Broadband2018- De Jong/ZoopSeisWP4_approaches/20220506_Saskia/"
# control: "/media/a5541/LaCie: Broadband2018- De Jong/ZoopSeisWP4_approaches/20220501_control/"
# WBAT1: "/media/a5541/LaCie/ZoopSeis_tokt/X2hydrophone_recordings/20220501_WBAT1/"
# WBAT2: "/media/a5541/LaCie/ZoopSeis_tokt/X2hydrophone_recordings/20220502_WBAT2/"
# WBAT3: "/media/a5541/LaCie/ZoopSeis_tokt/X2hydrophone_recordings/20220505_WBAT3/"
# Austevol: "/media/a5541/3CC1-9C5B/LucyX2/"
# Austevol: "/media/a5541/LaCie/ZoopSeis_Austevoll/LucyX2/"
# Close pass: "/media/a5541/LaCie/ZoopSeis_tokt/X2hydrophone_recordings/20220504_ClosePass"
# Emilie: "/media/a5541/LaCie: Broadband2018- De Jong/ZoopSeisWP1_Airgunrecs/
first_file = "SBW1279_20220506_094500.wav"
# Saskia6may: "SBW1279_20220506_094500.wav"
# Control "SBW1279_20220501_082441.wav"
# WBAT1: "SBW1279_20220501_132100.wav", previously "SBW1279_20220501_131000.wav", but that included noise
# WBAT2: "SBW1279_20220502_092000.wav"
# WBAT3: "SBW1279_20220505_114100.wav"
# Austevol: "SBW1770_20210219_081900.wav"
# Close pass: "SBW1279_20220504_153909.wav"
# Emilie: "SBW1770_20210219_081900.wav"

file2read1 = os.path.join(filepath, first_file)
samples1, sample_rate = sf.read(file2read1)
sound1 = samples1.astype(float)
HD = re.split("[_]", first_file)[0]
date = re.split("[_]", first_file)[1]
print("Sample rate {}| Audio size {}".format(sample_rate, samples1.shape))

# Saskia6may:
filetimes9 = list(range(94800, 96000, 100))  # Times to analyse: start, stop, length file, NB 0759, 0800
filetimes10 = list(range(100000, 104000, 100))  # Times to analyse: start, stop, length file, NB 0759, 0800
filetimes = list(filetimes9+filetimes10)




# Austevol:
filetimes7 = list(range(73000, 76000, 100))  # Times to analyse: start, stop, length file, NB 0759, 0800
filetimes8 = list(range(80000, 83000, 100))  # Times to analyse: start, stop, length file, NB 0759, 0800
filetimes = list(filetimes7+filetimes8)
filetimes = list(range(81800,82000,100)) ##closest pass

# # WBAT1
filetimes13 = list(range(132100, 136000, 100)) # list of filenames to analyse 13:21-14:00
filetimes14 = list(range(140000, 143100, 100)) # list of filenames to analyse 14:00-14:31
filetimes = list(filetimes14)

# # WBAT2
filetimes09 = list(range(92000, 96000, 100)) # list of filenames to analyse 09:20-10:00 (incl 2390s)
filetimes10 = list(range(100000, 106000, 100)) # list of filenames to analyse 10:00-11:00
filetimes11 = list(range(110000, 116000, 100)) # list of filenames to analyse 10:00-12:00
filetimes12 = list(range(120000, 120600, 100)) # list of filenames to analyse 11:00-12:06
filetimes = list(filetimes12)  # 1 hour per round (filetimes09 filetimes10 filetimes11 filetimes12)

# # WBAT3
filetimes11 = list(range(114100, 116000, 100)) # list of filenames to analyse 11:41-12:00
filetimes12 = list(range(120000, 123500, 100)) # list of filenames to analyse 12:00-12:36
filetimes = list(filetimes11+filetimes12)

# control
filetimes = list(range(82500, 85300, 100)) # list of filenames to analyse 08:25 - 08:54


# # Close pass
# filetimes15 = list(range(154700, 156000, 100))
# filetimes16 = list(range(160000, 163000, 100))
# filetimes = list(filetimes15+filetimes16)




print(filetimes)


# Fill in calibration values for Ocean Sonics X2 hydrophone (Lucy+)
V_ppk = 6  # max volt recorded (from comments)*2
N = 1
OCVR = 205  # sensitivity hydrophone (from comments)


# filter functions
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Run for-loop to load sound
# for one file: sound = sound1
del sound

for i in filetimes:
    if i < 100000:
        filename = f"{HD}_{date}_0{i}.wav"  # Adapt this to file name format (now it will add 0 before {i} before 10:00)
    else:
        filename = f"{HD}_{date}_{i}.wav"
    file2read = os.path.join(filepath, filename)
    samples, sample_rate = sf.read(file2read)
    if i == filetimes[0]:
        sound = samples.astype(float)
        first_filename = re.split("[.]", filename)[-2]
    else:
        sound = np.append(sound, samples.astype(float))
    print(f"{filename} successfully uploaded")
    print("Sample rate {}| Audio size {}| Audio length {}s".format(sample_rate, sound.shape, len(sound)/sample_rate))

#check first_filename, because it will be used for naming file
# filename = f"{HD}_{date}_0{filetimes}.wav"
# first_filename = re.split("[.]", filename)[-2]
print(first_filename)

# Now apply a filter (re-run from here for different filter)
## if needed
del filteredsound
del calibratedsound

del ztpSPL_all
del SEL1s_all
del SEL_all
del SPL_all


# broadband highpass 10Hz - run this one first, because it needs a lot of space
filtername = "HP10Hz" # will be used to name files
filteredsound = butter_highpass_filter(sound, 10, fs=sample_rate, order=3)  # highpass filter 10Hz

# OR: 125Hz center freq
filtername = "125Hz"  # will be used to name files
filteredsound = butter_bandpass_filter(sound, 111.4, 176.8, fs=sample_rate, order=2)

# Calibration assuming fs.read normalises
calibratedsound = filteredsound*V_ppk/(2**N)*10**-6*(1/10**(-OCVR/20))  # Multiply raw data with calibration


# Decide on time steps (re-run from here for more time-steps)

length_sound = int(len(calibratedsound)/sample_rate)
time_step = 10  # Size of time step in seconds
timeinsecs = list(range(0, length_sound, time_step))  # the last nr decides the time steps (1s or 10s)
print(timeinsecs)

# Create empty arrays
SPL_all = np.empty(0)  # Create empty array for SPL values
ztpSPL_all = np.empty(0)  # Create empty array for 0 to peak SPL values
ptpP_all = np.empty(0)  # Create empty array for peak to peak pressure values in Pascal
SEL_all = np.empty(0)  # Create empty array for SEL values
SEL1s_all = np.empty(0)  # Create empty array for SEL values

# Calculate and append values
for j in timeinsecs:
    # select blast if you know where they are
    # start = int((j + 2.5) * sample_rate)  # this selects 1s!
    # stop = int((j + 3.5) * sample_rate)
    # chunk1s = calibratedsound[start:stop]
    # select 10s chunk
    chunk = calibratedsound[j * sample_rate:(j + time_step) * sample_rate]
    # calculate SPL
    # rms = np.sqrt(np.mean(chunk ** 2))
    # SPL = 20 * np.log10(rms / 10 ** -6)
    # SPL_all = np.append(SPL_all, SPL)2390
    # calculate and append 0 to peak SPL
    ztpSPL = 20 * np.log10((max(chunk) / 10 ** -6))
    ztpSPL_all = np.append(ztpSPL_all, ztpSPL)
    # calculate and append peak to peak pressure
    ptpP = max(chunk) - min(chunk)  # pressure in Pa
    ptpP_all = np.append(ptpP_all, ptpP)
    # calculate and append SEL
    I = sum(chunk ** 2) / sample_rate
    SEL = 10 * np.log10(I / 10 ** -12)
    SEL_all = np.append(SEL_all, SEL)
    # calculate and append SEL for 1s
    # I1s = sum(chunk1s ** 2) / sample_rate
    # SEL1s = 10 * np.log10(I / 10 ** -12)
    # SEL1s_all = np.append(SEL1s_all, SEL1s)
    print(f"{(j+time_step)} seconds of {length_sound} analysed")


# Check length array and add times NB! Adapt to files!
times = timeinsecs ## for Austevoll data

## for Ekofisk:
import datetime
starttime = datetime.datetime(2022,5,6,9,48,00) ## NB write in start time. WBAT3: 5/5/2022 11:41:00, WBAT2: 2/5/2022 09:20:00, WBAT1: 1/5/2022 13:21:00
times = [datetime.datetime(2022,1,1,0,0,0)]*len(timeinsecs) ## insert random time to create variable

for i in timeinsecs:
    delta = datetime.timedelta(seconds=i)  # NB add seconds for more files
    if i == 0:
        j = int(0)
    else:
        j = int(i/10)
    times[j] = starttime+delta

if len(SEL_all) == len(times):
    print("lengths OK")
    print(times[0])
else:
    print("something wrong here")
    print(times)




## filtername = "All"

# Dump arrays in CSV
# pd.DataFrame(SPL_all, times).to_csv(
#     f"/homfirst_e/a5541/PycharmProjects/pythonProject/Data/SPLpp{time_step}s_{first_filename}_{filtername}Hz.csv")
pd.DataFrame(list(zip(times, timeinsecs, ztpSPL_all)), columns=['Time', 'sfromstart', 'Level']).to_csv(
    f"/home/a5541/PycharmProjects/pythonProject/Data/ztpSPLp{time_step}s_{first_filename}_{filtername}.csv")
pd.DataFrame(list(zip(times, timeinsecs, ptpP_all)), columns=['Time', 'sfromstart', 'Level']).to_csv(
    f"/home/a5541/PycharmProjects/pythonProject/Data/ptpPp{time_step}s_{first_filename}_{filtername}.csv")
pd.DataFrame(list(zip(times, timeinsecs, SEL_all)), columns=['Time', 'sfromstart', 'Level']).to_csv(
    f"/home/a5541/PycharmProjects/pythonProject/Data/SELp{time_step}s_{first_filename}_{filtername}.csv")
# pd.DataFrame(SEL1s_all, times).to_csv(
#     f"/home/a5541/PycharmProjects/pythonProject/Data/SEL1sp{time_step}s_{first_filename}_{filtername}Hz.csv")


#read CSV
SPL_all = pd.read_csv("/home/a5541/PycharmProjects/pythonProject/Data/SPLp10s.csv")
SPL_all_float = SPL_all.astype(float)
ztpSPL_all = pd.read_csv("/home/a5541/PycharmProjects/pythonProject/Data/ztpSPLp10s.csv")
ptpSPL_all = pd.read_csv("/home/a5541/PycharmProjects/pythonProject/Data/ptpSPLp10s.csv")
SEL_all = pd.read_csv("/home/a5541/PycharmProjects/pythonProject/Data/SELp10s.csv")
SEL1s_all = pd.read_csv("/home/a5541/PycharmProjects/pythonProject/Data/SELp1s.csv")
# Make figures

# if testing
# plt.figure(1)
# plt.title("Waveform")
# plt.plot(calibratedsound[(0*sample_rate):(0+10)*sample_rate])
# plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/test.jpg")


# Make figures

#plt.figure(6)
#plt.title("SPL in dB re 1 µPa")
#plt.plot(times, SPL_all)
#plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/SPLp1s.jpg")

plt.figure(11)
plt.title("Zero to peak SPL in dB re 1 µPa")
plt.plot(times, ztpSPL_all)
plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/ztpSPLp10s.jpg")

plt.figure(8)
plt.title("Peak to peak pressure in Pascal")
plt.plot(times, ptpP_all)
plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/ptpP10s_zoom.jpg")

plt.figure(12)
plt.title("SEL in dB re 1 µPa² s")
plt.scatter(times, SEL_all)
plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/SELp10s.jpg")

## plot waveform
sample_nr = [range(0,61440000,1)]
sample_axis = np.array(sample_nr, dtype="float")
timeaxis = np.array(sample_nr, dtype="float")
for i in sample_nr:
    timeaxis[0,i] = sample_axis[0,i]/512000*1000000
timeaxis[0,sample_rate]

import datetime
starttime = datetime.datetime(2021,2,19,8,18,0,0)# NB write in start time. WBAT3: 5/5/2022 11:41:00, WBAT2: 2/5/2022 09:20:00, WBAT1: 1/5/2022 13:21:00
del times
times = [datetime.datetime(2021,1,1,0,0,0,0)]*timeaxis.size ## insert random time to create variable

for i in range(61440000):
    delta_ms = timeaxis[0,i]
    delta_ms_round = round(delta_ms)
    delta = datetime.timedelta(microseconds=delta_ms_round)
    times[i] = starttime+delta


plt.figure(14)
plt.title("calibrated wave form")
plt.plot(times,calibratedsound)
plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/calibbratedwaveformtimes.jpg")
