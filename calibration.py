# import packages
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
from scipy.integrate import simps
from scipy.signal import butter, lfilter

# Read wav file

filepath = "/media/a5541/LaCie/ZoopSeis_tokt/X2hydrophone_recordings/20220501_WBAT1" ## "/media/a5541/LaCie: Broadband2018- De Jong/ZoopSeisWP1_Airgunrecs/
filename = "SBW1279_20220501_143000.wav" ##"SBW1770_20210219_081900.wav"
file2read = os.path.join(filepath, filename)

samples, sample_rate = sf.read(file2read)
samples = samples.astype(float)
print("Sample rate {}| Audio size {}| Audio length {}s".format(sample_rate, samples.shape, len(samples) / sample_rate))

# Select 10s

tstart = 0*sample_rate
tend = 10*sample_rate

first10s = samples[int(tstart):int(tend)]
time = np.linspace(0, len(first10s)/sample_rate, num=len(first10s))
plt.figure(1)
plt.title("Wave form")
plt.plot(time, first10s)
plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/ex_first10s.jpg")


# Select one blast

tstart = 7*sample_rate
tend = 8*sample_rate

blast = samples[int(tstart):int(tend)]
time = np.linspace(0, len(blast)/sample_rate, num=len(blast))


plt.figure(2)
plt.title("Wave form")
plt.plot(time, blast)
plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/ex_waveform.jpg")

# plt.show(block=True)
# plt.interactive(False)

# Calibration the whole file assuming fs.read normalises

V_ppk = 6  # max volt recorded (from comments)*2
N = 1
OCVR = 205  # sensitivity hydrophone (from comments)

samples_cal = samples*V_ppk/(2**N)*(10e-6)*(1/10**(-OCVR/20))  # Multiply raw data with calibration


##Select same blast

blast_cal = samples_cal[int(tstart):int(tend)]
time = np.linspace(0, len(blast_cal)/sample_rate, num=len(blast_cal))


plt.figure(3)
plt.title("Wave form")
plt.plot(time,blast_cal)
plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/ex_calibratedwaveform.jpg")

# Does this give reasonable numbers?

rms = np.sqrt(np.mean(blast_cal ** 2))
SPL = 20 * np.log10(rms / 10 ** -6)
ptpSPL = 20*np.log10((max(blast_cal)-min(blast_cal))/10**-6)
I = sum(blast_cal ** 2) / sample_rate
##I = simps(abs(blast_cal),dx=1/sample_rate) ##Simpson method, is this right? ##no
SEL = 10*np.log10(I/10**-12)
print(SPL,ptpSPL,SEL)

# Now apply a high-pass filter to calibrated signal

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

fsamples_cal = butter_highpass_filter(samples_cal,10,fs=sample_rate,order=3)

print("Sample rate {}| Audio size {}".format(sample_rate,fsamples_cal.shape))
print(fsamples_cal)

# Select same blast

fblast_cal = fsamples_cal[int(tstart):int(tend)] ##select 21-25s
ftime = np.linspace(0, len(fblast_cal)/ sample_rate, num=len(fblast_cal))

plt.figure(4)
plt.title("Wave form vs filtered wave form")
plt.plot(blast_cal)
plt.plot(fblast_cal)
plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/ex_filtcalwaveform.jpg")


# Now apply some band-pass filters to calibrated signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



fsamples_cal_low = butter_bandpass_filter(samples_cal, 10, 55.7, fs=sample_rate, order=2)
fsamples_cal_63 = butter_bandpass_filter(samples_cal, 55.7, 70.2, fs=sample_rate, order=2)
fsamples_cal_80 = butter_bandpass_filter(samples_cal, 70.2, 88.4, fs=sample_rate, order=2)
fsamples_cal_100 = butter_bandpass_filter(samples_cal, 88.4, 111.4, fs=sample_rate, order=2)
fsamples_cal_125 = butter_bandpass_filter(samples_cal, 111.4, 176.8, fs=sample_rate, order=2)
fsamples_cal_250 = butter_bandpass_filter(samples_cal, 176.8, 353.6, fs=sample_rate, order=2)
fsamples_cal_500 = butter_bandpass_filter(samples_cal, 353.6, 707.1, fs=sample_rate, order=3)
fsamples_cal_1000 = butter_bandpass_filter(samples_cal, 707.1, 1414.2, fs=sample_rate, order=3)
fsamples_cal_2000 = butter_bandpass_filter(samples_cal, 1414.2, 2828.4, fs=sample_rate, order=4)
fsamples_cal_4000 = butter_bandpass_filter(samples_cal, 2828.4, 5656.9, fs=sample_rate, order=4)
fsamples_cal_8000 = butter_bandpass_filter(samples_cal, 5656.9, 11313.7, fs=sample_rate, order=5)
fsamples_cal_16000 = butter_bandpass_filter(samples_cal, 11313.7, 22627.4, fs=sample_rate, order=5)



# Select same blast to check filters

fblast_low = fsamples_cal_low[int(tstart):int(tend)]
fblast_500 = fsamples_cal_500[int(tstart):int(tend)] ##select 21-25s
fblast_1000 = fsamples_cal_1000[int(tstart):int(tend)] ##select 21-25s


plt.figure(4)
plt.title("Wave form vs filtered wave form")
plt.plot(blast_cal)
plt.plot(fblast_low)
plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/ex_filtcalwaveform.jpg")


# Compare numbers in DB

rmsf = np.sqrt(np.mean(fblast_cal))
SPLf = 20*np.log10(rmsf/10**-6)
ptpSPLf = 20*np.log10((max(fblast_cal)-min(fblast_cal))/10**-6)
dt = len(fblast_cal)/sample_rate
If = sum(fblast_cal**2)/sample_rate
SELf = 10*np.log10(If/10**-12)
print(SPLf,ptpSPLf,SELf)
print(SPL,ptpSPL,SEL)


##Calculate values per ten second bins
#choose time steps
x = list(range(0, 60, 10))  # start, length file in seconds, time-steps to analyse (this gives 0,10,20,30,40,50)

# Create empty array for those values you want to analyse
column_names = ["ptpSPL_all", "SEL_all", "SEL_low_all", "SEL_63_all", "SEL_125", "SEL_500", "SEL_1000", "SEL_8000"]
analyses = pd.DataFrame(columns = column_names)
print(analyses.ptpSPL_all)



# Calculate and append values
for j in x:
        # select 10s from each of the chosen filters
        chunk = fsamples_cal[j * sample_rate:(j + 10) * sample_rate]
        chunk_low = fsamples_cal_low[j * sample_rate:(j + 10) * sample_rate]
        chunk_63 = fsamples_cal_63[j * sample_rate:(j + 10) * sample_rate]
        chunk_125 = fsamples_cal_125[j * sample_rate:(j + 10) * sample_rate]
        chunk_500 = fsamples_cal_500[j * sample_rate:(j + 10) * sample_rate]
        chunk_1000 = fsamples_cal_1000[j * sample_rate:(j + 10) * sample_rate]
        chunk_8000 = fsamples_cal_8000[j * sample_rate:(j + 10) * sample_rate]

        # calculate and append peak to peak SPL
        ptpSPL = 20 * np.log10((max(chunk) - min(chunk)) / 10 ** -6)
        # calculate and append SELs
        SEL = 10 * np.log10((sum(chunk ** 2) / sample_rate) / 10 ** -12)
        SEL_low = 10 * np.log10((sum(chunk_low ** 2) / sample_rate) / 10 ** -12)
        SEL_63 = 10 * np.log10((sum(chunk_63 ** 2) / sample_rate) / 10 ** -12)
        SEL_125 = 10 * np.log10((sum(chunk_125 ** 2) / sample_rate) / 10 ** -12)
        SEL_500 = 10 * np.log10((sum(chunk_500 ** 2) / sample_rate) / 10 ** -12)
        SEL_1000 = 10 * np.log10((sum(chunk_1000 ** 2) / sample_rate) / 10 ** -12)
        SEL_8000 = 10 * np.log10((sum(chunk_8000 ** 2) / sample_rate) / 10 ** -12)
        # append values to dataframe
        analyses = np.append(analyses, np.array([[ptpSPL, SEL, SEL_low, SEL_63, SEL_125, SEL_500, SEL_1000, SEL_8000]]), axis=0)

analyses = pd.DataFrame(analyses)
analyses[1:3]



plt.figure(5)
plt.title("Peak-to-peak and 1000Hz third-octave band")
plt.plot(analyses[:0])
plt.plot(analyses[:5])
plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/analyses_1file.jpg")



## Absolute values just for fun
Time1=int(22.875*sample_rate)
Time2=23*sample_rate

blastZ = samples_cal[Time1:Time2] ##select 23-23.5
timeZ = np.linspace(0, len(blastZ)/ sample_rate, num=len(blastZ))

fblastZ = fsamples_cal[Time1:Time2] ##select 21-25s
ftimeZ = np.linspace(0, len(fblastZ)/ sample_rate, num=len(fblastZ))

plt.figure(6)
plt.title("Absolute wave forms")
plt.plot(timeZ,abs(blastZ))
plt.plot(ftimeZ,abs(fblastZ))
plt.savefig("/home/a5541/PycharmProjects/pythonProject/Figures/ex_absolutewaveforms.jpg")