# Mini project
_authors_: Giulio Benedetti and Cristian Garcia

## Data Exploration

```python
import numpy as np
import pandas as pd

df1 = pd.read_csv("2023-11-06_09_24_37_Apple Watch.csv")

df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loggingTime(txt)</th>
      <th>locationTimestamp_since1970(s)</th>
      <th>locationLatitude(WGS84)</th>
      <th>locationLongitude(WGS84)</th>
      <th>locationAltitude(m)</th>
      <th>locationSpeed(m/s)</th>
      <th>locationSpeedAccuracy(m/s)</th>
      <th>locationCourse(°)</th>
      <th>locationCourseAccuracy(°)</th>
      <th>locationVerticalAccuracy(m)</th>
      <th>...</th>
      <th>pedometerDistance(m)</th>
      <th>pedometerFloorAscended(N)</th>
      <th>pedometerFloorDescended(N)</th>
      <th>pedometerEndDate(txt)</th>
      <th>altimeterTimestamp_sinceReboot(s)</th>
      <th>altimeterReset(bool)</th>
      <th>altimeterRelativeAltitude(m)</th>
      <th>altimeterPressure(kPa)</th>
      <th>batteryState(N)</th>
      <th>batteryLevel(R)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-11-06T09:24:37.756+02:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>7.403943</td>
      <td>0</td>
      <td>0</td>
      <td>2023-11-06T09:23:25.206+02:00</td>
      <td>7.209483e+08</td>
      <td>0</td>
      <td>-0.98999</td>
      <td>101.7212</td>
      <td>1</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-11-06T09:24:37.762+02:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>7.403943</td>
      <td>0</td>
      <td>0</td>
      <td>2023-11-06T09:23:25.206+02:00</td>
      <td>7.209483e+08</td>
      <td>0</td>
      <td>-0.98999</td>
      <td>101.7212</td>
      <td>1</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-11-06T09:24:37.776+02:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>7.403943</td>
      <td>0</td>
      <td>0</td>
      <td>2023-11-06T09:23:25.206+02:00</td>
      <td>7.209483e+08</td>
      <td>0</td>
      <td>-0.98999</td>
      <td>101.7212</td>
      <td>1</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-11-06T09:24:37.790+02:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>7.403943</td>
      <td>0</td>
      <td>0</td>
      <td>2023-11-06T09:23:25.206+02:00</td>
      <td>7.209483e+08</td>
      <td>0</td>
      <td>-0.98999</td>
      <td>101.7212</td>
      <td>1</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-11-06T09:24:37.809+02:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>7.403943</td>
      <td>0</td>
      <td>0</td>
      <td>2023-11-06T09:23:25.206+02:00</td>
      <td>7.209483e+08</td>
      <td>0</td>
      <td>-0.98999</td>
      <td>101.7212</td>
      <td>1</td>
      <td>0.7</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>



**Question A**: Remind yourself what sensors are provided, what they record and what they can be informative about.

Among the 58 variables, the monitored information icludes:
- logging and timestamp `[0, 1, 12, 16, 52]`
- 3D location and corresponding accuracy `[1:12]`
- 3D acceleration `[12:16]`
- 3D motion (e.g. rotation and gravity) `[16:39]`
- pedometer and activity type `[39:52]`

```python
df1.columns
```




    Index(['loggingTime(txt)', 'locationTimestamp_since1970(s)',
           'locationLatitude(WGS84)', 'locationLongitude(WGS84)',
           'locationAltitude(m)', 'locationSpeed(m/s)',
           'locationSpeedAccuracy(m/s)', 'locationCourse(°)',
           'locationCourseAccuracy(°)', 'locationVerticalAccuracy(m)',
           'locationHorizontalAccuracy(m)', 'locationFloor(Z)',
           'accelerometerTimestamp_sinceReboot(s)',
           'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)',
           'accelerometerAccelerationZ(G)', 'motionTimestamp_sinceReboot(s)',
           'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)',
           'motionRotationRateX(rad/s)', 'motionRotationRateY(rad/s)',
           'motionRotationRateZ(rad/s)', 'motionUserAccelerationX(G)',
           'motionUserAccelerationY(G)', 'motionUserAccelerationZ(G)',
           'motionAttitudeReferenceFrame(txt)', 'motionQuaternionX(R)',
           'motionQuaternionY(R)', 'motionQuaternionZ(R)', 'motionQuaternionW(R)',
           'motionGravityX(G)', 'motionGravityY(G)', 'motionGravityZ(G)',
           'motionMagneticFieldX(µT)', 'motionMagneticFieldY(µT)',
           'motionMagneticFieldZ(µT)', 'motionHeading(°)',
           'motionMagneticFieldCalibrationAccuracy(Z)',
           'activityTimestamp_sinceReboot(s)', 'activity(txt)',
           'activityActivityConfidence(Z)', 'activityActivityStartDate(txt)',
           'pedometerStartDate(txt)', 'pedometerNumberofSteps(N)',
           'pedometerAverageActivePace(s/m)', 'pedometerCurrentPace(s/m)',
           'pedometerCurrentCadence(steps/s)', 'pedometerDistance(m)',
           'pedometerFloorAscended(N)', 'pedometerFloorDescended(N)',
           'pedometerEndDate(txt)', 'altimeterTimestamp_sinceReboot(s)',
           'altimeterReset(bool)', 'altimeterRelativeAltitude(m)',
           'altimeterPressure(kPa)', 'batteryState(N)', 'batteryLevel(R)'],
          dtype='object')



**Question B**: Look into the `accelerometerAcceleration` and `motionUserAcceleration` sensors. What do you think is the relationship between the two? Confirm your suspicion in code.

The two sets of variables appear to be highly correlated. Assumed that there is no lag in the measurement, this is expected, because the accelerometer should detect a greater acceleration as the user speeds up their motion.

```python
from matplotlib import pyplot as plt
import seaborn as sns

corrX = np.corrcoef(df1["accelerometerAccelerationX(G)"], df1["motionUserAccelerationX(G)"])[1, 0]
corrY = np.corrcoef(df1["accelerometerAccelerationY(G)"], df1["motionUserAccelerationY(G)"])[1, 0]
corrZ = np.corrcoef(df1["accelerometerAccelerationZ(G)"], df1["motionUserAccelerationZ(G)"])[1, 0]

fig, axes = plt.subplots(1, 3)

fig.set_figheight(5)
fig.set_figwidth(15)
sns.set_style("ticks")

sns.scatterplot(df1, x="accelerometerAccelerationX(G)", y="motionUserAccelerationX(G)", ax=axes[0])
sns.scatterplot(df1, x="accelerometerAccelerationY(G)", y="motionUserAccelerationY(G)", ax=axes[1])
sns.scatterplot(df1, x="accelerometerAccelerationZ(G)", y="motionUserAccelerationZ(G)", ax=axes[2])

axes[0].set_title(f"X, corr: {corrX:.3f}")
axes[1].set_title(f"Y, corr: {corrY:.3f}")
axes[2].set_title(f"Z, corr: {corrZ:.3f}")

plt.tight_layout()
```


    
![png](report_files/report_6_0.png)
    


**Question C**: What useful features could be extracted from the three axes of these recordings?

Because acceleration is defined as the change in velocity, these variables can be used to assess motion status and direction: motion is stable in a given direction when corresponding acceleration is zero, and it undergoes change when acceleration is non-zero.

**Question E**: One of the first things, which is important to know is the Frequency sampling (Fs). We are not provided it directly, but how can we approximate it given the data?

Mean sampling frequency was 50.25 Hz. The sampling intervals seemed to vary more at the beginnig and at the end of the measurement, when motion may be less constant. In the centre, sampling frequency became relatively stable at 50.25 Hz.

```python
import datetime as dt

sample_times = [dt.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f+02:00") for time in df1["loggingTime(txt)"]]
sample_diffs = np.array([time.microseconds for time in np.diff(sample_times)]) / 1e6
df1["Time"] = np.array([time.seconds + time.microseconds / 1e6 for time in np.array(sample_times) - sample_times[0]])
sample_freq = 1 / np.mean(sample_diffs)

print(f"Mean Interval: {np.mean(sample_diffs):.4f} s")
print(f"Standard deviation: {np.std(sample_diffs):.4f} s")
print(f"Minimum: {np.min(sample_diffs)} s, Maximum: {np.max(sample_diffs)} s")
print(f"Sampling frequency: {sample_freq:.2f} Hz")
```

    Mean Interval: 0.0199 s
    Standard deviation: 0.0014 s
    Minimum: 0.002 s, Maximum: 0.046 s
    Sampling frequency: 50.25 Hz


```python
fig, axes = plt.subplots(2, 2)

fig.set_figheight(6)
fig.set_figwidth(10)
sns.set_style("ticks")

sns.lineplot(x=df1.index[:-1], y=sample_diffs, ax=axes[0, 0])
axes[0, 0].set_xlabel("Sample Position")
axes[0, 0].set_ylabel("Sample Interval (s)")

sns.boxplot(y=sample_diffs, ax=axes[0, 1])
axes[0, 1].set_xlabel("Interval Distribution")

sns.lineplot(x=df1.index[:-1], y=1/sample_diffs, ax=axes[1, 0])
axes[1, 0].set_xlabel("Sample Position")
axes[1, 0].set_ylabel("Sample Frequency (Hz)")

sns.boxplot(y=1/sample_diffs, ax=axes[1, 1])
axes[1, 1].set_xlabel("Frequency Distribution")

plt.tight_layout()
```


    
![png](report_files/report_11_0.png)
    


**Qestion F**: Given the found Fs, how many seconds are we observing in each session?

As shown above, we see an average of 20 milliseconds per sample. In total, the measurement was about 36-second long.

```python
# Compute measurement length
mes_len = sample_times[-1] - sample_times[0]
mes_len = mes_len.seconds + mes_len.microseconds / 1e6

print(f"Total length of the measurement: {mes_len} s")
```

    Total length of the measurement: 35.602 s


**Qestion D**: Plot (in any helpful way for you) pieces of data to explore what it looks like and what can be guessed about it based on the visual inspection. What are some of the interesting recorded sensors?

```python
import re

def plot_combinations(df, ylist, ax):

    for y in ylist:
        ylab = re.sub("(motion|pedometer)(User)?(.+)\(.+\)", "\\3", y)
        sns.lineplot(df, x="Time", y=y, ax=ax, label=ylab)

    ax.legend()
    ax.set_ylabel(re.sub(".+\((.+)\)", "\\1", ylist[0]))

plot_num = 5
fig, axes = plt.subplots(plot_num, sharex=True)

fig.set_figheight(2.2 * plot_num)
fig.set_figwidth(6)
sns.set_style("ticks")

plot_combinations(df1, ["motionUserAccelerationX(G)", "motionUserAccelerationY(G)", "motionUserAccelerationZ(G)"], ax=axes[0])
plot_combinations(df1, ["motionGravityX(G)", "motionGravityY(G)", "motionGravityZ(G)"], ax=axes[1])
plot_combinations(df1, ["motionYaw(rad)", "motionRoll(rad)", "motionPitch(rad)"], ax=axes[2])
plot_combinations(df1, ["motionRotationRateX(rad/s)", "motionRotationRateY(rad/s)", "motionRotationRateZ(rad/s)"], ax=axes[3])
plot_combinations(df1, ["pedometerCurrentPace(s/m)", "pedometerAverageActivePace(s/m)"], ax=axes[4])

axes[4].set_xlabel("Time (s)")
plt.tight_layout()

```


    
![png](report_files/report_15_0.png)
    


## Data Analysis

**Question A**: After the initial exploration, what can you say about the structure of the data? On different time-scales, what do you think the different intervals represent?

**Question B**: Using the tools we have learned, can you find the frequency of the steps? How? Describe in words the meaning behind your calculations.

According to FFT, the step frequency is about 0.05 Hz. This is unlikely, because it would convert to a strikingly slow step cadence of about 1 step every 20 seconds. A mistake might be present either in the conversion to frequency domain by FFT, or in the choice of the base frequency from the FFT plot.

```python
# Find number of observations
signal_length = len(df1["Time"])
# Convert time scale to frequency scale
freq = np.fft.fftfreq(signal_length)[:signal_length // 2]

# Perform FFT for the signal in every coordinate
spx = np.fft.fft(df1["motionUserAccelerationX(G)"])[:signal_length // 2]
spy = np.fft.fft(df1["motionUserAccelerationY(G)"])[:signal_length // 2]
spz = np.fft.fft(df1["motionUserAccelerationY(G)"])[:signal_length // 2]

# Find base frequency in every coordinate
step_freqX = freq[np.where(spx == max(spx))]
step_freqY = freq[np.where(spy == max(spy))]
step_freqZ = freq[np.where(spz == max(spz))]

# Take average of base frequency over coordinates
step_freq = (step_freqX + step_freqY + step_freqZ) / 3
print(f"The step frequency is {step_freq[0]:.3f} Hz.")
```

    The step frequency is 0.056 Hz.


```python
fig, axes = plt.subplots(3, 1, sharex=True)

fig.set_figheight(8)
fig.set_figwidth(5)

axes[0].plot(freq, spx.real)
axes[0].set_ylabel("Magnitude")

axes[1].plot(freq, spy.real)
axes[1].set_ylabel("Magnitude")

axes[2].plot(freq, spz.real)
axes[2].set_ylabel("Magnitude")
axes[2].set_xlabel("Frequency (Hz)")

plt.tight_layout()
```


    
![png](report_files/report_21_0.png)
    


**Question C**: Using specgram or an alternative, plot and analyse the signals’ frequencies’ prevalence over time.

```python
from scipy import signal

f, t, Sxx = signal.spectrogram(df1["motionUserAccelerationX(G)"], fs=sample_freq)

plt.pcolormesh(t, f, Sxx, shading='gouraud')

plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
```




    Text(0.5, 0, 'Time (s)')




    
![png](report_files/report_23_1.png)
    


**Question D**: Are the sessions stationary?

We define a stationary random process as a set of signals produced over time, each of whose points are random variables drawn from the same distribution regardless of their position in time. Therefore, it follows that the decision of considering a signal stationary or not chiefly depends on the time range under inspection: shorter signals may better reflect stationary processes, whereas longer signals may appear more subject to periodicity.

With that said, it is safe to consider walking and running sessions as stationary random processes, when looked at individually. Here, based on `motionUserAcceleration` we can infer that the portions of signal from 0 to 25 s, and between 25 and 30 s well represent stationary random processes. However, the signal over its full length contains multiple sessions, and thus it is not stationary.

**Question E**: Are the sessions ergodic?

We define an ergodic process as a set of signals produced over time, whose points from the begin to the end of each signal are drawn from a distribution that is identical to that of any random variable from the signal. Whereas all ergodic processes are also random stationary, not all of the latter meet the conditions to be ergodic. For example, if the random variables from two signals are drawn from a single bimodal distribution, but each signal contains random variables from only one mode of that distribution, the process is stationary but not ergodic.

With that said, we could consider the signals ergodic only if each signal contained either walking or running sessions, but not both. Because the two are actually combined, the process is not ergodic. However, if we narrow down the signals to one activity type, the process can be considered ergodic. Here, based on `motionUserAcceleration` the signal upto 25 s and that from 30 s on are ergodic with respect to one another.

**Question F**: Combining the answers to the previous questions, what could different
periods of the signal represent?

Different periods of the signal represent different activity states, such as walking and running.

**Question G**: Can you distinguish between two types of steps (are they right and left
steps)? How? ‘Try and error’ your approaches. Are you basing your
supposition on external data?

As shown above, right and left steps can be seen as positive and negative peaks based on `rotationRate`, because the watch is oppositely rotated in the two sides.

**Question H**: Were there unmentioned sensors that proved more useful for your
analysis?

Although `motionUserAcceleration(G)` was mainly used, `motionUserRotationRate(rad/s)` also showed similar patterns that may prove relevant for further analysis. In addition, the information on gravity, pitch, yaw, roll and pedometer seemed to vary with activity mode, but most of them showed a non-stationary behaviour.
