# IMPORTANT
We have 2 separate codes for MNIST/Fashion MNIST and MIT-BIH Arrhytmia datasets: 1) MNIST-DRBFDD.py and 2) ECG-DRBFDD.py

You need to adjust the parameters for the DRBFDD network in the parameters.yaml file.

Read the comments in the .yaml file carefully as given a dataset, you need to change a few things to apply a certain algorithm:

1. For MNIST/Fashion MNIST you can run D-RBFDD network as well as Fix-Res + RBFDD and Fine-Res + RBFDD networks and you will need to run MNIST-DRBFDD.py

2. For MIT-BIH Arrhytmia dataset, you will forcus on the parameters for the CNN1D network in the .yaml file and you will run ECG-DRBFDD.py


# DRBFDD
This repository contains the source code for the Deep Radial Basis Function Data Descriptor (DRBFDD) network.
The necessary ECG data, have been collected from MIT-BIH dataset, the individual heart-beats have been extracted, normalized between [0,1].
Each heart-beat comes with its corresponding label, in terms of being normal (label:0) or any one of the anomalies (labels:1,2,3,4). PLEASE NOTE: We have
only collected these 5 classes from the public dataset. The pre-processed ECG dataset can be found at: https://drive.google.com/drive/folders/1ugBr27gUpAlWFmRxwKxHMtpDBghdkTwr?usp=sharing

The pre-processing steps are detailed down below for all datasets used.

# Datasets Used

## MNIST and Fashion-MNIST
MNIST and Fashion-MNIST are already publicly available, and the code will automatically download them. The only pre-processing applied on these grey-scale images is normalization between [0,1].

## Pre-processing the MIT-BIH Arrhythmia Dataset
This is a bi-channel ECG dataset. The data is collected from 47 patients and for each patient we have 30 minute (360 samples/sec) worth of sensory readings. Every heart-beat within a given 30 minute recording is labelled. The majority of the dataset is comprised ofthe normal class and as for the anomalies, we have chosen the top 4 most frequent anomaly types in the dataset. For a healthy human heart, the ECG waveform for
each heart beat resembles the one depicted in the Figure below. It is composed of the P, Q, R, S, and T waves and beat detection is usually performed by searching for each
R-peak or QRS complex:

The labels of the classes used, along with the number of provided labels. N denotes the normal class and the pther 4 are the most frequent anomalies in the dataset:

1. 'N':  -- 72813
2. 'L':  -- 8075
3. 'R':  -- 7259
4. 'V':  -- 7129
5. '/':  -- 7028

![A healthy heart-beat](https://github.com/MLDawn/DRBFDD/blob/main/heartbeat.png)


Obviously, these are the number of labels for each heart-beat and we need to actually extract these heart-beats from the recordings along with these labels. We need to understand the general pattern of a heart-beat to help us come up with a heart-beat extraction strategy. First let's see a sample patient's heart-beat pattern for both channels in this dataset:

![A Sample Recording of the MIT-BIH Dataset for Both Channels](https://github.com/MLDawn/DRBFDD/blob/main/sample%20recording%20MIT.png)

The little cross signs show where the R peaks in a heart-beat have happened. The dataset provides us with these peak lists for all recordings and all patients. Below, we will cover the multiple pre-processing steps that we have taken for this dataset.

### Normalisation
We have done a patient-based channel-based z-norm normalisation. Meaning, for every patient, we have normalized each of the recordings of each sensor between [0,1], independent of the other channel, and other patients.

### Segmenting Heart-Beats

In order to segment individual heart-beats in these ECG recordings we use the mid-point of 2 consecutive R peaks, as the boundary between the 2 peaks. This way, the segmented part will for sure include the QRS complex (i.e., the most important part of a heart-beat) for that heart-beat at the very least and there will be no overlaps between the segments. In order to isolate heart-beat i in the ECG signal, we will consider the position of the peak for this heart-beat, p_i, the position of the peak for the previous heart-beat,p_{i-1}, and the position of the heart-beat in the next heart-beat, p_{i+1} (these positions are provided in the dataset). Then we will consider the mid-point between p_i and p_{i-1} to be the boundary between the i^{th} heart-beat and the (i-1)^{st} hearth-beat. Similarly, we will use the mid-point between o_i and p_{i+1} to be the boundary between the i^{th} heart-beat and the (i+1)^{st} hearth-beat. Then we will grab this portion of the signal as the $i^{th}$ heart-beat. This process will continue to segment each individual heart-beat across the dataset.

Now, we have individual heart-beats, with their corresponding class labels, and we are sure that the values for each heart-beat is between 0 and 1 as we have normalized the entire dataset in the previous step.

### Making Lengths Equal through Padding
After isolating each heart-beat, clearly, we will end up with heart-beats of different sizes. We have zero-padded all of them to the length of the longest ehart-beat (that is 542).

### Lowering the frequency
In order to lower the dimensionality of the data, we will down-sample the data to 187 Hz from the original sampling rate that is 360 Hz.


