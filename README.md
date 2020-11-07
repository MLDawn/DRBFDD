# DRBFDD
This repository contains the source code for the Deep Radial Basis Function Data Descriptor (DRBFDD) network.
The necessary ECG data, have been collected from MIT-BIH dataset, the individual heart-beats have been extracted, normalized between [0,1].
Each heart-beat comes with its corresponding label, in terms of being normal (label:0) or any one of the anomalies (labels:1,2,3,4). PLEASE NOTE: We have
only collected these 5 classes from the public dataset.

# Datasets Used

## MNIST and Fashion-MNIST
MNIST and Fashion-MNIST are already publicly available, and the code will automatically download them. The only pre-processing applied on these grey-scale images is normalization between [0,1].

\subsection{Pre-processing the MIT-BIH Arrhythmia Dataset}
This is a bi-channel ECG dataset. The data is collected from 47 patients and for each patient we have 30 minute (360 samples/sec) worth of sensory readings. Every heart-beat within a given 30 minute recording is labelled. The majority of the dataset is comprised ofthe normal class and as for the anomalies, we have chosen the top 4 most frequent anomaly types in the dataset. For a healthy human heart, the ECG waveform for
each heart beat resembles the one depicted in Fig.~~\ref{fig:Sample ECG signal}:
it is composed of the P, Q, R, S, and T waves and beat
detection is usually performed by searching for each
R-peak or QRS complex. The class label descriptions can be found in Table~\ref{datasets}:

\begin{itemize}
    \item 'N':  -- 72813
    \item 'L':  -- 8075
    \item 'R':  -- 7259
    \item 'V':  -- 7129
    \item '/':  -- 7028
\end{itemize}

\begin{figure}[t]
        \begin{center}
            \includegraphics[scale=.9]{template_2/figures/heartbeat.png}
            \caption{Sample Healthy Heartbeat}
             \label{fig:Sample Healthy Heartbeat}
            
        \end{center}
\end{figure}

Obviously, these are the number of labels for each heart-beat and we need to actually extract these heart-beats from the recordings along with these labels. We need to understand the general pattern of a heart-beat to help us come up with a heart-beat extraction strategy. First let's see a sample patient's heart-beat pattern for both channels in this dataset:

\begin{figure}[t]
        \begin{center}
            \includegraphics[scale=.45]{template_2/figures/sample recording MIT.png}
            \caption{Sample ECG signal for a random patient with both channels}
             \label{fig:Sample ECG signal}
            
        \end{center}
\end{figure}

The little cross signs show where the \textit{R} peaks in a heart-beat have happened. The dataset provides us with these peak lists for all recordings and all patients. Below, we will cover the multiple pre-processing steps that we have taken for this dataset.

\subsubsection{Normalisation}
According to the description of the dataset \footnote{https://www.physionet.org/physiobank/database/html/mitdbdir/intro.htm}, the sample values in this dataset range from 0 to 2047 inclusive, with 1024 corresponding to zero. This is due to the fact that, at the digitization step, a resolution of 11-bits has been used, which will result in $2^{11}$ levels, which are the actual resultant values of the signal. Thus, having the minimum and maximum of all possible values in this dataset can help us normalize the entire dataset between 0 and 1 by simply dividing all values by the maximum value (i.e., 2047). We do this as the very first stage of pre-processing!

\subsubsection{Segmenting Heart-Beats}
% The next step is to segment and isolate each heartbeat with its corresponding label. There are different ways to approach segmenting a continuous ECG signal either into smaller segments or into individual heart-beats. For instance, in \cite{tsipouras2002arrhythmia} the RR interval, that is the distance between the \textit{R} peaks of two consecutive heart-beats, is the unit of measure. In particular,a sliding window as big as 3 RR interval, centered in the middle RR interval is used. The task is to use a 3 RR interval sliding window in order to classify the middle heart-beat! The 3RR window size is indeed highly over-lapping and 2 consecutive windows will have as much as one R interval of over-lap with this method. As a result, this method ensures that the entire heart-beat in the middle is captured within the window but it captures some of the previous and next heart-beat too. Some other works such as \cite{6316585}, the authors have used a mechanism to extract the so-called \textit{QRS} complex portion of every heart-beat and use that as the data unit for their task. In particular, given an \textit{R} peak position for a given heart-beat, the on-set of the \textit{QRS} complex for that heart-beat is considered 64 ms (milli-second) earlier than the \textit{R} peak, and the ending of the \textit{QRS} is considered to be 188 ms after the \textit{R} peak. These \textit{QRS} complex segments are used for training the classifier as each one has a corresponding label as to being normal or anomalous (of some kind). In other works we have a mixture of using raw data as well as some features. For instance, in \cite{8050805}, for every R peak, 181 raw sample values around the peak were selected, as well as, the RR-intervals to the succeeding and preceding beats and this has been implemented for every R peak in the dataset. While the QRS complex is probably the most important part of a heart-beat \cite{zidelmal2012qrs}, certain anomalies can happen outside the QRS complex region \cite{ashleycardiology} and it would be better to try and capture the heart-beat as much as possible while avoiding the parts of the neighboring heart-beats.

In order to segment individual heart-beats in these ECG recordings, following the approach in \cite{xu2018towards}, we use the mid-point of 2 consecutive \textit{R} peaks, as the boundary between the 2 peaks. This way, the segmented part will for sure include the QRS complex (i.e., the most important part of a heart-beat \cite{zidelmal2012qrs}) for that heart-beat at the very least and there will be no overlaps between the segments. In order to isolate heart-beat \textit{i} in the ECG signal, we will consider the position of the peak for this heart-beat, $p_i$, the position of the peak for the previous heart-beat,$p_{i-1}$, and the position of the heart-beat in the next heart-beat, $p_{i+1}$ (these positions are provided in the dataset). Then we will consider the mid-point between $p_i$ and $p_{i-1}$ to be the boundary between the $i^{th}$ heart-beat and the $(i-1)^{st}$ hearth-beat. Similarly, we will use the mid-point between $p_i$ and $p_{i+1}$ to be the boundary between the $i^{th}$ heart-beat and the $(i+1)^{st}$ hearth-beat. Then we will grab this portion of the signal as the $i^{th}$ heart-beat. This process will continue to segment each individual heart-beat across the dataset, below you can see the math for this segmentation:

\begin{equation}\label{hear-beat segmentation}
    beat_i = Signal\left[p_i - \frac{p_i - p_{i-1}}{2}:p_i + \frac{p_{i+1} - p_i}{2}\right] 
\end{equation}

Now, we have individual heart-beats, with their corresponding class labels, and we are sure that the values for each heart-beat is between 0 and 1 as we have normalized the entire dataset in the previous step.

\subsubsection{Making Lengths Equal through Padding/Truncating}
after isolating each heart-beat, clearly, we will end up with heart-beats of different sizes. Following the approach by \citet{xu2018towards}, we consider a fixed length of samples, \textit{D}, per heart-beat through measuring the length of \textbf{all} the segmented heart-beats and a value that is larger than 95\% of all the measured duration is chosen to be assigned to $D$. Below, the distribution of these lengths (before padding) for normal, anomalous, as well as, all heart-beats together along with some useful statistics such as minimum length, maximum length, and the average (i.e., mean) length. The visualizations in Figure~\ref{fig:lengths boxplot} and Figure~\ref{fig:lengths histograms} will provide us with the big picture.

\begin{figure}[h]
        \begin{center}
            \includegraphics[scale=.25]{template_2/figures/lengths-1.png}
            \caption{Distribution of the lengths of normal, anomalous, and all heart-beats (from left to right) in MIT-BIH Arrhythmia dataset before zero-padding, visualized with boxplots. These heart-beats belong to either of the 5 classes that we have considered for this dataset.}
             \label{fig:lengths boxplot}
            
        \end{center}
\end{figure}

\begin{figure}[h]
        \begin{center}
            \includegraphics[scale=.25]{template_2/figures/lengths-2.png}
            \caption{The normalized distribution of the lengths of normal, anomalous, and all heart-beats (from left top right) in MIT-BIH Arrhythmia dataset before zero-padding, visualized with histograms. These heart-beats belong to either of the 5 classes that we have considered for the experiments in this dataset}
             \label{fig:lengths histograms}
            
        \end{center}
\end{figure}

It seems that the normal and anomalous class heart-beats have a similar distribution for the lengths of the segmented heart-beats, with a similar range. Looking at the distribution of all the lengths together (the black histogram), following the approach by \citet{xu2018towards} we will choose a length that is higher than 95\% of the heart-beat lengths, and it is 417. After this step, in \cite{xu2018towards} have an extra step of signal alignment, where they have centered the heart-beats in a window of size $D$ making sure that the $R$ peak of all heart-beats would fall on the mid-point of this window, and then they have the zero-padding/truncating stage (on one side or sometimes even on both sides of the signal where necessary) where they make sure that all signals will have the length of $D$. We would argue that the alignment stage, while making sense as a pre-processing stage, will prevent the anomaly detector to learn more robust and stronger features with more generalization ability. By not aligning the signals, we will force the anomaly detector to strive for stronger and perhaps more complex features to pick up the actual pattern within the training data. As we are not centralizing and aligning these heart-beats, we will then apply the padding and truncating to the end of the signals where necessary to make sure all signals will have the length of $D$ (i.e., 417). As in the original data, the sample values range from 0 to 2047 inclusive, with 1024 (the mid-point) corresponding to zero, after normalizing between 0 and 1, the original value 1024 will be translated to 0.50. This is the value by which we will pad our signals. In terms of truncating, we will simply truncating the ending of the signals.

\subsubsection{Lowering the frequency}
In order to lower the dimensionality of the data, we will down-sample the data to 187 Hz from the original sampling rate that is 360 Hz.


