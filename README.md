# Detecting Synthetic Speech with a Convolutional Neural Network

A convolutional neural network that classifies a short speech clip as either
genuine human speech or machine-generated speech, trained on the ASVspoof 2019
Logical Access dataset and served through a web application. The audio front end
and the network were assembled from PyTorch and librosa primitives rather than
from a prebuilt speech toolkit.

The model reaches 97.15% accuracy on the held-out development set, but that
figure sits only marginally above a trivial baseline, and the interesting part
of the project is understanding why.

## Background

A spoofing attack presents synthetic or converted speech to a system that
expects a real voice. The ASVspoof challenge series collects such attacks so
that countermeasures can be trained against them. The 2019 Logical Access set
contains genuine recordings labelled "bonafide" and synthetic recordings
produced by nineteen different text-to-speech and voice-conversion systems,
labelled "spoof".

Distinguishing the two by ear is often possible but slow, and does not scale.
A trained network produces a decision in milliseconds. The label a network
predicts is only ever "does this resemble the bonafide recordings it was trained
on", not "was this produced by a machine" in any absolute sense, and that
distinction turns out to matter for how the results should be read.

## Representation

Raw audio is a one-dimensional waveform of varying length, which a convolutional
network cannot consume directly. Each clip is resampled to 16 kHz, then padded
with silence or cropped to exactly 64,000 samples so that every input has the
same size. The fixed waveform is passed through a Constant-Q Transform, which
produces a two-dimensional array whose rows are log-spaced frequency bins and
whose columns are time frames, and the magnitudes are converted to decibels.

The Constant-Q Transform was chosen over mel spectrograms and over cepstral
coefficients because its log-frequency spacing matches musical pitch and because
it preserves the two-dimensional time-frequency structure a convolution operates
on, rather than collapsing it the way a cepstral representation does. The result
is 84 frequency bins by roughly 125 time frames, treated as a single-channel
image.

## Architecture

Three convolutional layers with 3x3 kernels widen the single input channel to 16,
then 32, then 64 feature maps, each followed by a ReLU activation and 2x2 max
pooling that halves both dimensions. An adaptive average pooling layer then
collapses whatever spatial size remains to a single value per feature map,
producing a 64-dimensional vector regardless of the exact number of time frames.
A final linear layer maps those 64 values to two class scores.

The adaptive pooling layer is what makes the network tolerant of small
differences in clip length, since it removes any dependence on the exact time
dimension before the linear layer. The saved model is roughly 95 KB.

## Training

The training split contains about 25,000 clips, of which approximately 90% are
spoof and 10% bonafide, because nineteen attack systems each contribute
recordings while genuine speech forms a single smaller category. This imbalance
is the single most important fact about the results below.

Training used a batch size of 32 and the Adam optimizer at a learning rate of
1e-3, with cross-entropy loss, for 16 epochs on an Apple M-series processor using
the MPS backend. Training accuracy rose from 89.72% in the first epoch to 98.39%
in the last, and average loss fell from 0.32 to 0.046. No normalization was
applied to the decibel spectrograms, since their values already occupy a fixed
range of roughly -80 to 0 and training was stable without it.

The first-epoch accuracy of 89.72% is close to the proportion of spoof examples
in the data, which is the expected starting point: a network that has learned
nothing yet can score about 90% simply by predicting the majority class every
time. The rise above that level over subsequent epochs is the evidence that the
model learned something beyond the majority-class shortcut on the training data.

## Results

The development set was evaluated once, after training was complete, using the
same preprocessing pipeline. It contains 24,844 clips from 20 speakers not seen
during training. Spoof is treated as the positive class throughout.

| | Value |
|---|---|
| Accuracy | 0.9715 |
| Precision | 0.9916 |
| Recall | 0.9765 |
| F1 | 0.9840 |

The confusion matrix underlying those figures:

| | Predicted bonafide | Predicted spoof |
|---|---|---|
| Actually bonafide | 2363 | 185 |
| Actually spoof | 524 | 21772 |

Because 89.7% of the development set is spoof, a model that predicted spoof for
every clip would already score 89.74% accuracy. The model's 97.15% is above that
floor, so it has learned genuine discrimination rather than only the majority
guess, but the floor is what makes the headline number interpretable: roughly
seven points of accuracy separate this model from a constant answer that ignores
the audio entirely.

Reading the two error types separately is more informative than the single
accuracy figure. Of 2,548 genuine clips, 185, or 7.3%, were wrongly flagged as
spoof. Of 22,296 spoof clips, 524 were missed. The high precision means a "spoof"
verdict is usually correct, while the false-positive rate on genuine speech is
the model's weaker side and the more consequential error in a system whose
purpose is to admit real users.

## Behaviour outside the training distribution

A phone recording of a real voice, made through the microphone and compression
of a consumer device rather than through the ASVspoof recording pipeline, is
classified as spoof by this model. This is not evidence that the recording is
fake. It is a consequence of the model having learned what bonafide clips look
like specifically as produced by the dataset's pipeline, so that audio arriving
through any other pipeline falls outside the distribution it can judge.

This is the same limitation the results section hints at, stated more strongly:
the accuracy figure holds only for audio that resembles the training data, and
degrades on anything else. It is the motivation for the interpretability work
described below.

## Limitations

* **Distribution scope.** Both the training and development data come from the
  same ASVspoof recording pipeline, so the reported accuracy measures performance
  on that pipeline only. Consumer recordings, other datasets, and attack systems
  newer than 2019 are outside the tested distribution and predictions there are
  unreliable.
* **Class imbalance.** The roughly nine-to-one spoof majority means accuracy is a
  weak summary on its own, which is why precision, recall, and the full confusion
  matrix are reported alongside it.
* **False positives on genuine speech.** The 7.3% rate at which real clips are
  flagged as spoof is the model's most practically important weakness, since the
  cost of rejecting a real user usually exceeds the cost of admitting one fake.
* **Fixed clip length.** Every input is forced to four seconds, so information
  beyond that in longer clips is discarded and short clips are padded with silence
  the model may or may not learn to ignore.
* **Single evaluation split.** Only the development set was used for evaluation;
  the separate ASVspoof evaluation set, which contains attack systems not present
  in training, was not run, so generalization to unseen attack types is untested.

## Interpretability audit (in progress)

The gap between the high reported accuracy and the failure on out-of-distribution
audio raises the question of what the network actually keys on. The planned
follow-up applies Grad-CAM to the final convolutional layer to produce heatmaps
over the time-frequency input, aggregates those maps across many spoof
predictions to look for a systematic pattern rather than a single anecdote, and
then tests any suspected shortcut causally by masking the implicated region and
measuring whether predictions collapse. The framing is an audit of what the model
learned and whether it is trustworthy, not a claim of improved accuracy; a
finding that the model relies on a shortcut would be a useful negative result.

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The training notebook downloads the dataset through kagglehub on first run, which
takes several minutes and requires roughly 23 GB of disk space.

```bash
jupyter notebook deepfaketraining.ipynb
```

Web application, requiring two terminals:

```bash
uvicorn main:app --reload                      # backend on port 8000
cd frontend && npm install && npm run dev      # frontend on port 5173
```

The page accepts an audio file, sends it to the backend, and displays the
predicted label with a confidence value. The backend decodes common audio formats
through librosa, so flac, wav, mp3, and m4a inputs all work.

## Files

```
deepfaketraining.ipynb    Data pipeline, model, training, dev-set evaluation
model.py                  AudioCNN class, shared by training and the API
preprocessing.py          Audio to CQT-spectrogram tensor, identical to training
main.py                   FastAPI backend, loads the model and serves /predict
audio_cnn_model.pth       Trained weights
requirements.txt          Python dependencies
frontend/                 React application (Vite)
```
