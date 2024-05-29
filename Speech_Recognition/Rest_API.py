import requests
import pandas as pd
import json
import torch
import torchaudio
import numpy as np

from Dataset import MFCC


# Function to process the audio data
def process_audio(waveform, sample_rate):
    featurizer = MFCC(sample_rate=sample_rate)
    mfcc = featurizer(waveform)
    return mfcc


# Define the URL of the model server
url = "http://localhost:8001/invocations"

# Path to the .wav file
sample_rate = 44100
# Use a validation sample to test the request
wav_file_path = "dataset/Validation/1/washington_d-0.wav"
waveform, sr = torchaudio.load(wav_file_path)
waveform = torch.mean(waveform, dim=0, keepdim=True)

data = process_audio(waveform, sample_rate).transpose(1, 2)

data = data.numpy().tolist()

# Send the POST request
response = requests.post(
    url,
    data=json.dumps({"inputs": data}),
    headers={"Content-Type": "application/json"},
    timeout=10,
)

# Check if the request was successful
if response.status_code == 200:
    predictions = response.json()
    print(
        "Prediction:",
        torch.sigmoid(
            torch.tensor(predictions["predictions"], dtype=torch.float32)
        ).numpy(),
    )
else:
    print("Request failed with status code:", response.status_code)
    print("Response body:", response.text)
