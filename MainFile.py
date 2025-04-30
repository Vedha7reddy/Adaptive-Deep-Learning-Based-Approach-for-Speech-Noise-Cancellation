import os
import librosa
import numpy as np
import pandas as pd
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split
import noisereduce as nr
import torch
import torchaudio

# Directory containing the .wav files
wav_directory = "DataSet/vox1_indian/content/vox_indian/id10003/5ablueV_1tw"

# Lists to hold data
file_names = []
audio_data = []
sampling_rates = []

######################## Preprocessing ##############################################
######################## Voice Activity Detection (VAD) #######################################
# Function for Voice Activity Detection (VAD) using librosa's silence detection
def vad_librosa(audio, sr, top_db=20):
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
    voiced_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
    return voiced_audio
# Function to load audio file
def load_audio(file_path):
    """Load an audio file using librosa."""
    audio, sr = librosa.load(file_path, sr=None)  # sr=None to retain original sample rate
    return audio, sr

# Noise reduction function using noisereduce
def noise_reduction(audio, sr):
    """Reduce noise using the noisereduce library (spectral gating)."""
    denoised_audio = nr.reduce_noise(y=audio, sr=sr)
    return denoised_audio

# Function to save the denoised audio
def save_audio(file_path, audio, sr):
    """Save the denoised audio using soundfile."""
    sf.write(file_path, audio, sr)
    print(f"Denoised audio saved to {file_path}")

# Function to play audio at a slower speed
def play_audio_slower(audio, sr, speed_factor=0.5):
    """Play audio at a slower speed."""
    # Resample the audio to change the playback speed
    slower_audio = librosa.resample(audio, orig_sr=sr, target_sr=int(sr * speed_factor))
    display(Audio(slower_audio, rate=int(sr * speed_factor)))  # Play the slowed audio
####################### Noise Reduction ###############################################
# Simple Noise Reduction Using Spectral Gating
def noise_reduction_librosa(audio, sr, hop_length=512, n_fft=2048, win_length=2048):
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S, phase = librosa.magphase(D)
    
    # Estimate noise by applying a basic threshold on the magnitude
    noise_estimation = np.median(S, axis=1, keepdims=True)
    S_denoised = np.maximum(S - noise_estimation, 0)
    
    # Reconstruct the audio from the denoised magnitude and original phase
    D_denoised = S_denoised * phase
    audio_denoised = librosa.istft(D_denoised, hop_length=hop_length, win_length=win_length)
    
    return audio_denoised

###################### Deep Learning-based Noise Reduction ############################################
# Noise reduction using noisereduce (deep learning-based)
def deep_learning_noise_reduction(audio, sr):
    denoised_audio = nr.reduce_noise(y=audio, sr=sr)
    return denoised_audio

##################### Beamforming ####################################

# Beamforming (requires multi-microphone input)
import pyroomacoustics as pra
import numpy as np

# Beamforming (requires multi-microphone input)
def beamforming(audio, sr, mic_positions, mic_array, sound_source_position):
    # Simulate a room with microphone array
    room = pra.Room()
    
    # Define the microphone positions and add them to the room
    mic_array = pra.MicrophoneArray(np.array(mic_positions).T, sr)
    room.add_microphone_array(mic_array)
    
    # Define sound source position
    room.add_source(sound_source_position)
    
    # Perform beamforming using the microphone array
    beamformed_signal = room.beamform()
    
    # Assuming beamformed_signal is processed and returned
    return beamformed_signal


########################## GAN-based Noise Reduction Function ###############################
# Placeholder GAN-based Noise Reduction Function (you can replace it with your pre-trained model)
def gan_noise_reduction(audio, sr, model):
    """
    Perform noise reduction using a pre-trained GAN model.
    
    :param audio: Noisy audio waveform (1D numpy array or torch tensor).
    :param sr: Sampling rate of the audio.
    :param model: The pre-trained GAN model for noise reduction.
    
    :return: Denoised audio waveform.
    """
    # Convert audio to a torch tensor if it's a numpy array
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio, dtype=torch.float32)
    
    # Ensure the audio tensor is 2D (channels, samples) as required by most models
    audio = audio.unsqueeze(0)  # Add a batch dimension
    
    # Model inference (pass the noisy audio through the GAN model)
    with torch.no_grad():
        denoised_audio = model(audio)  # Assuming the model accepts this input shape
    
    # Convert back to numpy array if needed
    denoised_audio = denoised_audio.squeeze(0).numpy()  # Remove batch dimension
    return denoised_audio


######################## Integration with Your Preprocessing Pipeline ###########

# Iterate over each file in the directory
for filename in os.listdir(wav_directory):
    if filename.endswith('.wav'):
        file_path = os.path.join(wav_directory, filename)
        
        # Load the audio file using librosa
        try:
            audio, sr = librosa.load(file_path, sr=None)  # sr=None keeps the original sampling rate
            file_names.append(filename)
            
            # Preprocessing Steps:
            
            # Step 1: Beamforming (if applicable - requires multi-microphone data)
            # Uncomment if you have multi-microphone data and want to perform beamforming
            # audio = beamforming(audio, sr)
            
            # Step 2: Noise Reduction (using spectral gating)
            denoised_audio = noise_reduction_librosa(audio, sr)
            print(f"Step 2: Noise Reduction - {filename}")
            display(Audio(denoised_audio, rate=sr))  # Display audio after noise reduction
            
            # Step 3: Voice Activity Detection (VAD)
            vad_audio = vad_librosa(denoised_audio, sr)
            print(f"Step 3: Voice Activity Detection - {filename}")
            display(Audio(vad_audio, rate=sr))  # Display audio after VAD
            
            # Step 4: Deep Learning-based Noise Reduction (Optional)
            # Uncomment if you have a deep learning model for noise reduction
            # denoised_audio = deep_learning_noise_reduction(vad_audio, sr)
            # print(f"Step 4: Deep Learning Noise Reduction - {filename}")
            # display(Audio(denoised_audio, rate=sr))  # Display audio after deep learning noise reduction
            
            # Step 5: Further Noise Reduction (Optional, based on specific needs)
            final_audio = vad_audio  # Placeholder for final processed audio
            
            # Append processed data to lists
            audio_data.append(final_audio)
            sampling_rates.append(sr)
            
        except Exception as e:
            print(f"Error loading or processing {filename}: {e}")
            continue

######################## Print the heads ########################################
# After processing all files, ensure the lists are of equal length
if len(file_names) == len(audio_data) == len(sampling_rates):
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'file_name': file_names,
        'audio_data': audio_data,
        'sampling_rate': sampling_rates
    })
    # Show the DataFrame
    print(df.head())
else:
    print("Error: The lists have different lengths!")

###############################################################################
#############################GAN-based Noise Reduction Function##########################################

# Placeholder GAN-based Noise Reduction Function
def gan_noise_reduction(audio, sr, model):
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio, dtype=torch.float32)
    audio = audio.unsqueeze(0)
    with torch.no_grad():
        denoised_audio = model(audio)
    return denoised_audio.squeeze(0).numpy()

# Iterate over each file in the directory
for filename in os.listdir(wav_directory):
    if filename.endswith('.wav'):
        file_path = os.path.join(wav_directory, filename)
        
        try:
            audio, sr = librosa.load(file_path, sr=None)
            file_names.append(filename)
            
            # Step 2: Noise Reduction
            denoised_audio = noise_reduction_librosa(audio, sr)
            print(f"Step 2: Noise Reduction - {filename}")
            display(Audio(denoised_audio, rate=sr))
            
            # Step 3: Voice Activity Detection (VAD)
            vad_audio = vad_librosa(denoised_audio, sr)
            print(f"Step 3: Voice Activity Detection - {filename}")
            display(Audio(vad_audio, rate=sr))
            
            # Final processed audio
            final_audio = vad_audio
            
            # Append processed data to lists
            audio_data.append(final_audio)
            sampling_rates.append(sr)
            
        except Exception as e:
            print(f"Error loading or processing {filename}: {e}")
            continue

# Check lengths before creating DataFrame
if len(file_names) == len(audio_data) == len(sampling_rates):
    df = pd.DataFrame({
        'file_name': file_names,
        'audio_data': audio_data,
        'sampling_rate': sampling_rates
    })
    print(df.head())
else:
    print(f"Error: The lists have different lengths! -> file_names: {len(file_names)}, audio_data: {len(audio_data)}, sampling_rates: {len(sampling_rates)}")

######################### Data Splitting ##############################################
# Assuming df has been created already from the previous code
if len(df) > 0:
    # We assume 'audio_data' contains processed audio, 'sampling_rate' is not required for splitting
    X = df['audio_data'].values  # The audio data (feature)
    y = df['file_name'].values   # The target variable (could be label or file name for reference)

    # Split the dataset into training (70%), validation (15%), and testing (15%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Check if the split has been performed correctly
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # Optionally, store the splits into DataFrames for further processing
    train_df = pd.DataFrame({'audio_data': X_train, 'file_name': y_train})
    val_df = pd.DataFrame({'audio_data': X_val, 'file_name': y_val})
    test_df = pd.DataFrame({'audio_data': X_test, 'file_name': y_test})

    # Optionally, save the splits to CSV files
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)

    # Optionally, display a few examples from each set
    print("Training data example:", train_df.head())
    print("Validation data example:", val_df.head())
    print("Test data example:", test_df.head())
else:
    print("Error: DataFrame is empty, cannot split the data.")

#############################DNN + Wiener Filter##########################################
import torch
import torch.nn as nn
import numpy as np

# Define the Noise Estimator DNN
class NoiseEstimatorDNN(nn.Module):
    def __init__(self):
        super(NoiseEstimatorDNN, self).__init__()
        # Example layers (you can customize this)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, 1)  # Output one value for noise estimation

    def forward(self, noisy_audio):
        x = self.conv1(noisy_audio)  # Input should be [batch_size, channels, length]
        x = nn.ReLU()(self.conv2(x))
        x = x.mean(dim=2)  # Global average pooling
        estimated_noise_spectrum = self.fc(x)
        return estimated_noise_spectrum

# Define the Wiener Filter function
def wiener_filter(noisy_audio, estimated_noise_spectrum):
    # Compute the power spectral density of the noisy audio
    noisy_psd = np.abs(np.fft.fft(noisy_audio))**2
    # Compute the Wiener filter
    wiener_gain = noisy_psd / (noisy_psd + estimated_noise_spectrum.numpy())
    # Apply the filter
    enhanced_audio = np.fft.ifft(np.fft.fft(noisy_audio) * wiener_gain)
    return enhanced_audio

# Example usage
noisy_audio = np.random.randn(16000)  # Simulated noisy audio signal (1 second at 16 kHz)
model = NoiseEstimatorDNN()

# Convert noisy audio to tensor and reshape it
noisy_audio_tensor = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 16000]

# Estimate noise spectrum
estimated_noise = model(noisy_audio_tensor)

# Enhance audio using Wiener filter
enhanced_audio = wiener_filter(noisy_audio, estimated_noise.detach())  # Detach tensor to convert to numpy


######################### End-to-End Speech Enhancement Networks#############################################
class EndToEndSpeechEnhancement(nn.Module):
    def __init__(self):
        super(EndToEndSpeechEnhancement, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.rnn_layers = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, noisy_audio):
        x = self.cnn_layers(noisy_audio.unsqueeze(1))  # Add channel dimension
        x, _ = self.rnn_layers(x.transpose(1, 2))  # Transpose for LSTM
        enhanced_audio = self.fc(x)
        return enhanced_audio.squeeze(1)
    
################################PERFORMANCE METRICES############################################
import numpy as np
from scipy.stats import ttest_rel
import librosa
import speech_recognition as sr  # For word recognition accuracy (optional)

##############################
# 1. Word Recognition Accuracy
##############################
# Example function for word recognition accuracy using speech recognition
import speech_recognition as sr  # Ensure the library is correctly imported

def word_recognition_accuracy(reference_audio, processed_audio, sr):
    """
    This function calculates the word recognition accuracy by converting both
    reference (clean) and processed audio to text and comparing them.
    """

    recognizer = sr.Recognizer()
    
    # Convert reference audio to text
    with sr.AudioFile(reference_audio) as source:
        audio = recognizer.record(source)
        try:
            reference_text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            reference_text = ""
        except sr.RequestError:
            reference_text = ""
    
    # Convert processed audio to text
    with sr.AudioFile(processed_audio) as source:
        audio = recognizer.record(source)
        try:
            processed_text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            processed_text = ""
        except sr.RequestError:
            processed_text = ""
    
    # Compare the texts
    correct_words = len(set(reference_text.split()).intersection(set(processed_text.split())))
    total_words = len(reference_text.split())
    
    accuracy = correct_words / total_words if total_words > 0 else 0
    return accuracy


#########################
# 2. Signal-to-Noise Ratio (SNR)
#########################

def calculate_snr(signal, noise):
    """
    Calculate Signal-to-Noise Ratio (SNR) in decibels.
    SNR = 10 * log10(signal_power / noise_power)
    """
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = np.sum(noise ** 2) / len(noise)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


##########################
# 3. Noise Type Comparison
##########################
def compare_noise_types(original_audio, processed_audio_1, processed_audio_2, sr):
    """
    Compare two different noise reduction techniques in terms of SNR.
    Processed_audio_1 and processed_audio_2 should be the outputs from two different noise reduction methods.
    """
    original_signal = original_audio
    processed_signal_1 = processed_audio_1
    processed_signal_2 = processed_audio_2
    
    # Compute SNR for both methods
    snr_original = calculate_snr(original_signal, original_signal - processed_signal_1)
    snr_method_1 = calculate_snr(processed_signal_1, processed_signal_1 - original_signal)
    snr_method_2 = calculate_snr(processed_signal_2, processed_signal_2 - original_signal)
    
    print(f"SNR for original vs method 1: {snr_original}")
    print(f"SNR for method 1: {snr_method_1}")
    print(f"SNR for method 2: {snr_method_2}")
    
    return snr_method_1, snr_method_2


##############################
# 4. Statistical Significance Test
##############################
def statistical_significance_test(snr_method_1, snr_method_2):
    """
    Perform paired t-test to compare the SNRs of two noise reduction methods.
    """
    t_statistic, p_value = ttest_rel(snr_method_1, snr_method_2)
    
    print(f"T-statistic: {t_statistic}, P-value: {p_value}")
    
    if p_value < 0.05:
        print("The difference between the two methods is statistically significant.")
    else:
        print("The difference between the two methods is not statistically significant.")


##############################
# Example Usage of Performance Metrics
##############################
# Simulate original, processed audios for the example
original_audio = np.random.randn(16000)  # 1 second of white noise
processed_audio_1 = np.random.randn(16000)  # Denoised audio method 1
processed_audio_2 = np.random.randn(16000)  # Denoised audio method 2

# Calculate SNR for each method
snr_method_1, snr_method_2 = compare_noise_types(original_audio, processed_audio_1, processed_audio_2, 16000)

# Perform statistical significance test
statistical_significance_test(snr_method_1, snr_method_2)
####################Word Recognition Accuracy Visualization################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample data for the comparison (these values should be replaced with your actual results)
methods = ['Beamforming', 'Spectral Gating', 'VAD', 'Deep Learning']
high_snr_accuracy = [0.85, 0.88, 0.83, 0.95]  # Accuracy at high SNR (low noise)
low_snr_accuracy = [0.65, 0.70, 0.68, 0.90]   # Accuracy at low SNR (high noise)

# Create a DataFrame for easy manipulation
df_accuracy = pd.DataFrame({
    'Method': methods,
    'High SNR Accuracy': high_snr_accuracy,
    'Low SNR Accuracy': low_snr_accuracy
})

# Plotting Word Recognition Accuracy
plt.figure(figsize=(6, 6))
bar_width = 0.35
index = np.arange(len(methods))

# Bar plots for high SNR and low SNR accuracy
plt.bar(index, df_accuracy['High SNR Accuracy'], bar_width, label='High SNR', color='b', alpha=0.7)
plt.bar(index + bar_width, df_accuracy['Low SNR Accuracy'], bar_width, label='Low SNR', color='r', alpha=0.7)

# Add labels and title
plt.xlabel('Methods', fontsize=12)
plt.ylabel('Word Recognition Accuracy', fontsize=12)
plt.title('Word Recognition Accuracy Comparison by Noise Reduction Method', fontsize=14)
plt.xticks(index + bar_width / 2, df_accuracy['Method'], rotation=45)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()
#######################Statistical Testing Result Visualization#########################################
from scipy import stats

# Sample data (these values should be replaced with your actual results)
beamforming_accuracy = [0.85, 0.87, 0.86, 0.82, 0.88]  # Results for Beamforming
spectral_gating_accuracy = [0.88, 0.89, 0.87, 0.85, 0.90]  # Results for Spectral Gating
vad_accuracy = [0.83, 0.82, 0.81, 0.80, 0.84]  # Results for VAD
deep_learning_accuracy = [0.95, 0.96, 0.94, 0.97, 0.93]  # Results for Deep Learning

# Perform t-test between Deep Learning and Beamforming (as an example)
t_stat, p_value = stats.ttest_ind(deep_learning_accuracy, beamforming_accuracy)

# Print p-value to check for statistical significance
print(f"P-value between Deep Learning and Beamforming: {p_value}")

# Bar chart for statistical results
methods = ['Beamforming', 'Spectral Gating', 'VAD', 'Deep Learning']
accuracies = [np.mean(beamforming_accuracy), np.mean(spectral_gating_accuracy),
              np.mean(vad_accuracy), np.mean(deep_learning_accuracy)]
std_devs = [np.std(beamforming_accuracy), np.std(spectral_gating_accuracy),
            np.std(vad_accuracy), np.std(deep_learning_accuracy)]

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(methods, accuracies, yerr=std_devs, capsize=5, color='lightblue', alpha=0.7)
plt.xlabel('Methods', fontsize=12)
plt.ylabel('Mean Word Recognition Accuracy', fontsize=12)
plt.title('Word Recognition Accuracy with Error Bars', fontsize=14)

# Display the plot
plt.tight_layout()
plt.show()
####################SNR vs. Word Recognition Accuracy Plot############################################
# Sample data for SNR vs Accuracy
snr_levels = [5, 10, 15, 20, 25]  # SNR Levels
beamforming_snr_accuracy = [0.60, 0.70, 0.75, 0.80, 0.85]
spectral_gating_snr_accuracy = [0.65, 0.75, 0.80, 0.85, 0.88]
vad_snr_accuracy = [0.60, 0.70, 0.74, 0.78, 0.82]
deep_learning_snr_accuracy = [0.70, 0.80, 0.85, 0.90, 0.95]

# Plotting SNR vs. Accuracy
plt.figure(figsize=(10, 6))
plt.plot(snr_levels, beamforming_snr_accuracy, label='Beamforming', marker='o', linestyle='-', color='blue')
plt.plot(snr_levels, spectral_gating_snr_accuracy, label='Spectral Gating', marker='o', linestyle='-', color='green')
plt.plot(snr_levels, vad_snr_accuracy, label='VAD', marker='o', linestyle='-', color='red')
plt.plot(snr_levels, deep_learning_snr_accuracy, label='Deep Learning', marker='o', linestyle='-', color='purple')

# Add labels and title
plt.xlabel('SNR Level (dB)', fontsize=12)
plt.ylabel('Word Recognition Accuracy', fontsize=12)
plt.title('SNR vs Word Recognition Accuracy for Different Methods', fontsize=14)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()
#########################PREDICTION############################################
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Sample data (replace these with your actual values)
snr_levels = np.array([5, 10, 15, 20, 25]).reshape(-1, 1)  # SNR Levels
deep_learning_accuracy = np.array([0.70, 0.80, 0.85, 0.90, 0.95])  # Deep Learning Accuracy

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(snr_levels, deep_learning_accuracy)

# Predict future values (e.g., SNR from 30 to 40)
future_snr = np.array([30, 35, 40]).reshape(-1, 1)
predicted_accuracy = model.predict(future_snr)

# Plotting the actual and predicted data
plt.figure(figsize=(10, 6))

# Plot the actual data
plt.scatter(snr_levels, deep_learning_accuracy, color='blue', label='Actual Data', marker='o')

# Plot the regression line
plt.plot(snr_levels, model.predict(snr_levels), color='green', label='Fitted Line')

# Plot the predicted data for future SNR levels
plt.scatter(future_snr, predicted_accuracy, color='red', label='Predicted Data', marker='x')

# Add labels and title
plt.xlabel('SNR Level (dB)', fontsize=12)
plt.ylabel('Word Recognition Accuracy', fontsize=12)
plt.title('Prediction of Word Recognition Accuracy vs. SNR Level (Deep Learning)', fontsize=14)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

# Display the future predictions
print("Future Predictions (SNR 30, 35, 40):")
for snr, accuracy in zip(future_snr.flatten(), predicted_accuracy):
    print(f"SNR = {snr} dB -> Predicted Accuracy = {(accuracy*100):.1f}")




#################################################################################

import os
import librosa
import numpy as np
import soundfile as sf  # To save the output as a .wav file
import noisereduce as nr  # For noise reduction
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from datetime import datetime
from IPython.display import Audio, display  # For audio playback in console



# Main function to denoise audio
def denoise_audio(input_file, output_folder):
    """Main function to denoise audio."""
    # Load the noisy audio
    audio, sr = load_audio(input_file)
    
    # Perform noise reduction
    denoised_audio = noise_reduction(audio, sr)
    
    # Create a unique output file name using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(output_folder, f"denoised_audio_{timestamp}.wav")
    
    # Save the denoised audio to the output file
    save_audio(output_file_path, denoised_audio, sr)
    
    # Play the denoised audio at a slower speed
    print("Playing denoised audio at a slower speed...")
    play_audio_slower(denoised_audio, sr, speed_factor=0.5)  # Adjust speed_factor as needed

    return denoised_audio, sr

# Function to upload a file using tkinter
def upload_file():
    """Function to upload a file using tkinter."""
    # Create a Tkinter root window (it will be hidden)
    Tk().withdraw()  # Prevents the root window from appearing

    # Open a file dialog to select a .wav file
    input_file_path = askopenfilename(title="Select a .wav file", filetypes=[("WAV files", "*.wav")])

    if input_file_path:
        print(f"File uploaded: {input_file_path}")
        
        # Create a directory for storing denoised audio files
        output_folder = "denoised_audios"
        os.makedirs(output_folder, exist_ok=True)  # Create directory if it doesn't exist
        
        # Denoise the uploaded audio file
        denoised_audio, sr = denoise_audio(input_file_path, output_folder)
        
        print("Denoised audio saved and ready to play.")
        return denoised_audio, sr
    else:
        print("No file selected.")

# Run the upload file function to allow the user to upload a .wav file
upload_file()
