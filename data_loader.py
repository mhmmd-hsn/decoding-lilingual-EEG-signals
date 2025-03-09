import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import torch
from connections import GMatrixCalculator
from torch.utils.data import Dataset
from sklearn.decomposition import FastICA

class EEGDataLoader(Dataset):
    def __init__(self, root_path: str, class_type, trial_type):        
        self.root_path = Path(root_path)
        self.sessions = [d for d in sorted(self.root_path.iterdir()) if d.is_dir()]
        
        # Electrode selection for different conditions
        self.selected_electrodes = {
            'AK-SREP': ["F8", "FCz", "Fp1", "AF7", "AF3", "C1", "FC4", "F1", "Pz", "F2", "P5", "P6"],
            'CTK-SREP': ["F1", "P5", "F4", "AF7", "PO5", "FC1", "FCz", "Fp2", "Fz", "PO3", "TP8", "F4"],

            'AK-SRES': ["AF8", "FC6", "F8", "T8", "C6", "AF7", "Fp2", "CP6", "Fpz", "F4", "TP8", "P6"],
            'CTK-SRES': ["F1", "P4", "F4", "AF3", "CP6", "P5", "Fz", "FC2", "FC1", "P6", "PC6", "F2"]
        }

        self.data, self.labels = self.get_trials(class_type=class_type, trial_type=trial_type)

    def load_eeg_data(self, parquet_file: str):
        df = pd.read_parquet(parquet_file)
        return df

    def _downsample(self, data, factor=2):
        """Downsamples the EEG signal along the last axis."""
        return data[:, :, ::factor]

    def _butterworth_lowpass_filter(self, data, fs=1000, cutoff=50, order=7):
        """
        Applies a Butterworth low-pass filter to EEG data.
        """
        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=-1)
        return filtered_data

    def _common_average_reference(self, data):
        """
        Applies Common Average Reference (CAR) to EEG data.
        """
        avg_reference = np.mean(data, axis=1, keepdims=True)  # Compute mean across electrodes
        return data - avg_reference  # Subtract average from each electrode

    def _apply_ica(self, data):
        """
        Applies Independent Component Analysis (ICA) to remove artifacts.
        """
        reshaped_data = data.reshape(data.shape[0], data.shape[1], -1)  # Flatten trials to 2D (trials, features)
        ica = FastICA(n_components=reshaped_data.shape[1])  # Number of components = number of channels
        cleaned_data = []
        for trial in reshaped_data:
            transformed = ica.fit_transform(trial.T)  # Apply ICA
            trial_cleaned = ica.inverse_transform(transformed).T  # Reconstruct cleaned signal
            cleaned_data.append(trial_cleaned)
        return np.array(cleaned_data).reshape(data.shape)  # Reshape back to (trials, channels, time_samples)

    def _min_max_normalization_per_signal(self, data, feature_range=(0, 1)):
        """
        Normalizes EEG data using Min-Max scaling per trial.
        """
        min_val, max_val = feature_range
        data_min = np.min(data, axis=-1, keepdims=True)
        data_max = np.max(data, axis=-1, keepdims=True)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1  # Avoid division by zero
        normalized_data = (data - data_min) / data_range
        normalized_data = normalized_data * (max_val - min_val) + min_val
        return normalized_data

    def extract_trials(self, df, class_type="AK-SREP", trial_type="reading", target_samples=4000):
        """
        Extracts EEG trials and labels from a DataFrame.
        """
        event_column = df.iloc[:, -1]
        selected_electrodes = self.selected_electrodes[class_type]
        trials = []
        labels = []
        start_event = 21 if trial_type == "reading" else 30
        event_indices = event_column[event_column.notna()].index.tolist()
        event_values = event_column.dropna().values

        for i in range(len(event_values)):
            if event_values[i] == start_event:  
                if i == 0:
                    continue
                label = event_values[i - 1] if trial_type == "reading" else event_values[i - 3]
                start_idx = event_indices[i]
                if i + 1 < len(event_indices):
                    end_idx = event_indices[i + 1]
                else:
                    continue
                trial_data = df.loc[start_idx:end_idx, selected_electrodes].values.T
                current_samples = trial_data.shape[1]

                if current_samples < target_samples:
                    padding = np.zeros((trial_data.shape[0], target_samples - current_samples))
                    trial_data = np.hstack((trial_data, padding))
                elif current_samples > target_samples:
                    trial_data = trial_data[:, :target_samples]

                trials.append(trial_data)
                labels.append(label)

        return np.array(trials), np.array(labels)

    def get_trials(self, class_type="AK-SREP", trial_type="reading"):
        """
        Extracts and processes all trials from all EEG files.
        """
        all_trials = []
        all_labels = []

        for session in tqdm(self.sessions, desc="Processing Sessions"):
            eeg_files = list(session.glob("*.parquet"))
            for eeg_file in eeg_files:
                df = self.load_eeg_data(str(eeg_file))
                trials, labels = self.extract_trials(df, class_type=class_type, trial_type=trial_type)
                all_trials.extend(trials)
                all_labels.extend(labels)

        all_trials = np.array(all_trials)
        all_labels = np.array(all_labels)

        # Processing Pipeline
        filtered_data = self._butterworth_lowpass_filter(all_trials, cutoff=50)  # 1. Apply Butterworth Filter
        referenced_data = self._common_average_reference(filtered_data)         # 2. Apply CAR
        cleaned_data = self._apply_ica(referenced_data)                         # 3. Apply ICA
        downsampled_data = self._downsample(cleaned_data)                       # 4. Downsampling
        normalized_data = self._min_max_normalization_per_signal(downsampled_data) # 5. Normalization

        print(f"Dataset shape: {normalized_data.shape}, Labels shape: {all_labels.shape}")

        return normalized_data, all_labels

    def __getitem__(self, idx):
        adjacency_matrix = torch.tensor(GMatrixCalculator._compute_G_matrix(self.data[idx]), dtype=torch.float32)
        feature_matrix = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature_matrix, adjacency_matrix, label - 1

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    dataset = EEGDataLoader("Processed_Data_", "AK-SREP", "reading")
