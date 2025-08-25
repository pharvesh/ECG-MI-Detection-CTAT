"""
Data processing utilities for ECG signals
"""
import wfdb  # Added missing import
import numpy as np
import scipy.stats as st
from scipy import signal, ndimage  # Added ndimage
from scipy.interpolate import interp1d
import random
import torch
from torch.utils.data import Dataset
import ast
import pandas as pd
import os


def gaussian_noise(x, min_snr=0.01, max_snr=0.1):
    """Add Gaussian noise to ECG signal - EXACTLY as in original"""
    std = np.std(x)
    noise_std = random.uniform(min_snr * std, max_snr * std)
    noise = np.random.normal(0.0, noise_std, size=x.shape)
    return x + noise.astype(np.float32)


def signal_power(X):
    """Calculate signal power - EXACTLY as in original"""
    return np.mean((X - np.median(X)) ** 2)


def random_spikes(X, snr, period):
    """Add random spikes to ECG signal - EXACTLY as in original"""
    Power = signal_power(X)
    
    # Number of samples for spike filter
    N = np.random.randint(7, 13)
    
    # Initialize filter bank - EXACTLY as original
    F = np.zeros((5,))
    F[0] = np.random.uniform(-0.15, 0.25, 1)[0]
    F[1] = np.random.uniform(0.25, 0.5, 1)[0]
    F[2] = np.random.uniform(1, 2, 1)[0]
    F[3] = np.random.uniform(-0.5, 0.25, 1)[0]
    F[4] = np.random.uniform(0, 0.25, 1)[0]
    
    # Interpolate to number of samples
    interp = interp1d(np.linspace(0, 1, F.size), F, kind='quadratic')
    F = interp(np.linspace(0, 1, N))
    E = (F ** 2).sum()
    F = F / np.sqrt(E)
    
    SNRdb = snr + np.random.uniform(low=-snr / N, high=snr / N)
    T = period + np.random.randint(low=-period / 4, high=period / 4)
    P = np.random.randint(low=0, high=T)
    
    # Train of deltas
    Noise = np.zeros(X.shape)
    Noise[P::T] = 1
    
    # Compute real period
    Treal = Noise.size / Noise.sum()
    
    # Compute noise power and amplitude
    NoisePower = Power / 10 ** (SNRdb / 10.)
    Amplitude = np.sqrt(NoisePower * Treal)
    F = Amplitude * F
    
    # Convolution of deltas
    Noise = np.convolve(Noise, F, 'same')
    return Noise + X


def flip_signal(X):
    """Flip ECG signal - EXACTLY as in original"""
    return np.negative(X)


def butter_highpass(cutoff, fs, order=5):
    """Design butterworth highpass filter - EXACTLY as in original"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    """Apply butterworth highpass filter - EXACTLY as in original"""
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def normalize_ecg(x):
    """Normalize ECG signal - EXACTLY as in original norm() function"""
    xx = butter_highpass_filter(x, 0.5, 250)
    xx = (xx - np.mean(xx)) / np.std(xx)
    return xx


def resample_and_normalize(x, target_length=2500):
    """Resample and normalize - EXACTLY as in original resamp() function"""
    x = signal.resample(x, target_length, axis=1)
    x = st.zscore(x, axis=-1)
    return x


class ECGDataset(Dataset):
    """Custom dataset class - EXACTLY matching original BatchDataSet"""
    
    def __init__(self, X, Y, path):
        self.X = X
        self.Y = Y
        self.path = path
        
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # Load ECG data - EXACTLY as original
        _x = np.load(self.path + str(self.X[index]) + '.npy')
        _x = resample_and_normalize(_x)
        _y = self.Y[index]
        
        return _x, _y


def create_segmentation_masks(data_path, num_records=200):
    """Create segmentation masks - EXACTLY as in original code"""
    mask = np.zeros((num_records, 12, 4, 5000), dtype=int)
    fpoints = ['p', 'N', 't']  # EXACTLY as original
    leads = ['i', 'ii', 'iii', 'avl', 'avr', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']  # EXACTLY as original
    
    for i in range(num_records):
        record = wfdb.rdsamp(data_path + str(i + 1))  # EXACTLY as original
        
        for l in range(12):
            ann = wfdb.rdann(data_path + str(i + 1), leads[l])  # EXACTLY as original
            
            for w in range(len(fpoints)):
                fidwav = ann.sample[np.in1d(ann.symbol, [fpoints[w]])]
                
                seg = []
                for j in fidwav:
                    if j == ann.sample[-1]:
                        continue
                    
                    ind = np.where(ann.sample == j)[0][0]
                    onset = ann.sample[ind - 1]
                    offset = ann.sample[ind + 1]
                    seg.append([onset, offset])
                
                for k in range(len(seg)):
                    mask[i][l][w][seg[k][0]:seg[k][1]] = 1
            
            # Background mask - EXACTLY as original
            mask[i][l][3] = np.bitwise_or(np.bitwise_or(mask[i][l][0], mask[i][l][1]), mask[i][l][2]) ^ 1
    
    # Flatten mask for all leads - EXACTLY as original
    fullmask = []
    for i in range(len(mask)):
        for j in range(12):
            fullmask.append(mask[i][j])
    
    return fullmask


def load_and_preprocess_teacher_data(data_path, num_records=200):
    """Load and preprocess teacher data - EXACTLY as in original"""
    # Create segmentation masks
    fullmask = create_segmentation_masks(data_path, num_records)
    
    # Load ECG data - EXACTLY as original
    ecg_data = []
    for i in range(num_records):
        temp = wfdb.rdsamp(data_path + str(i + 1))[0].T
        for j in range(12):
            ecg_data.append(normalize_ecg(temp[j]))
    
    # Resample data - EXACTLY as original  
    ecg_data = signal.resample(ecg_data, 2500, axis=1)
    fullmask = ndimage.interpolation.zoom(fullmask, (1, 1, 0.5))
    
    return ecg_data, fullmask


def apply_augmentation_exactly_as_original(ecg_data, labels):
    """Apply data augmentation EXACTLY as in original"""
    aug_inputs = []
    aug_labels = []
    
    for i in range(len(ecg_data)):
        aug_inputs.append(ecg_data[i])
        aug_inputs.append(gaussian_noise(ecg_data[i]))
        aug_inputs.append(random_spikes(ecg_data[i], 25, 250))  # EXACTLY as original
        aug_inputs.append(flip_signal(ecg_data[i]))

        aug_labels.append(labels[i])
        aug_labels.append(labels[i])
        aug_labels.append(labels[i])
        aug_labels.append(labels[i])
    
    return aug_inputs, aug_labels32)


def signal_power(X):
    """Calculate signal power"""
    return np.mean((X - np.median(X)) ** 2)


def random_spikes(X, snr, period):
    """Add random spikes to ECG signal"""
    Power = signal_power(X)
    
    # Number of samples for spike filter
    N = np.random.randint(7, 13)
    
    # Initialize filter bank
    F = np.zeros((5,))
    F[0] = np.random.uniform(-0.15, 0.25, 1)[0]
    F[1] = np.random.uniform(0.25, 0.5, 1)[0]
    F[2] = np.random.uniform(1, 2, 1)[0]
    F[3] = np.random.uniform(-0.5, 0.25, 1)[0]
    F[4] = np.random.uniform(0, 0.25, 1)[0]
    
    # Interpolate to number of samples
    interp = interp1d(np.linspace(0, 1, F.size), F, kind='quadratic')
    F = interp(np.linspace(0, 1, N))
    E = (F ** 2).sum()
    F = F / np.sqrt(E)
    
    SNRdb = snr + np.random.uniform(low=-snr / N, high=snr / N)
    T = period + np.random.randint(low=-period / 4, high=period / 4)
    P = np.random.randint(low=0, high=T)
    
    # Train of deltas
    Noise = np.zeros(X.shape)
    Noise[P::T] = 1
    
    # Compute real period
    Treal = Noise.size / Noise.sum()
    
    # Compute noise power and amplitude
    NoisePower = Power / 10 ** (SNRdb / 10.)
    Amplitude = np.sqrt(NoisePower * Treal)
    F = Amplitude * F
    
    # Convolution of deltas
    Noise = np.convolve(Noise, F, 'same')
    return Noise + X


def flip_signal(X):
    """Flip ECG signal"""
    return np.negative(X)


def butter_highpass(cutoff, fs, order=5):
    """Design butterworth highpass filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    """Apply butterworth highpass filter"""
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def normalize_ecg(x, method='zscore'):
    """Normalize ECG signal"""
    if method == 'zscore':
        xx = butter_highpass_filter(x, 0.5, 250)
        xx = (xx - np.mean(xx)) / np.std(xx)
    elif method == 'minmax':
        xx = (x - np.min(x)) / (np.max(x) - np.min(x))
    else:
        xx = (x - np.mean(x)) / np.std(x)
    return xx


def resample_and_normalize(x, target_length=2500, sampling_rate=250):
    """Resample and normalize ECG data"""
    x = signal.resample(x, target_length, axis=1)
    x = st.zscore(x, axis=-1)
    return x


class ECGDataset(Dataset):
    """Custom dataset class for ECG data"""
    
    def __init__(self, X, Y, path, transform=None):
        self.X = X
        self.Y = Y
        self.path = path
        self.transform = transform
        
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # Load ECG data
        _x = np.load(self.path + str(self.X[index]) + '.npy')
        _x = resample_and_normalize(_x)
        _y = self.Y[index]
        
        if self.transform:
            _x = self.transform(_x)
        
        return _x, _y


def load_ptbxl_data(path):
    """Load and process PTB-XL dataset"""
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    agg = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg1 = agg[agg.diagnostic == 1]
    
    def agg_diag(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg1.index:
                tmp.append(agg1.loc[key].diagnostic_class)
        return list(set(tmp))
    
    Y['diagnostic_superclass'] = Y.scp_codes.apply(agg_diag)
    return Y


def prepare_mi_classification_data(Y, data_path, test_fold=10, validation_fold=9):
    """Prepare data for MI classification"""
    # Split data by folds
    Y_train = Y[(Y.strat_fold != test_fold) & (Y.strat_fold != validation_fold)].diagnostic_superclass
    Y_test = Y[(Y.strat_fold == test_fold)].diagnostic_superclass
    Y_validation = Y[(Y.strat_fold == validation_fold)].diagnostic_superclass
    
    # Get available patient files
    train_in = np.array(Y_train.index)
    validation_in = np.array(Y_validation.index)
    
    arr = os.listdir(data_path + "train/")
    patients = [int(os.path.splitext(i)[0]) for i in arr]
    pat = np.array(np.unique(patients))
    
    train_inp = np.intersect1d(pat, train_in)
    validation_inp = np.intersect1d(pat, validation_in)
    
    # Extract labels
    def extract_labels(patient_indices, Y_data):
        labels = []
        for i in patient_indices:
            labels.append(Y_data[i])
        return labels
    
    label_tr = extract_labels(train_inp, Y_train)
    label_val = extract_labels(validation_inp, Y_validation)
    
    # Separate MI and Normal cases
    def separate_classes(patient_indices, labels):
        mi_patients = []
        norm_patients = []
        
        for i, label in enumerate(labels):
            if 'MI' in label:
                mi_patients.append(patient_indices[i])
            elif 'NORM' in label:
                norm_patients.append(patient_indices[i])
        
        return norm_patients, mi_patients
    
    ynorm_tr, ymi_tr = separate_classes(train_inp, label_tr)
    ynorm_val, ymi_val = separate_classes(validation_inp, label_val)
    
    # Balance training data
    train_inputs = ynorm_tr[0:4500] + ymi_tr
    train_labels = np.concatenate((np.zeros(len(ynorm_tr[0:4500])), np.ones(len(ymi_tr))))
    
    validation_inputs = ynorm_val + ymi_val
    validation_labels = np.concatenate((np.zeros(len(ynorm_val)), np.ones(len(ymi_val))))
    
    return train_inputs, train_labels, validation_inputs, validation_labels


def apply_augmentation(ecg_data, labels, augment_types=['gaussian', 'spikes', 'flip']):
    """Apply data augmentation to ECG signals"""
    aug_inputs = []
    aug_labels = []
    
    for i, ecg in enumerate(ecg_data):
        # Original signal
        aug_inputs.append(ecg)
        aug_labels.append(labels[i])
        
        # Apply augmentations
        if 'gaussian' in augment_types:
            aug_inputs.append(gaussian_noise(ecg))
            aug_labels.append(labels[i])
        
        if 'spikes' in augment_types:
            aug_inputs.append(random_spikes(ecg, 25, 250))
            aug_labels.append(labels[i])
        
        if 'flip' in augment_types:
            aug_inputs.append(flip_signal(ecg))
            aug_labels.append(labels[i])
    
    return aug_inputs, aug_labels


def create_segmentation_masks(data_path, num_records=200):
    """Create segmentation masks for ECG delineation"""
    mask = np.zeros((num_records, 12, 4, 5000), dtype=int)
    fpoints = ['p', 'N', 't']
    leads = ['i', 'ii', 'iii', 'avl', 'avr', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    
    for i in range(num_records):
        record = wfdb.rdsamp(data_path + str(i + 1))
        
        for l in range(12):
            ann = wfdb.rdann(data_path + str(i + 1), leads[l])
            
            for w in range(len(fpoints)):
                fidwav = ann.sample[np.in1d(ann.symbol, [fpoints[w]])]
                
                seg = []
                for j in fidwav:
                    if j == ann.sample[-1]:
                        continue
                    
                    ind = np.where(ann.sample == j)[0][0]
                    onset = ann.sample[ind - 1]
                    offset = ann.sample[ind + 1]
                    seg.append([onset, offset])
                
                for k in range(len(seg)):
                    mask[i][l][w][seg[k][0]:seg[k][1]] = 1
            
            # Background mask
            mask[i][l][3] = np.bitwise_or(np.bitwise_or(mask[i][l][0], mask[i][l][1]), mask[i][l][2]) ^ 1
    
    # Flatten mask for all leads
    fullmask = []
    for i in range(len(mask)):
        for j in range(12):
            fullmask.append(mask[i][j])
    
    return fullmask
