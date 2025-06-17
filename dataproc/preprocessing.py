from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



def clean_data(df, fs = 256, lowcut = 3, highcut = 30):
    
    """
    Clean the EEG data DataFrame by applying a bandpass filter, removing specific rows and columns, and scaling the data.

    Parameters:
        df : pd.DataFrame
            The DataFrame containing EEG data with columns for channels and labels.
        fs : int, optional
            The sampling frequency of the EEG data in Hz. Default is 256 Hz.
        lowcut : float, optional
            The low cutoff frequency for the bandpass filter in Hz. Default is 3 Hz.
        highcut : float, optional
            The high cutoff frequency for the bandpass filter in Hz. Default is 30 Hz.

    Returns:
        pd.DataFrame
            The cleaned DataFrame with filtered and scaled EEG data, excluding specified rows and columns.
    """

    # Remove rows where label == 'idle'
    df = df[df['label'] != 'idle']
    # Remove columns 'sequence', 'battery', and 'flags'
    df = df.drop(columns=['sequence', 'battery', 'flags'])
    # Remove rows where label is 'rest' or 'start'
    df = df[~df['label'].isin(['rest', 'start'])]
    # Remove the column 'timestamp'
    df = df.drop(columns=['timestamp'])  
    
    nyquist = 0.5 * fs # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    # Create Butterworth bandpass filter
    b, a = butter(N=4, Wn=[low, high], btype='band')
    # Apply filter to all channel columns
    filtered_data = df.filter(like='channel_').apply(lambda x: filtfilt(b, a, x), axis=0)
    # Combine filtered data back into the DataFrame
    df = pd.concat([filtered_data, df[['label']]], axis=1)
    # Scale all channel columns using StandardScaler
    scaler = StandardScaler()
    channel_columns = [col for col in df.columns if col.startswith('channel_')]
    df[channel_columns] = scaler.fit_transform(df[channel_columns])
    return df



def split_data_target(df_clean): 
    
    """
    Split the cleaned DataFrame into features and target labels, encoding the labels as integers.
    Parameters:
        df_clean : pd.DataFrame
            The cleaned DataFrame containing EEG data and labels.        
    Returns:
        tuple
            A tuple containing the features (X) as a NumPy array and the encoded target labels (y_int) as a NumPy array.
    """
    X = df_clean.filter(like='channel_')
    y = df_clean['label']

    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)
    
    return X.to_numpy(), y_int

#use a LDA to reduce the dimension of X

def lda_process(X, y, n_components=None):
    """
    Reduce the dimensionality of the feature set using Linear Discriminant Analysis (LDA).
    
    Parameters:
        X : np.ndarray
            The feature set as a NumPy array.
        y : np.ndarray
            The target labels as a NumPy array.
        n_components : int, optional
            The number of components to keep. Default is None.
    
    Returns:
        tuple
            A tuple containing the reduced feature set (X_reduced) and the fitted LDA model.
        
    """
    lda = LDA(n_components=n_components)
    X_reduced = lda.fit_transform(X, y)
    return X_reduced, lda

def get_data_for_variance(required_variance = 0.95, *, ca, X):  
    """
    Get the number of components required to reach a specified variance threshold using LDA.
    Parameters:
        required_variance : float, optional
            The required variance threshold to reach. Default is 0.95 (95%).
        lda : LDA object
            The fitted Linear Discriminant Analysis model.
        X : np.ndarray
            The original feature set as a NumPy array.
    Returns:
        np.ndarray
            The reduced feature set containing only the components required to reach the specified variance threshold.
    """  
    
    cumsum = np.cumsum(ca.explained_variance_ratio_)
    n_components_required = np.argmax(cumsum >= required_variance) + 1
    return X[:, :n_components_required]

