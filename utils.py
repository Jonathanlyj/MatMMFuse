import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def summarize_dataset(dataset_combined, text, labels):
        """
        Generate a comprehensive summary of the dataset suitable for publication
        """
       
        
        # Basic dataset statistics
        total_samples = len(dataset_combined)
        
        # Label statistics
        label_stats = {
            'mean': np.mean(labels),
            'std': np.std(labels),
            'min': np.min(labels),
            'max': np.max(labels),
            'median': np.median(labels)
        }
        
        # Text statistics
        text_lengths = [len(t.split()) for t in text]
        text_stats = {
            'avg_length': np.mean(text_lengths),
            'std_length': np.std(text_lengths),
            'min_length': np.min(text_lengths),
            'max_length': np.max(text_lengths),
            'median_length': np.median(text_lengths)
        }
        
        # Create visualizations
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Label distribution
        plt.subplot(1, 2, 1)
        sns.histplot(labels, bins=30)
        plt.title('Distribution of Target Values')
        plt.xlabel('Target Value')
        plt.ylabel('Count')
        mpl.rcParams.update({'font.size': 14})
        # Plot 2: Text length distribution
        plt.subplot(1, 2, 2)
        sns.histplot(text_lengths, bins=30)
        plt.title('Distribution of Text Lengths')
        plt.xlabel('Number of Words')
        plt.ylabel('Count')
        mpl.rcParams.update({'font.size': 14})
        plt.tight_layout()
        plt.savefig('./Results/Output/dataset_summary.png', dpi=600)
        plt.close()

        # Create summary dictionary
        summary = {
            'Dataset Size': {
                'Total Samples': total_samples,
                'Training Samples': int(0.8 * total_samples),  # based on train_ratio
                'Testing Samples': int(0.2 * total_samples)
            },
            'Target Variable Statistics': label_stats,
            'Text Description Statistics': text_stats
        }
        
        return summary