import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch
import itertools
import random

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

def get_desc(f):
        from pymatgen.core import Structure
        from robocrys import StructureCondenser, StructureDescriber
        structure = Structure.from_file(f) # other file formats also supported

        # alternatively, uncomment the lines below to use the MPRester object
        # to fetch structures from the Materials Project database
        # from mp_api.client import MPRester
        # structure = MPRester(api_key=None).get_structure_by_material_id("mp-856")

        condenser = StructureCondenser()
        describer = StructureDescriber()

        condensed_structure = condenser.condense_structure(structure)
        # description = describer.describe(condensed_structure)
        # return description
        try:
            description = describer.describe(condensed_structure)
            return description
        except TypeError as e:
            print(f"Warning: Description generation failed for structure {f}. Details: {e}")
            return {"error": "description_failed", "file": f}

def delete_random_chars(text, prob=0.1):
        return ''.join([char for char in text if random.random() > prob])
def replace_random_words(text, prob=0.1, replacement_words=["foo", "bar", "baz"]):
        words = text.split()
        return ' '.join([word if random.random() > prob else random.choice(replacement_words) for word in words])
def add_random_punctuation(text, prob=0.1):
        punctuation = ['.', ',', '!', '?', ';', ':']
        return ''.join([char if random.random() > prob else char + random.choice(punctuation) for char in text])

def combined_text(text):
        text = delete_random_chars(text, prob=0.6)
        text = replace_random_words(text, prob=0.6)
        text = add_random_punctuation(text, prob=0.6)
        return text

def plot_tsne(encoded_features, labels, prop, ind):

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt


        encoded_features = np.array(encoded_features.detach().cpu().numpy())
        # labels =np.array(labels)

        # if prop=='formation energy':
        #     labels=df['formation_energy']
        # elif prop=='fermi energy':
        #     labels=df['fermi_energy']
        # elif prop =='band gap':
        #     labels=df['band_gap']
        # elif prop =='energy_above_hull':
        #     labels=df['energy_above_hull']

        tsne = TSNE(n_components=2, random_state=42, perplexity = 5)
        tsne_features = tsne.fit_transform(encoded_features)

        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c= labels, cmap='viridis')
        plt.colorbar(label= 'Formation Energy')
        plt.title(f't-SNE Visualization of {prop} Encoded Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(f'./Results/Output/tsne_{prop}_{ind}.png', dpi=300)


def plot_tsne_all(supervised_emb, bert_emb, combined_emb, labels, prop):
        plot_tsne(encoded_features=supervised_emb, labels= labels, prop=prop, ind="Supervised" )
        plot_tsne(encoded_features=bert_emb,labels=labels, prop=prop, ind="Bert")
        plot_tsne(encoded_features=combined_emb, labels=labels, prop=prop, ind= "Combined")

def investigate_data(model, outputs):
        lbls =[]
        activations = {}

# Hook function to capture outputs
        def get_hook(layer_name):
                activations[layer_name] =[]
                def hook_fn(module, input, output):
                    activations[layer_name].append(output.detach())  # Store the output
                return hook_fn

#Register the hook on the embedding layer
        hook_supervised = model.supervised_proj.register_forward_hook(get_hook('supervised_proj'))
        hook_bert = model.transformer_proj.register_forward_hook(get_hook('transformer_proj'))
        hook_combined = model.attention_combiner.fc_out.register_forward_hook(get_hook('comb_proj'))
        lbls= list(itertools.chain.from_iterable(lbls))

        for name in activations:
            activations[name] = torch.cat(activations[name], dim=0).squeeze()
            
        plot_tsne_all(supervised_emb=activations['supervised_proj'], 
                    bert_emb= activations['transformer_proj'], 
                    combined_emb= activations['comb_proj'], 
                    labels=lbls,
                    prop='formation_energy')
        
        np.savetxt('./Results/Output/Jarvis_fe_combined.txt',outputs)
        
        hook_supervised.remove()
        hook_bert.remove()
        hook_combined.remove()