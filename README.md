# MatMMFuse

MatMMFuse is a deep learning framework that combines multiple modalities (text and graph) for materials property prediction. It uses a fusion of CGCNN (Crystal Graph Convolutional Neural Network) and transformer-based models to learn from both structural and textual descriptions of materials.

## Features

- Multi-modal fusion of material structure and text descriptions
- CGCNN-based graph neural network for crystal structure processing
- Transformer-based text encoding using SciBERT
- Support for both regression and classification tasks
- Customizable training parameters and model architectures

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MatMMFuse.git
cd MatMMFuse
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
MatMMFuse/
├── data/                    # Data directory
│   ├── bulk_data/          # Crystal structure files
│   └── text_data/          # Text descriptions
├── matdeeplearn/           # Core ML modules
├── models/                 # Model implementations
├── utils/                  # Utility functions
├── config.yml             # Configuration file
├── main.py                # Main entry point
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Usage

1. Prepare your data:
   - Place crystal structure files in `data/bulk_data/`
   - Place text descriptions in `data/text_data/`
   - Create a CSV file with target properties

2. Configure the model:
   - Modify `config.yml` to set training parameters
   - Adjust model architecture if needed

3. Run training:
```bash
python main.py
```

## Configuration

The `config.yml` file contains all the necessary parameters for:
- Data processing
- Model architecture
- Training parameters
- Evaluation settings

## Dependencies

- PyTorch
- PyTorch Geometric
- Transformers (Hugging Face)
- ASE (Atomic Simulation Environment)
- scikit-learn
- NumPy
- pandas

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{bhattacharya2025matmmfusemultimodalfusionmodel,
      title={MatMMFuse: Multi-Modal Fusion model for Material Property Prediction}, 
      author={Abhiroop Bhattacharya and Sylvain G. Cloutier},
      year={2025},
      eprint={2505.04634},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.04634}, 
}
``` 
