# AI-Powered Documentation Generation System

## Project Structure
```
GENAI/
├── data/
│   ├── raw/              # Raw dataset files
│   └── processed/        # Processed data files
├── models/
│   ├── bpe/             # BPE tokenizer models
│   ├── word2vec/        # Word2Vec embeddings
│   └── lstm/            # LSTM model checkpoints
├── src/
│   ├── process_data.py  # Data processing scripts
│   └── train.py         # Training scripts
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place it in the `data/raw/` directory.

5. Process the data:
```bash
python src/process_data.py
```

6. Train the model:
```bash
python src/train.py
```

## Project Components

1. **Data Processing**
   - BPE Tokenization
   - Word2Vec Embeddings
   - Dataset preparation

2. **Model Architecture**
   - LSTM-based language model
   - Context-aware generation
   - Attention mechanism

3. **Training**
   - Batch processing
   - Gradient clipping
   - Early stopping

## Results

Training metrics and model performance will be saved in the respective model directories.

## Requirements
- Python 3.8+
- PyTorch
- NLTK
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm
