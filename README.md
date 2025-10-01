# Small LLM - Educational Language Model Implementation

A comprehensive educational implementation of a small-scale language model based on the GPT (Generative Pre-trained Transformer) architecture. This project demonstrates how to build and train a language model from scratch using PyTorch, making it perfect for learning the fundamentals of modern AI.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Specifications](#model-specifications)
- [Training Process](#training-process)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Educational Value](#educational-value)
- [Credits](#credits)

## 🎯 Overview

This project implements a small-scale language model (19 million parameters) that demonstrates the core concepts of modern transformer-based language models. Unlike large models like GPT-3 (175B parameters) or GPT-4, this implementation is designed to be:

- **Educational**: Clear, well-commented code for learning
- **Accessible**: Runs on a single GPU with 4GB+ VRAM
- **Complete**: Full training pipeline from data to inference
- **Practical**: Real text generation capabilities

## 🏗️ Architecture

### Core Components

The model follows the **GPT (Generative Pre-trained Transformer)** architecture:

```
Input Text → Tokenization → Embeddings → Transformer Blocks → Output Logits → Generated Text
```

### Detailed Architecture

1. **Token Embeddings**: Convert token IDs to dense vectors (384 dimensions)
2. **Position Embeddings**: Encode position information for each token
3. **Transformer Blocks** (7 layers):
   - **Multi-Head Attention**: 7 attention heads per layer
   - **Feed-Forward Networks**: 6x expansion (384 → 2304 → 384)
   - **Layer Normalization**: Before attention and feed-forward
   - **Residual Connections**: Skip connections for gradient flow
4. **Output Layer**: Linear projection to vocabulary size

### Attention Mechanism

- **Self-Attention**: Tokens attend to previous tokens in the sequence
- **Causal Masking**: Prevents looking at future tokens (autoregressive)
- **Multi-Head**: 7 parallel attention heads for different relationship types
- **Scaled Dot-Product**: Standard attention computation with scaling

## ✨ Features

### Model Features
- **19 Million Parameters**: Small enough for educational purposes
- **512 Token Context**: Reasonable sequence length for learning
- **7 Transformer Layers**: Sufficient depth for meaningful learning
- **7 Attention Heads**: Parallel attention processing
- **384 Embedding Dimensions**: Balanced representation size

### Training Features
- **Autoregressive Training**: Next token prediction
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Cosine annealing
- **Weight Decay**: L2 regularization for different parameter types
- **Checkpointing**: Save and resume training
- **Real-time Monitoring**: Weights & Biases integration

### Generation Features
- **Text Generation**: Autoregressive sampling
- **Interactive Mode**: Real-time text completion
- **Temperature Control**: Adjustable randomness
- **Context Window**: Sliding window for long sequences

## 📦 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended)
- **RAM**: 8GB+ system memory
- **Storage**: 2GB+ free space

### Software Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0+ (with CUDA support)
- **CUDA**: 11.8+ (for GPU acceleration)

### Python Dependencies
```
torch>=2.0.0
sentencepiece>=0.1.99
tqdm>=4.64.0
wandb>=0.15.0
ipdb>=0.13.0
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llm_course
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install sentencepiece tqdm wandb ipdb
   ```

3. **Download the dataset** (automatic on first run):
   - The notebook will automatically download the Wikipedia dataset
   - Pre-trained tokenizer and encoded data will be downloaded

## 💻 Usage

### Quick Start

1. **Open the notebook**:
   ```bash
   jupyter notebook small_llm_official.ipynb
   ```

2. **Run all cells** in order to:
   - Download and prepare the dataset
   - Initialize the model
   - Start training
   - Generate text

### Training Configuration

Key parameters you can adjust:

```python
# Architecture
batch_size = 8          # Adjust based on GPU memory
context = 512          # Sequence length
embed_size = 384       # Embedding dimensions
n_layers = 7          # Number of transformer blocks
n_heads = 7           # Number of attention heads

# Training
train_iters = 100000   # Number of training iterations
lr = 3e-4             # Learning rate
dropout = 0.05        # Dropout rate
weight_decay = 0.01   # L2 regularization
```

### Interactive Generation

Enable inference mode for interactive text generation:

```python
inference = True  # Set to True for interactive mode
```

## 📊 Model Specifications

| Component | Value | Description |
|-----------|-------|-------------|
| **Parameters** | 19M | Total trainable parameters |
| **Layers** | 7 | Transformer blocks |
| **Heads** | 7 | Attention heads per layer |
| **Embedding Size** | 384 | Token representation dimensions |
| **Context Length** | 512 | Maximum sequence length |
| **Vocabulary** | ~8K | Token vocabulary size |
| **Memory Usage** | ~4GB | GPU memory requirement |

### Comparison with Large Models

| Model | Parameters | Context | Training Cost |
|-------|------------|---------|---------------|
| **This Model** | 19M | 512 | $0 (single GPU) |
| **GPT-3** | 175B | 2048 | $12M+ |
| **GPT-4** | 1.7T+ | 8192+ | $100M+ |

## 🎓 Training Process

### Training Pipeline

1. **Data Preparation**:
   - Download Wikipedia dataset
   - Tokenize text using SentencePiece
   - Split into training/validation sets

2. **Model Initialization**:
   - Initialize weights with normal distribution
   - Set up optimizer with parameter groups
   - Configure learning rate scheduler

3. **Training Loop**:
   - Forward pass through model
   - Calculate cross-entropy loss
   - Backward pass with gradient clipping
   - Update parameters with AdamW optimizer
   - Adjust learning rate with cosine annealing

4. **Evaluation**:
   - Monitor training/validation loss
   - Generate sample text
   - Save best checkpoints

### Key Training Concepts

- **Autoregressive Training**: Model learns to predict next token
- **Teacher Forcing**: Use ground truth during training
- **Gradient Clipping**: Prevent exploding gradients
- **Learning Rate Scheduling**: Adaptive learning rate
- **Regularization**: Dropout and weight decay

## 📚 Dataset

### Source
- **Wikipedia**: Small subset of English Wikipedia articles
- **Size**: ~1M tokens (educational scale)
- **Format**: Plain text with diverse topics
- **Quality**: High-quality, well-structured text

### Preprocessing
- **Tokenization**: SentencePiece subword tokenization
- **Vocabulary**: ~8K tokens (balanced size)
- **Splitting**: 90% training, 10% validation
- **Encoding**: Pre-tokenized for efficiency

## 🔧 Implementation Details

### Key Technologies

- **PyTorch**: Deep learning framework
- **SentencePiece**: Subword tokenization
- **Weights & Biases**: Experiment tracking
- **CUDA**: GPU acceleration
- **Transformers**: Self-attention mechanism

### Architecture Highlights

1. **Causal Attention**: Prevents future token access
2. **Residual Connections**: Gradient flow preservation
3. **Layer Normalization**: Training stability
4. **Multi-Head Attention**: Parallel attention processing
5. **Feed-Forward Networks**: Non-linear transformations

### Training Optimizations

- **Mixed Precision**: bfloat16 for efficiency
- **Gradient Clipping**: Prevents gradient explosion
- **Parameter Grouping**: Different weight decay for different parameters
- **Learning Rate Scheduling**: Cosine annealing
- **Checkpointing**: Resume training capability

## 📈 Results

### Expected Performance

- **Initial Loss**: ~8-10 (random predictions)
- **Final Loss**: ~2-4 (meaningful predictions)
- **Training Time**: 2-4 hours on modern GPU
- **Memory Usage**: 4-6GB GPU memory

### Text Generation Quality

**Before Training** (random):
```
"The mountain in my city is xyz random tokens..."
```

**After Training** (coherent):
```
"The mountain in my city is a beautiful landmark that attracts many visitors..."
```

## 🎓 Educational Value

### Learning Objectives

This project teaches:

1. **Transformer Architecture**: Core components and interactions
2. **Attention Mechanisms**: Self-attention and multi-head attention
3. **Training Dynamics**: Loss functions, optimizers, and scheduling
4. **Text Generation**: Autoregressive sampling and decoding
5. **Model Scaling**: Parameter efficiency and memory management

### Key Concepts Covered

- **Neural Networks**: Deep learning fundamentals
- **Natural Language Processing**: Text understanding and generation
- **Machine Learning**: Training, validation, and evaluation
- **Software Engineering**: Code organization and best practices
- **Research Methods**: Experiment tracking and analysis

### Prerequisites

- **Python Programming**: Basic to intermediate level
- **Machine Learning**: Basic understanding of neural networks
- **Linear Algebra**: Matrix operations and transformations
- **Probability**: Basic probability and statistics

## 🏆 Credits

### Original Implementation
- **Author**: Javier Ideami
- **Website**: [ideami.com](https://ideami.com)
- **Notebook**: Official implementation #vj30

### Educational Enhancements
- **Comprehensive Comments**: Detailed explanations for every component
- **Architecture Documentation**: Clear breakdown of model components
- **Training Process**: Step-by-step training explanations
- **Best Practices**: Code organization and optimization techniques

### Acknowledgments
- **OpenAI**: GPT architecture inspiration
- **Google**: Transformer paper and SentencePiece
- **PyTorch Team**: Deep learning framework
- **Hugging Face**: Transformer implementations
- **Weights & Biases**: Experiment tracking platform

## 📄 License

This project is for educational purposes. Please respect the original author's work and use responsibly.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Model Architecture**: Experiment with different configurations
- **Training Techniques**: Advanced optimization methods
- **Evaluation Metrics**: Better performance measurement
- **Documentation**: Additional explanations and examples
- **Visualization**: Training progress and attention patterns

## 📞 Support

For questions and support:

- **Issues**: GitHub issues for bug reports
- **Discussions**: GitHub discussions for questions
- **Documentation**: Comprehensive comments in the notebook
- **Community**: Educational AI communities

---

**Happy Learning! 🚀**

*This project demonstrates that you don't need massive resources to understand and implement state-of-the-art language models. With the right approach and educational focus, anyone can build and train their own language model.*
