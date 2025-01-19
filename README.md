# Multilingual Transformer Translation System

This repository contains a PyTorch-based implementation of a multilingual transformer model inspired by the **"Attention is All You Need"** paper. The model supports translation between:  
- **English ↔ Italian**  
- **English ↔ French**  

Additionally, it includes a Flask-based web interface for interacting with the trained models, featuring both text and voice input capabilities for seamless translation.

## Features
- **Custom Transformer Implementation**: Built from scratch, including self-attention and cross-attention mechanisms.
- **Flask Interface**: An intuitive web interface allowing users to input text or speak to receive instant translations.
- **Multilingual Support**: Pre-trained models for English ↔ Italian and English ↔ French translation.

## Model Weights
Due to storage constraints, the trained model weights are not included in this repository. You can access them on Hugging Face:  
- [English-Italian Transformer](https://huggingface.co/vrathi101/en_it_transformer)  
- [English-French Transformer](https://huggingface.co/vrathi101/en_fr_transformer)  
