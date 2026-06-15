"""
train.py
--------
Entrypoint to train all models. Run this before launching the app.
Usage: python train.py
"""

from src.trainer import train_all

if __name__ == "__main__":
    train_all()
