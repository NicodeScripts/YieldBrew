# ☕ YieldBrew

YieldBrew is a simple AI-powered tool designed to estimate coffee yield (in kg/ha) based on environmental and agricultural conditions. It was built as a lightweight proof-of-concept for a hackathon focused on sustainable and tech-enabled coffee farming.

---

## 🎯 What It Does

YieldBrew uses a regression-based neural network trained on synthetic agricultural data to predict coffee yield from various input features like rainfall, temperature, fertilizer usage, soil pH, and more.

The model is wrapped in a simple interactive Gradio interface for easy demonstration and experimentation.

---

## 🚀 How to Run

> **Prerequisites:** Python 3.9+, PyTorch, scikit-learn, Gradio, NumPy, pandas

1. Clone the repo:
   git clone https://github.com/NicodeScripts/YieldBrew.git
   cd YieldBrew
2. Install Dependencies:
   pip install -r requirements.txt
3. Train Model
  cd model
  jupyter notebook -> model_train.ipynb
4. Run Demo
   cd demo
   python gradio_app.py

---
