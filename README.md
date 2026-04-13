# 🔬 ModelXray — See Inside an AI Model

**An interactive, visual guide to how Neural Networks actually work — from raw matrices to final predictions.**

Most AI tutorials show abstract diagrams of circles and arrows. This project rips the cover off and shows you the **exact numbers, matrices, and math operations** that a real trained PyTorch model uses to make predictions.

---

## 🎯 What This Project Does

We built a simple Neural Network that works as a **calculator** (supports `+`, `-`, `*`, `/`).  
Then we built a web dashboard that lets you:

1. **See the real weight matrices** stored inside the model
2. **Trace any calculation step-by-step** through every layer (dot products, bias additions, ReLU activations)
3. **Watch how the matrices evolved** from random noise into a trained calculator

## 🖥️ Live Demo

### Calculator Page — Trace the Math
Enter any calculation and watch the AI process it layer by layer, showing every matrix multiplication:

> `10 + 5` → Input Matrix → W1 dot product → ReLU → W2 dot product → ReLU → Output

### Tutorial Page — How We Built It
An interactive walkthrough explaining:
- Why we chose **One-Hot Encoding** for operators
- Why the model has **8 hidden neurons**
- A **timeline viewer** showing how weights evolve from random → trained

---

## 🚀 Quick Start

### 1. Train the Model
```bash
pip install torch numpy pandas matplotlib networkx
python train.py
```

This generates:
- `model_weights.json` — The trained AI model (just a JSON file!)
- `training_timeline.json` — Snapshots of weights at different training stages

### 2. Launch the Visualizer
```bash
python -m http.server 8000
```

Then open:
- **Calculator:** [http://localhost:8000](http://localhost:8000)
- **Tutorial:** [http://localhost:8000/tutorial.html](http://localhost:8000/tutorial.html)

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `train.py` | PyTorch training script (generates model + timeline snapshots) |
| `calculator_data.csv` | 10,000 training examples (arithmetic operations) |
| `model_weights.json` | The trained model — just arrays of numbers! |
| `training_timeline.json` | Weight snapshots: Initial → Row 1 → Row 2 → Final |
| `index.html` | Calculator page with step-by-step matrix trace |
| `tutorial.html` | Educational page explaining the architecture |
| `app.js` | Calculator page logic (matrix math in pure JS) |
| `tutorial.js` | Tutorial page logic (timeline viewer + diff) |
| `style.css` | Shared dark-mode glassmorphic styling |

## 🧠 Key Insight

The entire "AI model" is just a **JSON file** containing arrays of decimal numbers.  
No PyTorch needed to run it — our web app performs the matrix math in plain JavaScript.  
You could port this model to C++, Excel, or even a calculator app.

---

## 📜 License

MIT — Feel free to use this for teaching, presentations, or learning!
