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
4. **Understand the training process** with clear explanations and code-style pseudocode

## 🧠 Model Architecture

This project uses the real model from `train.py`:

- Input: 6 values (`Var1`, one-hot operator, `Var2`)
- Hidden layers: 32 neurons → 32 neurons → 32 neurons
- Output: 1 value

Full shape: `Input[6] → Linear[32] → ReLU → Linear[32] → ReLU → Linear[32] → ReLU → Linear[1]`

---

## 🚀 Quick Start

### 1. Train the Model
```bash
pip install torch numpy pandas
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
- **Concepts:** [http://localhost:8000/concepts.html](http://localhost:8000/concepts.html)

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `train.py` | PyTorch training script (generates model + timeline snapshots) |
| `calculator_data.csv` | 10,000 training examples (arithmetic operations) |
| `model_weights.json` | The trained model — just arrays of weights and biases |
| `training_timeline.json` | Weight snapshots: Initial → Row 1 → Row 2 → Final |
| `index.html` | Calculator page with step-by-step matrix trace |
| `tutorial.html` | Educational page explaining training and architecture |
| `concepts.html` | Explorer page for AI topics and encoding |
| `app.js` | Calculator page logic (matrix math in pure JS) |
| `tutorial.js` | Tutorial page logic (timeline viewer + diff) |
| `concepts.js` | Concepts page interactivity |
| `style.css` | Shared dark-mode glassmorphic styling |

## 🚧 What Changed

- Updated the documentation to reflect the real model architecture from `train.py`.
- Added clearer explanations of one-hot encoding, normalization, weights, biases, and training examples.
- Included a stronger tutorial path for people familiar with C/C++ and step-by-step logic.

## 🧠 Key Insight

The entire "AI model" is just a set of numbers in arrays.  
Training is the process of nudging those numbers until input data produces the right answer.  
The web app runs the same math in JavaScript, so you can inspect every step.

---

## 📜 License

MIT — Feel free to use this for teaching, presentations, or learning!
