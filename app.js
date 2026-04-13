const OP_DICT = { 'p': 0, 's': 1, 'm': 2, 'd': 3 };

let weights = null;

async function loadWeights() {
    try {
        const response = await fetch('model_weights.json?v=' + Date.now());
        weights = await response.json();
    } catch (e) {
        console.error("Failed to load weights.", e);
        document.getElementById('math-trace').innerHTML = `<div class="placeholder-text" style="color: red;">Failed to load model_weights.json. Is the training done?</div>`;
    }
}

// Math Helpers
function transpose(m) {
    if (m[0].length === undefined) return [m]; // 1D to 2D row
    return m[0].map((_, i) => m.map(row => row[i]));
}

function relu(arr) {
    return arr.map(x => Math.max(0, x));
}

function matmul(input, wBase, b) {
    // pyTorch linear is y = x @ W.T + b
    // wBase shape: [out_features, in_features]. input is length in_features
    let result = new Array(wBase.length).fill(0);
    for (let i = 0; i < wBase.length; i++) {
        let sum = b[i];
        for (let j = 0; j < input.length; j++) {
            sum += input[j] * wBase[i][j];
        }
        result[i] = sum;
    }
    return result;
}

// UI Formatting helpers
function formatNum(num) {
    if (Math.abs(num) < 0.001 && num !== 0) return "0.00";
    return num.toFixed(2);
}

function getNumClass(num) {
    if (num === 0) return "cell-zero";
    if (num > 0) return "cell-pos";
    return "cell-neg";
}

function renderMatrix(name, matrix, isTranspose=false) {
    // matrix is expected as 2D array
    let rows = matrix;
    if (matrix[0] === undefined || matrix[0].length === undefined) {
        // It's a 1D vector. Let's render as 1xN by default
        rows = [matrix];
    }
    
    if (isTranspose) {
        rows = transpose(rows);
    }
    
    let html = `<div class="matrix-wrapper"><div class="matrix-title">${name}</div><div class="matrix">`;
    for (let r = 0; r < rows.length; r++) {
        html += `<div class="matrix-row">`;
        for (let c = 0; c < rows[r].length; c++) {
            let val = rows[r][c];
            html += `<div class="matrix-cell ${getNumClass(val)}">${formatNum(val)}</div>`;
        }
        html += `</div>`;
    }
    html += `</div></div>`;
    return html;
}

function generateStepHTML(stepName, explanation, eqHTML) {
    return `
        <div class="step-card">
            <div class="step-header">${stepName}</div>
            <div class="explanation-box">${explanation}</div>
            <div class="math-eq">
                ${eqHTML}
            </div>
        </div>
    `;
}

// Main Calculate Logic
async function visualize() {
    if (!weights) {
        await loadWeights();
        if (!weights) return;
    }
    
    const var1 = parseFloat(document.getElementById('var1').value);
    const var2 = parseFloat(document.getElementById('var2').value);
    const op = document.getElementById('op').value;
    
    const opVec = [0, 0, 0, 0];
    if (OP_DICT[op] !== undefined) opVec[OP_DICT[op]] = 1;
    
    const x = [var1, ...opVec, var2];
    
    const traceDiv = document.getElementById('math-trace');
    traceDiv.innerHTML = ''; // clear
    
    let currentX = x;
    
    // Step 0: Explain Model
    let html = generateStepHTML(
        "🧠 1. This is our Model!",
        "An AI model isn't magic; it's literally just a bunch of memory grids (called Matrices) filled with numbers! These numbers are called <strong>Weights</strong> and <strong>Biases</strong>. During training, the computer tweaked these numbers thousands of times until they accurately represented mathematical rules. Our model has <strong>3 hidden layers with 32 neurons each</strong>.",
        `${renderMatrix("Weight 1 [6×32]", weights.fc1_w, true)}
         ${renderMatrix("Weight 2 [32×32]", weights.fc2_w, true)}
         ${renderMatrix("Weight 3 [32×32]", weights.fc3_w, true)}
         ${renderMatrix("Weight Out [32×1]", weights.out_w, true)}`
    );

    // Step 1: Explain Input
    html += generateStepHTML(
        "🧑‍💻 2. This is your Input",
        `You asked the computer to calculate: <strong>${var1} ${op} ${var2}</strong>. AI models cannot read text symbols like '+' or '-'. They only eat raw arrays of numbers!`,
        `<div style="font-size:1.5rem; color:#fff;">User Typed: <span style="color:var(--accent)">${var1}</span> <span style="color:#aaa">${document.getElementById('op').options[document.getElementById('op').selectedIndex].text}</span> <span style="color:var(--accent)">${var2}</span></div>`
    );

    // Step 2: Parsing token
    html += generateStepHTML(
        "⚙️ 3. Step 1: Converting to Math Tokens",
        `We must parse your input into a numeric grid. We put <strong>${var1}</strong> at the start, and <strong>${var2}</strong> at the end. For the operator, we use a 4-slot switch called "One-Hot Encoding" [Plus, Minus, Multiply, Divide]. Since you chose your operator, we put a <code>1</code> in its slot and <code>0</code> in the others!`,
        `${renderMatrix("Input Matrix X [1×6]", currentX)}`
    );
    
    // Layer 1 Linear
    const z1 = matmul(currentX, weights.fc1_w, weights.fc1_b);
    html += generateStepHTML(
        "🔢 4. Step 2: Layer 1 'Thinking'",
        `The AI multiplies your Input against the first weight grid (6×32). Think of this like asking 32 different questions about your numbers simultaneously. The <strong>Bias</strong> is a padding adjustment added to each result.`,
        `${renderMatrix("X", currentX)} <div class="op-char">·</div> ${renderMatrix("W1^T", weights.fc1_w, true)} <div class="op-char">+</div> ${renderMatrix("B1", weights.fc1_b)} <div class="op-char">=</div> ${renderMatrix("Z1", z1)}`
    );
    
    // Layer 1 Activation (ReLU)
    const a1 = relu(z1);
    html += generateStepHTML(
        "🛡️ 5. Step 3: Layer 1 Activation (ReLU)",
        `The AI uses <strong>ReLU</strong> to delete any negative numbers. This is how the AI "makes choices" — it drops useless information!`,
        `<div style="color:var(--text-secondary); font-size: 1.2rem;">ReLU (</div>${renderMatrix("Z1", z1)}<div style="color:var(--text-secondary); font-size: 1.2rem;">)</div><div class="op-char">=</div>${renderMatrix("A1", a1)}`
    );
    
    currentX = a1;
    
    // Layer 2 Linear
    const z2 = matmul(currentX, weights.fc2_w, weights.fc2_b);
    html += generateStepHTML(
        "🧠 6. Step 4: Layer 2 Deep Thinking",
        `The AI takes the filtered choices (A1) and multiplies them against the second weight grid. This combination builds more complex mathematical logic.`,
        `${renderMatrix("A1", currentX)} <div class="op-char">·</div> ${renderMatrix("W2^T", weights.fc2_w, true)} <div class="op-char">+</div> ${renderMatrix("B2", weights.fc2_b)} <div class="op-char">=</div> ${renderMatrix("Z2", z2)}`
    );
    
    // Layer 2 Activation
    const a2 = relu(z2);
    html += generateStepHTML(
        "🛡️ 7. Step 5: Layer 2 Activation",
        `Again, negative signals are clamped to zero!`,
        `<div style="color:var(--text-secondary); font-size: 1.2rem;">ReLU (</div>${renderMatrix("Z2", z2)}<div style="color:var(--text-secondary); font-size: 1.2rem;">)</div><div class="op-char">=</div>${renderMatrix("A2", a2)}`
    );
    
    currentX = a2;
    
    // Layer 3 Linear
    const z3 = matmul(currentX, weights.fc3_w, weights.fc3_b);
    html += generateStepHTML(
        "🧠 8. Step 6: Layer 3 — Even Deeper",
        `A third layer of processing! More neurons means more "rulers" to approximate complex curves like multiplication. This is WHY bigger networks are smarter.`,
        `${renderMatrix("A2", currentX)} <div class="op-char">·</div> ${renderMatrix("W3^T", weights.fc3_w, true)} <div class="op-char">+</div> ${renderMatrix("B3", weights.fc3_b)} <div class="op-char">=</div> ${renderMatrix("Z3", z3)}`
    );
    
    const a3 = relu(z3);
    html += generateStepHTML(
        "🛡️ 9. Step 7: Layer 3 Activation",
        `Final filter before the output — clamping negatives one last time.`,
        `<div style="color:var(--text-secondary); font-size: 1.2rem;">ReLU (</div>${renderMatrix("Z3", z3)}<div style="color:var(--text-secondary); font-size: 1.2rem;">)</div><div class="op-char">=</div>${renderMatrix("A3", a3)}`
    );
    
    currentX = a3;
    
    // Output Layer
    const out_raw = matmul(currentX, weights.out_w, weights.out_b);
    
    // Denormalize: real_output = raw_output * y_std + y_mean
    const y_mean = weights.__metadata__.y_mean;
    const y_std = weights.__metadata__.y_std;
    const out_real = [out_raw[0] * y_std + y_mean];
    
    html += generateStepHTML(
        "🎯 10. Step 8: Final Prediction (Output + Denormalization)",
        `The AI squeezes its 32 thoughts down to a single number. But during training, we <strong>normalized</strong> the outputs (scaled them to a small range for easier learning). Now we reverse this: <code>real_answer = raw_output × ${y_std.toFixed(2)} + ${y_mean.toFixed(2)}</code>`,
        `${renderMatrix("A3", currentX)} <div class="op-char">·</div> ${renderMatrix("W_out^T", weights.out_w, true)} <div class="op-char">+</div> ${renderMatrix("B_out", weights.out_b)} <div class="op-char">=</div> ${renderMatrix("Raw Output", out_raw)} <div class="op-char">→</div> ${renderMatrix("Real Answer", out_real)}`
    );

    traceDiv.innerHTML = html;
    
    // Animate cards displaying in sequence
    const cards = document.querySelectorAll('.step-card');
    cards.forEach(c => c.style.opacity = '0');
    for(let i=0; i<cards.length; i++) {
        setTimeout(() => {
            cards[i].style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            cards[i].style.opacity = '1';
            cards[i].style.transform = 'translateY(0)';
            
            // Scroll to newly shown card
            if (i === cards.length - 1) {
                cards[i].scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
        }, i * 400);
        
        cards[i].style.transform = 'translateY(20px)';
    }
    
    document.getElementById('final-result').innerText = out_real[0].toFixed(2);
}

document.getElementById('calc-btn').addEventListener('click', visualize);

// Initial Load
loadWeights();
