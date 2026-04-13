const OP_DICT = { 'p': 0, 's': 1, 'm': 2, 'd': 3 };

let weights = null;

async function loadWeights() {
    try {
        const response = await fetch('model_weights.json');
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
        "An AI model isn't magic; it's literally just a bunch of memory grids (called Matrices) filled with numbers! These numbers are called <strong>Weights</strong> and <strong>Biases</strong>. During training, the computer tweaked these numbers thousands of times until they accurately represented mathematical rules. Here are the EXACT numbers holding the memory inside our model right now:",
        `${renderMatrix("Weight 1 [6x8]", weights.fc1_w, true)}
         ${renderMatrix("Weight 2 [8x8]", weights.fc2_w, true)}
         ${renderMatrix("Weight Out [8x1]", weights.out_w, true)}`
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
        `${renderMatrix("Input Matrix X [1x6]", currentX)}`
    );
    
    // Layer 1 Linear
    const z1 = matmul(currentX, weights.fc1_w, weights.fc1_b);
    html += generateStepHTML(
        "🔢 4. Step 2: Layer 1 'Thinking' (Multiplication)",
        `The AI is now processing your request. It takes your Input Matrix and multiplies it against the first weight memory grid. Think of this like asking 8 different questions about your numbers simultaneously. Finally, it adds the <strong>Bias</strong> (a little padding adjustment) to get the raw thoughts (Z1)!`,
        `${renderMatrix("X", currentX)} <div class="op-char">·</div> ${renderMatrix("W1^T", weights.fc1_w, true)} <div class="op-char">+</div> ${renderMatrix("B1", weights.fc1_b)} <div class="op-char">=</div> ${renderMatrix("Z1", z1)}`
    );
    
    // Layer 1 Activation (ReLU)
    const a1 = relu(z1);
    html += generateStepHTML(
        "🛡️ 5. Step 3: Layer 1 Activation (Making Choices)",
        `Look closely at the <strong>Z1</strong> matrix above. See any negative numbers? The AI uses a filter called <strong>ReLU</strong> (Rectified Linear Unit), which strictly deletes any number below zero. This is how the AI "makes choices" and drops useless information!`,
        `<div style="color:var(--text-secondary); font-size: 1.2rem;">ReLU (</div>${renderMatrix("Z1", z1)}<div style="color:var(--text-secondary); font-size: 1.2rem;">)</div><div class="op-char">=</div>${renderMatrix("A1 (Choices)", a1)}`
    );
    
    currentX = a1;
    
    // Layer 2 Linear
    const z2 = matmul(currentX, weights.fc2_w, weights.fc2_b);
    html += generateStepHTML(
        "🧠 6. Step 4: Layer 2 Deep Thinking",
        `Now the AI takes the filtered choices (A1) from the previous step, and rubs them against the next memory grid (W2). This combination creates complex logic, figuring out exactly what math operation you wanted.`,
        `${renderMatrix("A1", currentX)} <div class="op-char">·</div> ${renderMatrix("W2^T", weights.fc2_w, true)} <div class="op-char">+</div> ${renderMatrix("B2", weights.fc2_b)} <div class="op-char">=</div> ${renderMatrix("Z2", z2)}`
    );
    
    // Layer 2 Activation
    const a2 = relu(z2);
    html += generateStepHTML(
        "🛡️ 7. Step 5: Layer 2 Activation",
        `Again, any negative 'bad thoughts' are instantly clamped to zero!`,
        `<div style="color:var(--text-secondary); font-size: 1.2rem;">ReLU (</div>${renderMatrix("Z2", z2)}<div style="color:var(--text-secondary); font-size: 1.2rem;">)</div><div class="op-char">=</div>${renderMatrix("A2 (Choices)", a2)}`
    );
    
    currentX = a2;
    
    // Output Layer
    const out = matmul(currentX, weights.out_w, weights.out_b);
    html += generateStepHTML(
        "🎯 8. Step 6: Final Prediction (Output)",
        `We made it! The AI takes its profound 8-number thought (A2) and squeezes it down into a single final answer by multiplying it via the final small weight grid. The resulting number is what the AI predicts the mathematical answer is!`,
        `${renderMatrix("A2", currentX)} <div class="op-char">·</div> ${renderMatrix("Output Weights", weights.out_w, true)} <div class="op-char">+</div> ${renderMatrix("Output Bias", weights.out_b)} <div class="op-char">=</div> ${renderMatrix("Final Prediction", out)}`
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
        }, i * 600); // 600ms stagger between math steps
        
        cards[i].style.transform = 'translateY(20px)';
    }
    
    document.getElementById('final-result').innerText = out[0].toFixed(4);
}

document.getElementById('calc-btn').addEventListener('click', visualize);

// Initial Load
loadWeights();
