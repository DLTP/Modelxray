// tutorial.js — Powers the training evolution page

let timeline = null;

async function loadTimeline() {
    try {
        const response = await fetch('training_timeline.json');
        timeline = await response.json();
        console.log("Training timeline loaded!");
        showSnapshot('initial');
        showDiff();
    } catch (e) {
        console.error("Failed to load timeline.", e);
        document.getElementById('snapshot-info').innerHTML = 
            '<div style="color: red; padding: 20px;">Failed to load training_timeline.json. Please run train.py first!</div>';
    }
}

// ---- Matrix rendering (shared logic) ----

function formatNum(num) {
    if (Math.abs(num) < 0.001 && num !== 0) return "0.00";
    return num.toFixed(2);
}

function getNumClass(num) {
    if (num === 0) return "cell-zero";
    if (num > 0) return "cell-pos";
    return "cell-neg";
}

function transpose(m) {
    if (m[0].length === undefined) return [m];
    return m[0].map((_, i) => m.map(row => row[i]));
}

function renderMatrix(name, matrix, isTranspose = false) {
    let rows = matrix;
    if (matrix[0] === undefined || matrix[0].length === undefined) {
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

// ---- Diff rendering (compare two weight sets) ----

function renderDiffMatrix(name, matrixBefore, matrixAfter, isTranspose = false) {
    let rowsBefore = matrixBefore;
    let rowsAfter = matrixAfter;
    
    if (matrixBefore[0] === undefined || matrixBefore[0].length === undefined) {
        rowsBefore = [matrixBefore];
        rowsAfter = [matrixAfter];
    }
    if (isTranspose) {
        rowsBefore = transpose(rowsBefore);
        rowsAfter = transpose(rowsAfter);
    }
    
    let html = `<div class="matrix-wrapper"><div class="matrix-title">${name} (Δ change)</div><div class="matrix">`;
    for (let r = 0; r < rowsBefore.length; r++) {
        html += `<div class="matrix-row">`;
        for (let c = 0; c < rowsBefore[r].length; c++) {
            let diff = rowsAfter[r][c] - rowsBefore[r][c];
            let cls = diff > 0.01 ? "cell-pos" : diff < -0.01 ? "cell-neg" : "cell-zero";
            let arrow = diff > 0.01 ? "↑" : diff < -0.01 ? "↓" : "·";
            html += `<div class="matrix-cell ${cls}">${arrow}${formatNum(diff)}</div>`;
        }
        html += `</div>`;
    }
    html += `</div></div>`;
    return html;
}

// ---- Snapshot viewer ----

function showSnapshot(stepKey) {
    if (!timeline) return;
    
    const snap = timeline.snapshots[stepKey];
    const infoDiv = document.getElementById('snapshot-info');
    const matDiv = document.getElementById('snapshot-matrices');
    
    // Show context info
    let infoHTML = `<div class="explanation-box">
        <strong>${snap.label}</strong><br>${snap.description}`;
    
    // If it's row1 or row2 show the training sample
    if (stepKey === 'after_row1') {
        const r = timeline.training_rows.row1;
        infoHTML += `<br><br>Training Sample: <code>[${r.input.map(v => v.toFixed(1)).join(', ')}]</code> → Expected: <strong>${r.expected}</strong>`;
    } else if (stepKey === 'after_row2') {
        const r = timeline.training_rows.row2;
        infoHTML += `<br><br>Training Sample: <code>[${r.input.map(v => v.toFixed(1)).join(', ')}]</code> → Expected: <strong>${r.expected}</strong>`;
    }
    
    infoHTML += `</div>`;
    infoDiv.innerHTML = infoHTML;
    
    // Render weight matrices
    matDiv.innerHTML = 
        renderMatrix("Weight 1 [6×8]", snap.fc1_w, true) +
        renderMatrix("Bias 1 [1×8]", snap.fc1_b) +
        renderMatrix("Weight 2 [8×8]", snap.fc2_w, true) +
        renderMatrix("Bias 2 [1×8]", snap.fc2_b) +
        renderMatrix("Weight Out [8×1]", snap.out_w, true) +
        renderMatrix("Bias Out [1×1]", snap.out_b);
    
    // Update active button
    document.querySelectorAll('.timeline-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.timeline-btn[data-step="${stepKey}"]`).classList.add('active');
}

function showDiff() {
    if (!timeline) return;
    
    const initial = timeline.snapshots.initial;
    const final = timeline.snapshots.final;
    const diffDiv = document.getElementById('diff-view');
    
    diffDiv.innerHTML = 
        renderDiffMatrix("ΔWeight 1", initial.fc1_w, final.fc1_w, true) +
        renderDiffMatrix("ΔBias 1", initial.fc1_b, final.fc1_b) +
        renderDiffMatrix("ΔWeight 2", initial.fc2_w, final.fc2_w, true) +
        renderDiffMatrix("ΔBias 2", initial.fc2_b, final.fc2_b) +
        renderDiffMatrix("ΔWeight Out", initial.out_w, final.out_w, true) +
        renderDiffMatrix("ΔBias Out", initial.out_b, final.out_b);
}

// ---- Event listeners ----

document.querySelectorAll('.timeline-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        showSnapshot(btn.dataset.step);
    });
});

// Load on page ready
loadTimeline();
