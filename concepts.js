document.addEventListener('DOMContentLoaded', () => {
    // 🌳 SECTION: Tree vs Forest
    const treeSlider = document.getElementById('tree-slider');
    const treeDisplay = document.getElementById('tree-display');
    const treeCountVal = document.getElementById('tree-count-val');

    treeSlider.addEventListener('input', () => {
        const count = treeSlider.value;
        treeCountVal.textContent = count;
        treeDisplay.innerHTML = '';
        for (let i = 0; i < count; i++) {
            const tree = document.createElement('span');
            tree.className = 'tree-icon';
            tree.textContent = '🌳';
            tree.style.transitionDelay = `${i * 0.05}s`;
            treeDisplay.appendChild(tree);
        }
    });

    // 🏗️ SECTION: Matrix Viz (Network)
    const neuronSlider = document.getElementById('neuron-slider');
    const layerSlider = document.getElementById('layer-slider');
    const networkViz = document.getElementById('network-viz');
    const neuronCountVal = document.getElementById('neuron-count-val');
    const layerCountVal = document.getElementById('layer-count-val');
    const connCount = document.getElementById('conn-count');

    function updateNetwork() {
        const neurons = parseInt(neuronSlider.value);
        const layers = parseInt(layerSlider.value);
        neuronCountVal.textContent = neurons;
        layerCountVal.textContent = layers;
        networkViz.innerHTML = '';

        // Input layer (static 3 for this demo)
        const total = 3 + (neurons * layers) + 1;
        // Calculation for connections:
        // Input(3) -> Layer1(neurons) = 3 * neurons
        // LayerN(neurons) -> LayerN+1(neurons) = (layers-1) * (neurons * neurons)
        // LastLayer(neurons) -> Output(1) = neurons * 1
        let connections = (3 * neurons) + (layers > 1 ? (layers - 1) * (neurons * neurons) : 0) + (neurons * 1);
        connCount.textContent = connections.toLocaleString();

        // Create layers
        for (let l = 0; l <= layers + 1; l++) {
            const layerRow = document.createElement('div');
            layerRow.className = 'layer-row';
            
            let nodeCount = 0;
            if (l === 0) nodeCount = 3; // Input
            else if (l === layers + 1) nodeCount = 1; // Output
            else nodeCount = neurons; // Hidden

            for (let n = 0; n < nodeCount; n++) {
                const neuron = document.createElement('div');
                neuron.className = 'neuron';
                layerRow.appendChild(neuron);
            }
            networkViz.appendChild(layerRow);
        }
    }

    neuronSlider.addEventListener('input', updateNetwork);
    layerSlider.addEventListener('input', updateNetwork);
    updateNetwork();

    // 🏷️ SECTION: Encoding
    const btnLabel = document.getElementById('btn-label');
    const btnOneHot = document.getElementById('btn-onehot');
    const encodingDisplay = document.getElementById('encoding-display');
    const encodingInsight = document.getElementById('encoding-insight');

    const categories = ['Red', 'Green', 'Blue', 'Yellow', 'Purple'];

    function showLabelEncoding() {
        btnLabel.classList.add('active');
        btnOneHot.classList.remove('active');
        encodingDisplay.innerHTML = '';
        encodingInsight.innerHTML = 'Cells used per data point: <span class="stat-badge">1</span>';

        const row = document.createElement('div');
        row.style.marginBottom = '20px';
        row.innerHTML = '<strong>Data Point: </strong> <span style="color:var(--accent);">Green</span>';
        encodingDisplay.appendChild(row);

        const result = document.createElement('div');
        result.innerHTML = 'Encoded Value: <span class="encoding-cell">2</span>';
        encodingDisplay.appendChild(result);
    }

    function showOneHotEncoding() {
        btnOneHot.classList.add('active');
        btnLabel.classList.remove('active');
        encodingDisplay.innerHTML = '';
        encodingInsight.innerHTML = 'Cells used per data point: <span class="stat-badge">5</span>';

        const row = document.createElement('div');
        row.style.marginBottom = '20px';
        row.innerHTML = '<strong>Data Point: </strong> <span style="color:var(--accent);">Green</span>';
        encodingDisplay.appendChild(row);

        const result = document.createElement('div');
        result.innerHTML = 'Encoded Vector: ';
        categories.forEach((cat, idx) => {
            const cell = document.createElement('span');
            cell.className = 'encoding-cell' + (idx === 1 ? ' active' : '');
            cell.textContent = idx === 1 ? '1' : '0';
            result.appendChild(cell);
        });
        encodingDisplay.appendChild(result);
        
        const key = document.createElement('div');
        key.style.fontSize = '0.8rem';
        key.style.marginTop = '10px';
        key.style.color = 'var(--text-secondary)';
        key.textContent = '[Red, Green, Blue, Yellow, Purple]';
        encodingDisplay.appendChild(key);
    }

    btnLabel.addEventListener('click', showLabelEncoding);
    btnOneHot.addEventListener('click', showOneHotEncoding);
    showLabelEncoding(); // Initial
});
