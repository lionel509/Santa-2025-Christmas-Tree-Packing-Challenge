// WebSocket connection and state management
let ws = null;
let isPlaying = true;
let currentPuzzle = null;
let updateSpeed = 5;

// Canvas setup
const canvas = document.getElementById('tree-canvas');
const ctx = canvas.getContext('2d');

// Initialize puzzle selector
function initializePuzzleSelector() {
    const select = document.getElementById('puzzle-select');
    for (let i = 1; i <= 200; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `${i} trees`;
        select.appendChild(option);
    }
}

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/updates`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        document.getElementById('connection-status').textContent = 'Connected';
        addLog('Connected to server', 'info');
    };
    
    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleMessage(message);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        document.getElementById('connection-status').textContent = 'Disconnected';
        addLog('Disconnected from server', 'info');
        
        // Attempt reconnect
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        addLog('Connection error', 'error');
    };
}

// Handle incoming messages
function handleMessage(message) {
    if (!isPlaying && message.type !== 'init') {
        return; // Ignore updates when paused
    }
    
    switch (message.type) {
        case 'init':
            updateSummary(message.data);
            break;
        case 'state_update':
            if (currentPuzzle === message.puzzle_n) {
                updatePuzzleDisplay(message.data);
            }
            break;
        case 'progress':
            updateSummary(message.data);
            break;
        case 'improvement':
            handleImprovement(message);
            break;
        case 'puzzle_data':
            updatePuzzleDisplay(message.data);
            break;
    }
}

// Update summary statistics
function updateSummary(data) {
    if (data.total_score !== undefined) {
        document.getElementById('total-score').textContent = data.total_score.toFixed(2);
    }
    if (data.total_puzzles !== undefined) {
        document.getElementById('puzzles-completed').textContent = data.total_puzzles;
    }
    if (data.avg_score !== undefined) {
        document.getElementById('avg-score').textContent = data.avg_score.toFixed(6);
    }
    if (data.total_iterations !== undefined) {
        document.getElementById('total-iterations').textContent = data.total_iterations.toLocaleString();
    }
    if (data.uptime_seconds !== undefined) {
        document.getElementById('uptime').textContent = formatTime(data.uptime_seconds);
    }
}

// Handle improvement notification
function handleImprovement(message) {
    const improvement = message.improvement.toFixed(6);
    addLog(`Puzzle ${message.puzzle_n}: Score improved by ${improvement}`, 'improvement');
}

// Update puzzle display
function updatePuzzleDisplay(data) {
    document.getElementById('current-puzzle').textContent = data.n;
    document.getElementById('num-trees').textContent = data.trees ? data.trees.length : 0;
    document.getElementById('current-score').textContent = data.score.toFixed(6);
    document.getElementById('side-length').textContent = data.side_length.toFixed(4);
    document.getElementById('iterations').textContent = data.iterations || 0;
    
    // Draw trees
    if (data.trees) {
        drawTrees(data.trees, data.bounding_box);
    }
}

// Draw tree visualization
function drawTrees(trees, boundingBox) {
    // Clear canvas
    ctx.fillStyle = 'rgba(10, 10, 20, 0.8)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    if (trees.length === 0) return;
    
    // Calculate scale and offset
    const padding = 50;
    const canvasSize = Math.min(canvas.width, canvas.height) - 2 * padding;
    
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    trees.forEach(tree => {
        const poly = getTreePolygon(tree.x, tree.y, tree.deg);
        poly.forEach(([x, y]) => {
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        });
    });
    
    const dataWidth = maxX - minX;
    const dataHeight = maxY - minY;
    const dataSize = Math.max(dataWidth, dataHeight);
    const scale = canvasSize / (dataSize * 1.1); // 10% extra padding
    
    const offsetX = canvas.width / 2 - (minX + maxX) / 2 * scale;
    const offsetY = canvas.height / 2 - (minY + maxY) / 2 * scale;
    
    // Draw bounding square
    if (boundingBox) {
        ctx.strokeStyle = 'rgba(255, 107, 107, 0.5)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        
        const x = boundingBox.x * scale + offsetX;
        const y = boundingBox.y * scale + offsetY;
        const side = boundingBox.side * scale;
        
        ctx.strokeRect(x, canvas.height - y - side, side, side);
        ctx.setLineDash([]);
    }
    
    // Draw trees
    trees.forEach((tree, index) => {
        const poly = getTreePolygon(tree.x, tree.y, tree.deg);
        
        // Color gradient
        const hue = (index / trees.length) * 360;
        ctx.fillStyle = `hsla(${hue}, 70%, 60%, 0.7)`;
        ctx.strokeStyle = `hsla(${hue}, 70%, 40%, 1)`;
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        poly.forEach(([x, y], i) => {
            const canvasX = x * scale + offsetX;
            const canvasY = canvas.height - (y * scale + offsetY); // Flip Y axis
            
            if (i === 0) {
                ctx.moveTo(canvasX, canvasY);
            } else {
                ctx.lineTo(canvasX, canvasY);
            }
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    });
}

// Get tree polygon coordinates
function getTreePolygon(x, y, deg) {
    const TRUNK_W = 0.15;
    const TRUNK_H = 0.2;
    const BASE_W = 0.7;
    const MID_W = 0.4;
    const TOP_W = 0.25;
    const TIP_Y = 0.8;
    const TIER_1_Y = 0.5;
    const TIER_2_Y = 0.25;
    const BASE_Y = 0.0;
    const trunk_bottom_y = -TRUNK_H;
    
    const basePolygon = [
        [0.0, TIP_Y],
        [TOP_W / 2, TIER_1_Y],
        [TOP_W / 4, TIER_1_Y],
        [MID_W / 2, TIER_2_Y],
        [MID_W / 4, TIER_2_Y],
        [BASE_W / 2, BASE_Y],
        [TRUNK_W / 2, BASE_Y],
        [TRUNK_W / 2, trunk_bottom_y],
        [-TRUNK_W / 2, trunk_bottom_y],
        [-TRUNK_W / 2, BASE_Y],
        [-BASE_W / 2, BASE_Y],
        [-MID_W / 4, TIER_2_Y],
        [-MID_W / 2, TIER_2_Y],
        [-TOP_W / 4, TIER_1_Y],
        [-TOP_W / 2, TIER_1_Y],
    ];
    
    // Rotate and translate
    const rad = deg * Math.PI / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    
    return basePolygon.map(([px, py]) => {
        const rx = px * cos - py * sin;
        const ry = px * sin + py * cos;
        return [rx + x, ry + y];
    });
}

// Add log entry
function addLog(message, type = 'info') {
    const log = document.getElementById('activity-log');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    
    const timestamp = new Date().toLocaleTimeString();
    entry.textContent = `[${timestamp}] ${message}`;
    
    log.insertBefore(entry, log.firstChild);
    
    // Limit log entries
    while (log.children.length > 100) {
        log.removeChild(log.lastChild);
    }
}

// Format time
function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

// Request puzzle data
function requestPuzzle(n) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'get_puzzle', n: n }));
    } else {
        // Fetch via HTTP if WebSocket not ready
        fetch(`/api/puzzle/${n}`)
            .then(response => response.json())
            .then(data => updatePuzzleDisplay(data))
            .catch(error => {
                console.error('Error fetching puzzle:', error);
                addLog(`Error loading puzzle ${n}`, 'error');
            });
    }
}

// Export submission
function exportSubmission() {
    addLog('Exporting submission...', 'info');
    
    fetch('/api/export')
        .then(response => {
            if (response.ok) {
                return response.blob();
            } else {
                throw new Error('Export failed');
            }
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'submission.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            addLog('Submission exported successfully', 'improvement');
        })
        .catch(error => {
            console.error('Export error:', error);
            addLog('Export failed', 'error');
        });
}

// Save state
function saveState() {
    addLog('Saving state...', 'info');
    
    fetch('/api/save', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            addLog('State saved successfully', 'improvement');
        })
        .catch(error => {
            console.error('Save error:', error);
            addLog('Save failed', 'error');
        });
}

// Event listeners
document.getElementById('puzzle-select').addEventListener('change', (e) => {
    const n = parseInt(e.target.value);
    if (n > 0) {
        currentPuzzle = n;
        requestPuzzle(n);
        addLog(`Viewing puzzle ${n}`, 'info');
    }
});

document.getElementById('play-pause-btn').addEventListener('click', (e) => {
    isPlaying = !isPlaying;
    const btn = e.target;
    
    if (isPlaying) {
        btn.textContent = '⏸ Pause Updates';
        btn.classList.add('active');
        addLog('Updates resumed', 'info');
    } else {
        btn.textContent = '▶ Play Updates';
        btn.classList.remove('active');
        addLog('Updates paused', 'info');
    }
});

document.getElementById('speed-control').addEventListener('input', (e) => {
    updateSpeed = parseInt(e.target.value);
    document.getElementById('speed-value').textContent = `${updateSpeed}x`;
});

document.getElementById('export-btn').addEventListener('click', exportSubmission);
document.getElementById('save-btn').addEventListener('click', saveState);

// Initialize
initializePuzzleSelector();
connectWebSocket();

// Periodic updates
setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN && isPlaying) {
        // Refresh current summary
        fetch('/api/state')
            .then(response => response.json())
            .then(data => updateSummary(data))
            .catch(error => console.error('Error fetching state:', error));
    }
}, 2000);

// Initial load
setTimeout(() => {
    fetch('/api/state')
        .then(response => response.json())
        .then(data => {
            updateSummary(data);
            addLog('System initialized', 'info');
        });
}, 500);
