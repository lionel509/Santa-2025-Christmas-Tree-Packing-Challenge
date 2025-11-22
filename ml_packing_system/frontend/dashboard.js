// Live Dashboard - Stock Market Style Updates
let ws = null;
let puzzleData = {};
let activeRange = { start: 1, end: 200 };
let updateQueue = [];
let isProcessing = false;

// Canvas cache for performance
const canvasCache = new Map();
// Throttling for visual updates
const lastDrawTime = new Map();
const isFetching = new Map(); // Track active fetches
const DRAW_THROTTLE_MS = 1000; // Only redraw every 1 second max per puzzle

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/updates`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        initializeDashboard();
    };
    
    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleMessage(message);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Handle incoming messages
function handleMessage(message) {
    switch (message.type) {
        case 'init':
            updateGlobalStats(message.data);
            break;
        case 'state_update':
            queueUpdate(message.puzzle_n, message.data);
            break;
        case 'progress':
            updateGlobalStats(message.data);
            break;
        case 'improvement':
            handleImprovement(message);
            break;
    }
}

// Queue updates to prevent overwhelming the UI
function queueUpdate(n, data) {
    updateQueue.push({ n, data });
    processQueue();
}

// Process update queue
async function processQueue() {
    if (isProcessing || updateQueue.length === 0) return;
    
    isProcessing = true;
    
    while (updateQueue.length > 0) {
        const { n, data } = updateQueue.shift();
        updatePuzzleCard(n, data);
        await new Promise(resolve => setTimeout(resolve, 10)); // Throttle
    }
    
    isProcessing = false;
}

// Initialize dashboard with all puzzles
async function initializeDashboard() {
    try {
        const response = await fetch('/api/puzzles/all');
        const data = await response.json();
        
        if (data.puzzles) {
            renderGrid(data.puzzles);
            data.puzzles.forEach(p => {
                puzzleData[p.n] = p;
            });
        }
        
        // Request detailed data for visible puzzles
        requestVisiblePuzzles();
    } catch (error) {
        console.error('Error initializing:', error);
    }
}

// Render puzzle grid
function renderGrid(puzzles) {
    const container = document.getElementById('grid-container');
    container.innerHTML = '';
    
    puzzles.forEach(puzzle => {
        if (puzzle.n >= activeRange.start && puzzle.n <= activeRange.end) {
            const card = createPuzzleCard(puzzle);
            container.appendChild(card);
        }
    });
}

// Create puzzle card
function createPuzzleCard(puzzle) {
    const card = document.createElement('div');
    card.className = 'puzzle-card';
    card.id = `puzzle-${puzzle.n}`;
    card.dataset.n = puzzle.n;
    
    card.innerHTML = `
        <div class="puzzle-header">
            <div>
                <div class="puzzle-number">Puzzle #${puzzle.n}</div>
                <div class="puzzle-trees">${puzzle.n} trees</div>
            </div>
        </div>
        <div class="canvas-wrapper">
            <canvas class="mini-canvas" id="canvas-${puzzle.n}" width="250" height="170"></canvas>
        </div>
        <div class="puzzle-stats">
            <div class="mini-stat">
                <div class="mini-stat-label">Score</div>
                <div class="mini-stat-value" id="score-${puzzle.n}">${puzzle.score.toFixed(6)}</div>
            </div>
            <div class="mini-stat">
                <div class="mini-stat-label">Side</div>
                <div class="mini-stat-value" id="side-${puzzle.n}">${puzzle.side_length.toFixed(4)}</div>
            </div>
            <div class="mini-stat">
                <div class="mini-stat-label">Iterations</div>
                <div class="mini-stat-value" id="iter-${puzzle.n}">${puzzle.iterations || 0}</div>
            </div>
            <div class="mini-stat">
                <div class="mini-stat-label">Trees</div>
                <div class="mini-stat-value">${puzzle.n}</div>
            </div>
        </div>
        <div class="last-update" id="update-${puzzle.n}">Loading...</div>
    `;
    
    // Add click to expand
    card.addEventListener('click', () => {
        window.open(`/app/index.html?puzzle=${puzzle.n}`, '_blank');
    });
    
    return card;
}

// Update puzzle card with new data
function updatePuzzleCard(n, data) {
    const card = document.getElementById(`puzzle-${n}`);
    if (!card) return;
    
    const oldScore = puzzleData[n]?.score || 0;
    const newScore = data.score;
    
    // Update data
    puzzleData[n] = data;
    
    // Update UI
    const scoreEl = document.getElementById(`score-${n}`);
    const sideEl = document.getElementById(`side-${n}`);
    const iterEl = document.getElementById(`iter-${n}`);
    const updateEl = document.getElementById(`update-${n}`);
    
    if (scoreEl) scoreEl.textContent = newScore.toFixed(6);
    if (sideEl) sideEl.textContent = data.side_length.toFixed(4);
    if (iterEl) iterEl.textContent = data.iterations || 0;
    if (updateEl) updateEl.textContent = `Updated ${new Date().toLocaleTimeString()}`;
    
    // Visual feedback
    card.classList.add('updating');
    setTimeout(() => card.classList.remove('updating'), 500);
    
    // Show improvement
    if (newScore < oldScore && oldScore > 0) {
        showImprovement(card, oldScore - newScore);
    }
    
    // Request full data and draw (throttled)
    const now = Date.now();
    const lastDraw = lastDrawTime.get(n) || 0;
    
    // Always draw if it's an improvement (newScore < oldScore) or if enough time passed
    if (newScore < oldScore || now - lastDraw > DRAW_THROTTLE_MS) {
        lastDrawTime.set(n, now);
        requestPuzzleData(n);
    }
}

// Show improvement badge
function showImprovement(card, improvement) {
    card.classList.add('improved');
    
    const badge = document.createElement('div');
    badge.className = 'improvement-badge';
    badge.textContent = `↓${improvement.toFixed(6)}`;
    card.appendChild(badge);
    
    setTimeout(() => {
        card.classList.remove('improved');
        badge.remove();
    }, 3000);
}

// Request puzzle data from API
async function requestPuzzleData(n) {
    if (isFetching.get(n)) return; // Skip if already fetching
    
    try {
        isFetching.set(n, true);
        const response = await fetch(`/api/puzzle/${n}`);
        const data = await response.json();
        
        if (data.trees) {
            drawMiniVisualization(n, data);
        }
    } catch (error) {
        console.error(`Error fetching puzzle ${n}:`, error);
    } finally {
        isFetching.set(n, false);
    }
}

// Draw mini visualization
function drawMiniVisualization(n, data) {
    const canvas = document.getElementById(`canvas-${n}`);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const trees = data.trees;
    
    // Clear
    ctx.fillStyle = 'rgba(10, 10, 20, 0.9)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    if (!trees || trees.length === 0) return;
    
    // Calculate bounds
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
    const padding = 10;
    const scale = (Math.min(canvas.width, canvas.height) - 2 * padding) / (dataSize * 1.1);
    
    const offsetX = canvas.width / 2 - (minX + maxX) / 2 * scale;
    const offsetY = canvas.height / 2 - (minY + maxY) / 2 * scale;
    
    // Draw trees (simplified for performance)
    trees.forEach((tree, index) => {
        const poly = getTreePolygon(tree.x, tree.y, tree.deg);
        const hue = (index / trees.length) * 360;
        
        ctx.fillStyle = `hsla(${hue}, 70%, 60%, 0.7)`;
        ctx.strokeStyle = `hsla(${hue}, 70%, 40%, 0.9)`;
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        poly.forEach(([x, y], i) => {
            const canvasX = x * scale + offsetX;
            const canvasY = canvas.height - (y * scale + offsetY);
            
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
    
    // Draw bounding box
    if (data.bounding_box) {
        ctx.strokeStyle = 'rgba(255, 107, 107, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        
        const x = data.bounding_box.x * scale + offsetX;
        const y = data.bounding_box.y * scale + offsetY;
        const side = data.bounding_box.side * scale;
        
        ctx.strokeRect(x, canvas.height - y - side, side, side);
        ctx.setLineDash([]);
    }
}

// Get tree polygon (same as before)
function getTreePolygon(x, y, deg) {
    const TRUNK_W = 0.15, TRUNK_H = 0.2;
    const BASE_W = 0.7, MID_W = 0.4, TOP_W = 0.25;
    const TIP_Y = 0.8, TIER_1_Y = 0.5, TIER_2_Y = 0.25, BASE_Y = 0.0;
    const trunk_bottom_y = -TRUNK_H;
    
    const base = [
        [0, TIP_Y], [TOP_W/2, TIER_1_Y], [TOP_W/4, TIER_1_Y],
        [MID_W/2, TIER_2_Y], [MID_W/4, TIER_2_Y], [BASE_W/2, BASE_Y],
        [TRUNK_W/2, BASE_Y], [TRUNK_W/2, trunk_bottom_y],
        [-TRUNK_W/2, trunk_bottom_y], [-TRUNK_W/2, BASE_Y],
        [-BASE_W/2, BASE_Y], [-MID_W/4, TIER_2_Y], [-MID_W/2, TIER_2_Y],
        [-TOP_W/4, TIER_1_Y], [-TOP_W/2, TIER_1_Y]
    ];
    
    const rad = deg * Math.PI / 180;
    const cos = Math.cos(rad), sin = Math.sin(rad);
    
    return base.map(([px, py]) => {
        const rx = px * cos - py * sin;
        const ry = px * sin + py * cos;
        return [rx + x, ry + y];
    });
}

// Handle improvement notification
function handleImprovement(message) {
    const n = message.puzzle_n;
    const card = document.getElementById(`puzzle-${n}`);
    
    if (card) {
        showImprovement(card, message.improvement);
    }
    
    // Show toast notification
    showToast(message.message || `Puzzle #${n} improved!`);
}

// Show toast notification
function showToast(message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
        <span class="toast-icon">🎉</span>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Remove after animation
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// Update global stats
function updateGlobalStats(data) {
    if (data.total_score !== undefined) {
        document.getElementById('total-score').textContent = data.total_score.toFixed(2);
    }
    if (data.avg_score !== undefined) {
        document.getElementById('avg-score').textContent = data.avg_score.toFixed(6);
    }
    if (data.total_puzzles !== undefined) {
        document.getElementById('puzzles-count').textContent = `${data.total_puzzles}/200`;
    }
    if (data.total_iterations !== undefined) {
        document.getElementById('total-iterations').textContent = data.total_iterations.toLocaleString();
    }
}

// Request visible puzzles
function requestVisiblePuzzles() {
    const cards = document.querySelectorAll('.puzzle-card');
    cards.forEach(card => {
        const n = parseInt(card.dataset.n);
        if (n) {
            setTimeout(() => requestPuzzleData(n), Math.random() * 2000);
        }
    });
}

// View toggle
document.querySelectorAll('.view-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        const view = btn.dataset.view;
        switch (view) {
            case 'all':
                activeRange = { start: 1, end: 200 };
                break;
            case 'small':
                activeRange = { start: 1, end: 50 };
                break;
            case 'medium':
                activeRange = { start: 51, end: 150 };
                break;
            case 'large':
                activeRange = { start: 151, end: 200 };
                break;
        }
        
        if (view !== 'active') {
            document.getElementById('range-start').value = activeRange.start;
            document.getElementById('range-end').value = activeRange.end;
            initializeDashboard();
        }
    });
});

// Range filter
document.getElementById('range-start').addEventListener('change', (e) => {
    activeRange.start = parseInt(e.target.value);
    initializeDashboard();
});

document.getElementById('range-end').addEventListener('change', (e) => {
    activeRange.end = parseInt(e.target.value);
    initializeDashboard();
});

// Grid size
document.getElementById('grid-size').addEventListener('change', (e) => {
    const container = document.getElementById('grid-container');
    container.style.gridTemplateColumns = `repeat(auto-fill, minmax(${e.target.value}, 1fr))`;
});

// Refresh all
document.getElementById('refresh-all').addEventListener('click', () => {
    requestVisiblePuzzles();
});

// Export
document.getElementById('export-btn').addEventListener('click', () => {
    window.location.href = '/api/export';
});

// Verify all puzzles
document.getElementById('verify-btn').addEventListener('click', async () => {
    const btn = document.getElementById('verify-btn');
    btn.disabled = true;
    btn.textContent = '⏳ Verifying...';
    
    try {
        const response = await fetch('/api/verify/summary');
        const data = await response.json();
        
        // Update verification stats
        document.getElementById('valid-count').textContent = data.valid_puzzles || 0;
        document.getElementById('collision-count').textContent = data.puzzles_with_collisions || 0;
        
        // Format min gap - show scientific notation if very small
        const minGap = data.min_gap || 0;
        let gapText;
        if (minGap < 0.0001) {
            gapText = minGap.toExponential(2);
        } else {
            gapText = minGap.toFixed(6);
        }
        document.getElementById('min-gap').textContent = gapText;
        
        // Show toast notification
        showToast(`✓ Verification Complete: ${data.valid_puzzles}/200 valid, ${data.puzzles_with_collisions} collisions`, 'success');
        
        btn.textContent = '✓ Verify All';
        btn.disabled = false;
    } catch (error) {
        console.error('Verification error:', error);
        showToast('❌ Verification failed', 'error');
        btn.textContent = '✓ Verify All';
        btn.disabled = false;
    }
});

// Auto-fetch verification summary on load and periodically
async function fetchVerificationSummary() {
    try {
        const response = await fetch('/api/verify/summary');
        const data = await response.json();
        
        document.getElementById('valid-count').textContent = data.valid_puzzles || 0;
        document.getElementById('collision-count').textContent = data.puzzles_with_collisions || 0;
        
        const minGap = data.min_gap || 0;
        let gapText;
        if (minGap < 0.0001) {
            gapText = minGap.toExponential(2);
        } else {
            gapText = minGap.toFixed(6);
        }
        document.getElementById('min-gap').textContent = gapText;
    } catch (error) {
        console.error('Error fetching verification summary:', error);
    }
}

// Initialize
connectWebSocket();

// Fetch verification summary on load
setTimeout(fetchVerificationSummary, 2000);

// Periodic refresh for visible puzzles and verification
setInterval(() => {
    const randomN = Math.floor(Math.random() * (activeRange.end - activeRange.start + 1)) + activeRange.start;
    requestPuzzleData(randomN);
}, 50); // 20 times per second (very fast)

// Refresh verification summary every 30 seconds
setInterval(fetchVerificationSummary, 30000);
