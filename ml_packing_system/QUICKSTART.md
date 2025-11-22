# Quick Start Guide

## 🚀 Running the System

### 1. Navigate to the project directory:
```bash
cd ml_packing_system
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the system:
```bash
python run.py
```

### 4. Open your browser:

**Live Dashboard (Stock Market Style):**
Navigate to: **http://127.0.0.1:8000/**
- See ALL puzzles updating simultaneously
- Grid view with live visualizations
- Flash animations when scores improve
- Filter by puzzle range

**Single Puzzle View:**
Navigate to: **http://127.0.0.1:8000/single**
- Focus on one puzzle at a time
- Detailed visualization
- Full controls

---

## ✨ What Happens Now?

The system will **automatically**:

1. ✅ Initialize all 200 puzzles (1-200 trees each)
2. 🔄 Start continuous optimization using ML + heuristics
3. 💾 Auto-save progress every 5 cycles
4. 📊 Broadcast real-time updates to the web interface
5. 🎯 Generate the final 20,100-row submission CSV

**You just watch!** No manual intervention needed.

---

## 🖥️ Web Interface Features

- **Live Visualization**: Watch trees being optimized in real-time
- **Statistics Dashboard**: Track total score, iterations, improvements
- **Puzzle Selector**: View any of the 200 puzzles
- **Play/Pause**: Control update stream
- **Export Button**: Download submission CSV anytime
- **Activity Log**: See what the system is doing

---

## 📥 Getting Your Submission

### Option 1: Via Web Interface
Click the **"Export Submission"** button in the browser

### Option 2: Via API
```bash
curl http://127.0.0.1:8000/api/export -o submission.csv
```

### Option 3: From Disk
The file is automatically created at:
```
ml_packing_system/data/submission.csv
```

---

## ⚙️ Command Line Options

```bash
# Run on different port
python run.py --port 8080

# Use GPU (if available)
python run.py --device cuda

# Disable ML (heuristics only)
python run.py --no-ml

# Run on all network interfaces
python run.py --host 0.0.0.0
```

---

## 🛑 Stopping the System

Press **Ctrl+C** in the terminal

The system will:
- Save current state
- Export final submission
- Shut down gracefully

---

## 📊 Monitoring Progress

### Real-time Web Dashboard
- Open http://127.0.0.1:8000/app

### API Endpoints
```bash
# System status
curl http://127.0.0.1:8000/api/state

# Specific puzzle
curl http://127.0.0.1:8000/api/puzzle/42

# All puzzles summary
curl http://127.0.0.1:8000/api/puzzles/all
```

### Terminal Output
Watch the terminal for optimization progress and improvements

---

## 💡 Tips

1. **Let it run**: The longer it runs, the better the scores
2. **Check progress**: Use the web interface to monitor in real-time
3. **Auto-save**: Your progress is saved automatically
4. **Resume anytime**: Stop and restart - it picks up where it left off
5. **Export often**: Download submission CSV periodically

---

## ❓ Troubleshooting

**Import errors?**
→ Make sure you're in the `ml_packing_system` directory

**Port already in use?**
→ Use `python run.py --port 8080`

**Can't connect to WebSocket?**
→ Check firewall or try `python run.py --host 0.0.0.0`

**Out of memory?**
→ Use `python run.py --no-ml` for heuristics only

---

## 🎯 Expected Results

- **Initialization**: 30-60 seconds for all 200 puzzles
- **First cycle**: 5-10 minutes
- **Continuous improvement**: Scores decrease over time
- **Submission**: Ready anytime, automatically validated

---

**Enjoy watching the ML optimize your packing! 🎄✨**
