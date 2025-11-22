"""Main FastAPI application."""

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional

from .websocket import ws_manager
from ..state import PuzzleManager, LayoutStorage
from ..geometry import calculate_bounding_square, get_square_bounds


# Initialize app
app = FastAPI(title="Santa 2025 ML Packing System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
puzzle_manager: Optional[PuzzleManager] = None
storage: Optional[LayoutStorage] = None


def initialize_backend():
    """Initialize backend components."""
    global puzzle_manager, storage
    
    storage = LayoutStorage("data")
    
    # Try to load existing state
    loaded = storage.load()
    if loaded:
        puzzle_manager = loaded
        print("Loaded existing puzzle states")
    else:
        puzzle_manager = PuzzleManager()
        print("Created new puzzle manager")


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    initialize_backend()
    print("FastAPI server started")


@app.get("/")
async def root():
    """Root endpoint - redirect to live dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app/dashboard.html")


@app.get("/single")
async def single_view():
    """Single puzzle view."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app/index.html")


@app.get("/api/state")
async def get_state():
    """Get current state summary."""
    if puzzle_manager is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    summary = puzzle_manager.get_summary()
    return JSONResponse(content=summary)


@app.get("/api/puzzle/{n}")
async def get_puzzle(n: int):
    """Get specific puzzle state."""
    if puzzle_manager is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if n < 1 or n > 200:
        raise HTTPException(status_code=400, detail="Invalid puzzle number (must be 1-200)")
    
    puzzle = puzzle_manager.get_puzzle(n)
    
    if puzzle is None:
        raise HTTPException(status_code=404, detail=f"Puzzle {n} not found")
    
    # Convert to JSON-serializable format
    data = {
        'n': puzzle.n,
        'score': puzzle.score,
        'side_length': puzzle.side_length,
        'iterations': puzzle.iterations,
        'last_improvement': puzzle.last_improvement,
        'trees': [
            {
                'x': tree.x,
                'y': tree.y,
                'deg': tree.deg,
                'id': f"{n:03d}_{i}"
            }
            for i, tree in enumerate(puzzle.trees)
        ]
    }
    
    # Add bounding box info
    if puzzle.trees:
        square_x, square_y, side = get_square_bounds(puzzle.trees)
        data['bounding_box'] = {
            'x': square_x,
            'y': square_y,
            'side': side
        }
    
    return JSONResponse(content=data)


@app.get("/api/puzzles/all")
async def get_all_puzzles():
    """Get all puzzle summaries."""
    if puzzle_manager is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    puzzles = puzzle_manager.get_all_puzzles()
    
    data = [
        {
            'n': p.n,
            'score': p.score,
            'side_length': p.side_length,
            'iterations': p.iterations,
            'num_trees': len(p.trees)
        }
        for p in puzzles
    ]
    
    return JSONResponse(content={'puzzles': data, 'total_score': puzzle_manager.total_score})


@app.post("/api/save")
async def save_state():
    """Save current state to disk."""
    if puzzle_manager is None or storage is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    success = storage.save(puzzle_manager)
    
    if success:
        return {"message": "State saved successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save state")


@app.get("/api/export")
async def export_submission():
    """Export submission CSV file."""
    if puzzle_manager is None or storage is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Check if all puzzles are present
    missing = []
    for n in range(1, 201):
        if puzzle_manager.get_puzzle(n) is None:
            missing.append(n)
    
    if missing:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Missing puzzles",
                "missing_puzzles": missing
            }
        )
    
    # Export submission
    success = storage.export_submission(puzzle_manager, "submission.csv")
    
    if success:
        file_path = Path("data") / "submission.csv"
        return FileResponse(
            path=file_path,
            filename="submission.csv",
            media_type="text/csv"
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to export submission")


@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await ws_manager.connect(websocket)
    
    try:
        # Send initial state
        if puzzle_manager is not None:
            summary = puzzle_manager.get_summary()
            await ws_manager.send_personal_message(
                {'type': 'init', 'data': summary},
                websocket
            )
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_json()
            
            # Handle client requests
            if data.get('type') == 'get_puzzle':
                n = data.get('n')
                if puzzle_manager and 1 <= n <= 200:
                    puzzle = puzzle_manager.get_puzzle(n)
                    if puzzle:
                        puzzle_data = {
                            'n': puzzle.n,
                            'score': puzzle.score,
                            'side_length': puzzle.side_length,
                            'trees': [
                                {'x': t.x, 'y': t.y, 'deg': t.deg}
                                for t in puzzle.trees
                            ]
                        }
                        await ws_manager.send_personal_message(
                            {'type': 'puzzle_data', 'data': puzzle_data},
                            websocket
                        )
    
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# Mount static files (frontend)
frontend_path = Path(__file__).parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    print(f"Frontend mounted at: {frontend_path}")
else:
    print(f"Warning: Frontend directory not found at {frontend_path}")
