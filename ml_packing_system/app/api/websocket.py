"""WebSocket manager for real-time updates."""

import json
import asyncio
from typing import Set, Dict, Any
from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[Any, Any], websocket: WebSocket):
        """Send message to specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[Any, Any]):
        """Broadcast message to all connected clients."""
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_state_update(self, puzzle_n: int, state_data: dict):
        """Broadcast puzzle state update."""
        message = {
            'type': 'state_update',
            'puzzle_n': puzzle_n,
            'data': state_data
        }
        await self.broadcast(message)
    
    async def broadcast_progress(self, summary: dict):
        """Broadcast overall progress summary."""
        message = {
            'type': 'progress',
            'data': summary
        }
        await self.broadcast(message)
    
    async def broadcast_improvement(self, puzzle_n: int, old_score: float, new_score: float):
        """Broadcast score improvement notification."""
        improvement = old_score - new_score
        message = {
            'type': 'improvement',
            'puzzle_n': puzzle_n,
            'old_score': old_score,
            'new_score': new_score,
            'improvement': improvement,
            'message': f'Puzzle #{puzzle_n} improved by {improvement:.6f}'
        }
        await self.broadcast(message)


# Global instance
ws_manager = ConnectionManager()
