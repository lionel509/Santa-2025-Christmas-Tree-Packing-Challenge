"""Main entry point for the ML packing system."""

import sys
import argparse
import threading
import uvicorn
from app.main import get_app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Santa 2025 ML Packing System"
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host address (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port number (default: 8000)'
    )
    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Disable ML optimization (use heuristics only)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for ML computation (default: cpu)'
    )
    parser.add_argument(
        'mode',
        nargs='?',
        choices=['standard', 'backinpacking', 'backpacking'],
        default='backinpacking',
        help='Operation mode (standard or backinpacking)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Santa 2025 - Christmas Tree Packing Challenge")
    print("ML-Powered Autonomous Optimization System")
    print("="*60)
    print()
    print(f"🚀 Web server starting on http://{args.host}:{args.port}")
    print(f"📊 Dashboard available at: http://{args.host}:{args.port}/")
    print()
    print("✨ Dashboard is accessible immediately!")
    print("⚙️  Initialization and optimization running in background...")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    # Get application instance (but don't initialize yet)
    app = get_app(initialize=False)
    app.use_ml = not args.no_ml
    app.device = args.device
    app.backpacking_mode = args.mode in ['backinpacking', 'backpacking']
    
    # Run initialization and optimization in background thread
    def background_init():
        try:
            print("[Background] Starting initialization...")
            app.initialize()
            print("[Background] ✅ Initialization complete!")
            print("[Background] Starting optimization...")
            app.start_optimization()
            print("[Background] ✅ Optimization loop running!")
        except Exception as e:
            print(f"[Background] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    init_thread = threading.Thread(target=background_init, daemon=True)
    init_thread.start()
    
    try:
        # Run FastAPI server immediately (non-blocking)
        uvicorn.run(
            "app.api.main:app",
            host=args.host,
            port=args.port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        app.stop()
    except Exception as e:
        print(f"\nError: {e}")
        app.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
