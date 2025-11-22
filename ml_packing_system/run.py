"""Main entry point for the ML packing system."""

import sys
import argparse
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
    
    args = parser.parse_args()
    
    print("="*60)
    print("Santa 2025 - Christmas Tree Packing Challenge")
    print("ML-Powered Autonomous Optimization System")
    print("="*60)
    print()
    
    # Initialize application
    app = get_app()
    app.use_ml = not args.no_ml
    app.device = args.device
    app.initialize()
    
    # Start optimization
    app.start_optimization()
    
    print()
    print(f"Starting web server on http://{args.host}:{args.port}")
    print(f"Open http://{args.host}:{args.port}/ in your browser to view the interface")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    try:
        # Run FastAPI server
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
