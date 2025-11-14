"""Entry point for multi-image inference: python -m src.inference.multi"""
import sys
from src.inference.multi.run_inference import main

if __name__ == "__main__":
    sys.exit(main())
