"""Entry point for single-image inference: python -m src.inference.single"""
import sys
from src.inference.single.run_inference import main

if __name__ == "__main__":
    sys.exit(main())
