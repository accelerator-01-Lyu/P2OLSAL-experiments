"""Launch sweep with 15 workers."""
import sys, os

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.sweep_all import main
    main(workers=15)
