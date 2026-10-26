"""Refit RQ3 history and conditional logistic models from the final Excel labels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduce import main

if __name__=='__main__':
    sys.argv.insert(1,'models')
    main()
