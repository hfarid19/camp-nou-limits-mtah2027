import os
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "candidates"))
import bench
m = bench.load_candidate(sys.argv[1])
bench.run_match(m, sys.argv[2])
