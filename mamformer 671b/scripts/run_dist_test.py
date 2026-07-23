"""
Multi-process launcher for Windows (no libuv needed)
Usage: python scripts/run_dist_test.py --nproc 2
"""
import os, sys, argparse, subprocess, tempfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=2)
    args = parser.parse_args()

    script = os.path.join(os.path.dirname(__file__), "test_all.py")
    procs = []

    for rank in range(args.nproc):
        env = os.environ.copy()
        env["WORLD_SIZE"] = str(args.nproc)
        env["LOCAL_RANK"] = str(rank)
        env["RANK"] = str(rank)
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = "29500"

        p = subprocess.Popen(
            [sys.executable, script],
            env=env,
            stdout=None,  # Share stdout
            stderr=None,
        )
        procs.append(p)

    # Wait for all
    for p in procs:
        p.wait()

    success = all(p.returncode == 0 for p in procs)
    print(f"\n{'='*50}")
    print(f"  Distributed test: {'PASSED' if success else 'FAILED'}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
