"""Chain: after main sweep finishes, launch new-dataset sweeps.

Small new datasets (balance, aggregation, compound): 30 seeds via sweep_all
Large new datasets (letter, shuttle, usps, fashion): 8 seeds via sweep_large
"""
import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def wait_for_sweep(log_path, timeout=7200):
    """Wait until sweep log contains 'DONE'."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(log_path):
            with open(log_path) as f:
                content = f.read()
            if 'DONE' in content:
                return True
        time.sleep(30)
    return False


def main():
    log = os.path.join(HERE, 'results', 'sweep_run.log')
    chain_log = os.path.join(HERE, 'results', 'chain.log')

    with open(chain_log, 'w') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] waiting for main sweep\n')

    # Wait for main 6-dataset sweep
    done = wait_for_sweep(log, timeout=14400)  # 4 hours max
    with open(chain_log, 'a') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] main sweep done={done}\n')

    # Launch small new datasets (30 seeds)
    small = ['balance', 'aggregation', 'compound']
    with open(chain_log, 'a') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] launching small new sweep: {small}\n')
    subprocess.run([PYTHON, '-u', '-m', 'src.sweep_all'] + small,
                   cwd=HERE, check=False)

    # Launch large new datasets (8 seeds)
    large = ['letter', 'shuttle', 'usps', 'fashion']
    with open(chain_log, 'a') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] launching large new sweep: {large}\n')
    subprocess.run([PYTHON, '-u', '-m', 'src.sweep_large'] + large,
                   cwd=HERE, check=False)

    with open(chain_log, 'a') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] ALL SWEEPS DONE\n')


if __name__ == '__main__':
    main()
