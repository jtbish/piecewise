import random
import subprocess


def main():
    budget = 1
    for exp_num in range(budget):
        subprocess_cmd = ["python3", "cartpole_xcsr.py"]
        params = _sample_params()
        for k, v in params.items():
            subprocess_cmd.append(f"{k}={v}")
        subprocess_cmd.append(f"--experiment-name=cartpole_xcsr_{exp_num}")
        print(subprocess_cmd)

        subprocess.run(subprocess_cmd)


def _sample_params():
    return {}


if __name__ == "__main__":
    main()
