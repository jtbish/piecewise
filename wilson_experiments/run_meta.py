import random
import subprocess


def main():
    budget = 10
    for exp_num in range(budget):
        subprocess_cmd = ["python3", "6_mux.py"]
        params = _sample_params()
        for k, v in params.items():
            subprocess_cmd.append(f"{k}={v}")
        subprocess_cmd.append(f"--experiment-name=exp{exp_num}")
        print(subprocess_cmd)

        subprocess.run(subprocess_cmd)


def _sample_params():
    return {
        "--num-training-samples": 1000,
        "--N": random.choice([400, 500, 600]),
        "--epsilon-nought": random.uniform(0.01, 0.05),
        "--gamma": random.uniform(0.71, 0.99),
        "--prediction-I": 1e-3,
        "--epsilon-I": 1e-3,
        "--fitness-I": 1e-3,
        "--alg-seed": random.choice([1, 2, 3, 4, 5])
    }


if __name__ == "__main__":
    main()
