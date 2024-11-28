from environments.environment_examples import (
    create_simple_environment, create_linear_environment, create_complex_environment
)
from solvers.baseline_solver import BaselineSolver
from evaluation.evaluation_suite import evaluate_solver

def main():
    """
    Main script to run the assignment pipeline.
    """
    # Step 1: Instantiate environments
    envs = [
        create_simple_environment(),
        create_linear_environment(),
        create_complex_environment()
    ]

    # Step 2: Evaluate baseline solver
    baseline_solver = BaselineSolver(envs[0])  # TODO: Test baseline on all environments
    evaluation_results = evaluate_solver(baseline_solver, envs)
    print("Baseline Solver Evaluation Results:", evaluation_results)

    # Step 3: Train and test deep learning solver
    # TODO: Implement and test the deep learning solver here.

if __name__ == "__main__":
    main()
