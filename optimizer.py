import numpy as np
import json
from api import predict

# ── Load controllable bounds ──────────────────────────────────────────────────
with open("col_stats.json", "r") as f:
    COL_STATS = json.load(f)

CONTROLLABLE = list(COL_STATS.keys())  # order matters for chromosome

# ── Genetic Algorithm ─────────────────────────────────────────────────────────

def decode_chromosome(chromosome: np.ndarray) -> dict:
    """Map chromosome genes to controllable param names."""
    return {col: chromosome[i] for i, col in enumerate(CONTROLLABLE)}

def fitness(chromosome: np.ndarray, fixed_params: dict) -> float:
    """
    Evaluate one individual.
    fixed_params: uncontrollable values for this timestep {col: value}
    Returns predicted ROP (higher = better).
    """
    controlled = decode_chromosome(chromosome)
    input_data = {k: [v] for k, v in {**fixed_params, **controlled}.items()}
    return predict(input_data)

def random_individual() -> np.ndarray:
    """Create a random chromosome within bounds."""
    return np.array([
        np.random.uniform(COL_STATS[col]["min"], COL_STATS[col]["max"])
        for col in CONTROLLABLE
    ])

def clip_to_bounds(chromosome: np.ndarray) -> np.ndarray:
    """Ensure genes stay within valid ranges."""
    for i, col in enumerate(CONTROLLABLE):
        chromosome[i] = np.clip(
            chromosome[i],
            COL_STATS[col]["min"],
            COL_STATS[col]["max"]
        )
    return chromosome

def select_parents(population: list, fitnesses: np.ndarray, n_parents: int) -> list:
    """Tournament selection."""
    parents = []
    pop_size = len(population)
    for _ in range(n_parents):
        candidates = np.random.choice(pop_size, size=3, replace=False)
        winner = candidates[np.argmax(fitnesses[candidates])]
        parents.append(population[winner].copy())
    return parents

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple:
    """Simulated Binary Crossover (SBX)."""
    eta = 2.0  # distribution index
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            u = np.random.rand()
            beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
    return child1, child2

def mutate(chromosome: np.ndarray, mutation_rate: float = 0.2, mutation_strength: float = 0.1) -> np.ndarray:
    """Gaussian mutation scaled to each param's range."""
    mutated = chromosome.copy()
    for i, col in enumerate(CONTROLLABLE):
        if np.random.rand() < mutation_rate:
            range_size = COL_STATS[col]["max"] - COL_STATS[col]["min"]
            mutated[i] += np.random.normal(0, mutation_strength * range_size)
    return clip_to_bounds(mutated)

def optimize(
    fixed_params: dict,
    pop_size: int = 50,
    n_generations: int = 100,
    mutation_rate: float = 0.2,
    mutation_strength: float = 0.1,
    elite_size: int = 2,
    callback=None
) -> dict:
    """
    Run the genetic algorithm.

    Args:
        fixed_params:       Uncontrollable params {col: value} for this timestep.
        pop_size:           Number of individuals per generation.
        n_generations:      Number of generations to evolve.
        mutation_rate:      Probability of mutating each gene.
        mutation_strength:  Gaussian noise scale relative to param range.
        elite_size:         Number of top individuals carried over unchanged.
        callback:           Optional callable(generation, best_rop, best_params, all_rops)
                            called each generation (useful for UI progress).
                            all_rops is the list of every chromosome's ROP in that generation.

    Returns:
        {
            "best_rop":    float,
            "best_params": {col: value},   # optimized controllable params
            "history":     [best_rop_per_generation],
        }
    """
    # Initialize population
    population = [random_individual() for _ in range(pop_size)]
    history = []

    for gen in range(n_generations):
        # Evaluate fitness
        fitnesses = np.array([fitness(ind, fixed_params) for ind in population])

        # Track best
        best_idx = np.argmax(fitnesses)
        best_rop = fitnesses[best_idx]
        history.append(float(best_rop))

        if callback:
            all_rops = fitnesses.tolist()
            callback(gen + 1, best_rop, decode_chromosome(population[best_idx]), all_rops)

        # Elitism — carry top individuals unchanged
        sorted_idx = np.argsort(fitnesses)[::-1]
        elites = [population[i].copy() for i in sorted_idx[:elite_size]]

        # Build next generation
        next_gen = elites[:]
        while len(next_gen) < pop_size:
            parents = select_parents(population, fitnesses, 2)
            child1, child2 = crossover(parents[0], parents[1])
            child1 = mutate(child1, mutation_rate, mutation_strength)
            child2 = mutate(child2, mutation_rate, mutation_strength)
            next_gen.extend([child1, child2])

        population = next_gen[:pop_size]

    # Final evaluation
    fitnesses = np.array([fitness(ind, fixed_params) for ind in population])
    best_idx = np.argmax(fitnesses)

    return {
        "best_rop":    float(fitnesses[best_idx]),
        "best_params": decode_chromosome(population[best_idx]),
        "history":     history,
    }


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: fix uncontrollable params, let GA find best controllable ones
    fixed = {
        "depth":        1500.0,
        "block_height": 12.5,
        "bit_depth":    1498.0,
        "hookload":     210.0,
        "flow_out":     850.0,
        "temp_in":      35.0,
        "temp_out":     42.0,
    }

    def progress(gen, rop, params):
        if gen % 10 == 0:
            print(f"  Gen {gen:>4} | Best ROP: {rop:.4f}")

    print("Running Genetic Algorithm Optimization...")
    result = optimize(fixed, pop_size=50, n_generations=100, callback=progress)

    print(f"\n✅ Best ROP:    {result['best_rop']:.4f}")
    print("✅ Best Params:")
    for k, v in result["best_params"].items():
        print(f"   {k:>15}: {v:.4f}")