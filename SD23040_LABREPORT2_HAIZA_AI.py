import streamlit as st
import numpy as np
import random
import pandas as pd

# -------------------- GA Problem --------------------
CHROM_LENGTH_DEFAULT = 80
POP_SIZE_DEFAULT = 300
GENERATIONS_DEFAULT = 50
MUT_RATE_DEFAULT = 0.01

def fitness(chrom):
    """Fitness = 80 when number of ones = 50"""
    return 80 - abs(np.sum(chrom) - 50)

# -------------------- GA Operators --------------------
def init_population(pop_size, chrom_length):
    return np.random.randint(0, 2, (pop_size, chrom_length))

def selection(pop, fit, tournament_k):
    idx = np.random.choice(len(pop), tournament_k)
    best_idx = idx[np.argmax(fit[idx])]
    idx2 = np.random.choice(len(pop), tournament_k)
    best_idx2 = idx2[np.argmax(fit[idx2])]
    return pop[best_idx].copy(), pop[best_idx2].copy()

def crossover(p1, p2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, len(p1)-1)
        return np.concatenate([p1[:point], p2[point:]]), np.concatenate([p2[:point], p1[point:]])
    else:
        return p1.copy(), p2.copy()

def mutation(child, mut_rate):
    for i in range(len(child)):
        if random.random() < mut_rate:
            child[i] = 1 - child[i]
    return child

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm", layout="wide")
st.title("Genetic Algorithm - Target 50 Ones (Bitstring)")

with st.sidebar:
    st.header("GA Parameters")
    chrom_length = st.number_input("Chromosome length (bits)", min_value=8, max_value=4096, value=CHROM_LENGTH_DEFAULT, step=8)
    pop_size = st.number_input("Population size", min_value=10, max_value=5000, value=POP_SIZE_DEFAULT, step=10)
    generations = st.number_input("Generations", min_value=1, max_value=1000, value=GENERATIONS_DEFAULT, step=1)
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.9, 0.05)
    mutation_rate = st.slider("Mutation rate (per gene)", 0.0, 1.0, MUT_RATE_DEFAULT, 0.005)
    tournament_k = st.slider("Tournament size", 2, 10, 3)
    elitism = st.slider("Elites per generation", 0, 100, 2)
    seed = st.number_input("Random seed (optional)", min_value=0, max_value=2**32-1, value=42)
    live_chart = st.checkbox("Live chart while running", value=True)

left, right = st.columns([2, 1])

if left.button("Run GA"):
    np.random.seed(seed)
    random.seed(seed)
    population = init_population(pop_size, chrom_length)
    history_best = []
    history_avg = []
    history_worst = []

    chart_area = left.empty()
    
    for gen in range(generations):
        fitness_values = np.array([fitness(ind) for ind in population])
        history_best.append(np.max(fitness_values))
        history_avg.append(np.mean(fitness_values))
        history_worst.append(np.min(fitness_values))

        # Live chart
        if live_chart:
            df = pd.DataFrame({
                "Best": history_best,
                "Average": history_avg,
                "Worst": history_worst
            })
            chart_area.line_chart(df)

        # Elitism: keep top E individuals
        E = min(elitism, pop_size)
        elite_idx = np.argpartition(fitness_values, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = population[elite_idx] if E > 0 else np.empty((0, chrom_length))

        # GA evolution
        next_pop = []
        while len(next_pop) < pop_size - E:
            p1, p2 = selection(population, fitness_values, tournament_k)
            c1, c2 = crossover(p1, p2, crossover_rate)
            c1 = mutation(c1, mutation_rate)
            c2 = mutation(c2, mutation_rate)
            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        population = np.vstack([np.array(next_pop), elites]) if E > 0 else np.array(next_pop)

    # -------------------- Results --------------------
    fitness_values = np.array([fitness(ind) for ind in population])
    best_idx = np.argmax(fitness_values)
    best = population[best_idx]

    left.subheader("Best Solution")
    left.write(f"Best fitness: {fitness_values[best_idx]}")
    bitstring = ''.join(map(str, best.astype(int)))
    left.code(bitstring)
    left.write(f"Number of ones: {np.sum(best)}")

    # Store final population for snapshot
    st.session_state["_final_pop"] = population
    st.session_state["_final_fit"] = fitness_values

# -------------------- Population Snapshot --------------------
right.subheader("Final Population Snapshot")
if right.button("Show final population"):
    pop = st.session_state.get("_final_pop")
    fit = st.session_state.get("_final_fit")
    if pop is None or fit is None:
        right.info("Run GA first to view the final population.")
    else:
        nshow = min(20, pop.shape[0])
        df = pd.DataFrame(pop[:nshow])
        df["fitness"] = fit[:nshow]
        right.dataframe(df)
