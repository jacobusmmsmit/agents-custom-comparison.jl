using StatsBase
using Agents # Core Agent functionality
using Random # To set seed
using BenchmarkTools # Timing functions for performance optimisations
using Profile # Profiling functions for performance optimisations
using PProf # Visualising profiles

include("../src/hybrid_impl.jl")

Random.seed!(1)

begin # Setup
    player_pm = PerceptionMistake((0.00,))
    player_em = ExecutionMistake(0.01)

    judge_pm = PerceptionMistake((0.00, 0.00))
    judge_em = ExecutionMistake(0.01)

    benefit = 5
    cost = 1

    params = SystemParameters(benefit, cost)

    numagents = 500
    strategies = [Strategy(rand(1:3)) for i = 1:numagents]
    # strategies = [Strategy(15) for i in 1:numagents]

    model = initialize(
        numagents = numagents,
        strategies = strategies,
        norm = Norm(0),
        player_pm = player_pm,
        player_em = player_em,
        judge_pm = judge_pm,
        judge_em = judge_em,
        params = params,
        mutation_probability = 0.1,
        reproduce_every = 2 * numagents,
    )
end

# @benchmark agent_step!($model.agents[1], $model)
# @benchmark model_step!($model)

nsteps = 500
@benchmark run!($model, dummystep, complex_step!, nsteps)
