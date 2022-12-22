using Random
using StatsBase

include("simplified_core.jl")

# In the custom case an ABMAgent is just the same as Agent as utilities aren't stored in the agent.

mutable struct ABMMutables
    step_counter::Int64
    cooperation_counter::Int64
end

struct ABMConstants
    nagents::Int64
    reputations::BitArray
    utilities::Vector{Float64}
    judge::Judge
    params::SystemParameters
    selection_intensity::Float64
    mutation_probability::Float64
    reproduce_every::Int64
end

struct ABM{R<:AbstractRNG}
    agents::Vector{Player}
    constants::ABMConstants
    mutables::ABMMutables
    rng::R
end

function ABM(agents, constants, mutables, rng=Random.default_rng())
    ABM{typeof(rng)}(agents, constants, mutables, rng)
end

@inline function evaluate(agent::Agent{N}, information::NTuple{N,Bool}, model) where {N}
    perceived_information = map(information, agent.pm.rates) do infobyte, misperception_rate
        rand(model.rng) > misperception_rate ? infobyte : !infobyte
    end
    intention = agent.rule(perceived_information...)
    outcome = rand(model.rng) > agent.em.rate ? intention : !intention
    return outcome
end

@inline evaluate(agent::Agent{1}, information::Bool, model) =
    evaluate(agent, Tuple(information), model)

function agent_step!(donor_id::Integer, model)
    # Find interaction partner from all other agents (well-mixed)
    recipient_id = rand(1:model.constants.nagents) # Possible optimisation, precalculate all recipeints
    while recipient_id == donor_id
        recipient_id = rand(1:model.constants.nagents)
    end
    pairwise_agent_step!(donor_id, recipient_id, model)
    return nothing
end

function pairwise_agent_step!(donor_id, recipient_id, model)
    is_good = model.constants.reputations[recipient_id]
    # model is passed to evaluate to use model.rng to ensure reproducibility
    outcome = evaluate(model.agents[donor_id], is_good, model) # outcome is strictly Boolean
    model.mutables.cooperation_counter += outcome
    # Judge the donor's action and therewith update the reputations
    judgement = evaluate(model.constants.judge, (is_good, outcome), model) # judgement also boolean
    model.constants.reputations[donor_id] = judgement
    if outcome
        model.constants.utilities[donor_id] -= model.constants.params.cost
        model.constants.utilities[recipient_id] += model.constants.params.benefit
    end
    return nothing
end

function model_step!(model)
    # Choose two agents and let one reproduce with the Fermi update rule
    nagents = model.constants.nagents
    imitator_id, imitated_id = sample(1:nagents, 2)
    imitation_probability =
        1 / (
            1 + exp(
                -model.constants.selection_intensity * (
                    (
                        model.constants.utilities[imitated_id] -
                        model.constants.utilities[imitator_id]
                    ) / model.constants.reproduce_every
                ),
            )
        )
    if rand(model.rng) < imitation_probability
        model.agents[imitator_id] = model.agents[imitated_id]
    end
    # Reset utilities
    model.constants.utilities .= 0.0
    # Randomly mutate an agent
    if rand(model.rng) < model.constants.mutation_probability
        mutated_id = rand(1:nagents)
        model.agents[mutated_id] = Player(
            Strategy(rand(model.rng, 0:3)),
            model.agents[mutated_id].pm,
            model.agents[mutated_id].em,
        )
    end
    return nothing
end

#TODO: Profile complex_step!
function complex_step!(model)
    model.mutables.cooperation_counter = 0
    model.mutables.step_counter += 1
    for id = 1:model.constraints.nagents
        agent_step!(id, model)
    end
    if model.mutables.step_counter % model.constants.reproduce_every == 0
        model_step!(model)
    end
end

function initialize(;
    numagents=50,
    norm::Norm=Norm(0),
    strategies=[Strategy(rand(model.rng, 0:3)) for i = 1:numagents],
    player_pm::PerceptionMistake{1},
    player_em::ExecutionMistake,
    judge_pm::PerceptionMistake{2},
    judge_em::ExecutionMistake,
    reputations=trues(numagents),
    params::SystemParameters,
    selection_intensity=1.0,
    mutation_probability=0.01,
    reproduce_every=100,
    seed=125
)
    judge = Judge(norm, judge_pm, judge_em)
    mutables = ABMMutables(0, 0)
    constants = ABMConstants(
        numagents,
        reputations,
        zeros(numagents),
        judge,
        params,
        selection_intensity,
        mutation_probability,
        reproduce_every,
    )
    rng = Random.MersenneTwister(seed)
    agents = [Player(strategy, player_pm, player_em) for strategy in strategies]
    model = ABM(agents, constants, mutables, rng)
    return model
end
