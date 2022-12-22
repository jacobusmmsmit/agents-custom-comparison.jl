using StatsBase
using Agents

include("simplified_core.jl")

# Now shifting attention to the Agents.jl ABM

# Agent type used in the ABM, a wrapper around EGT agents with ABM specific information.
@agent ABMAgent{N} NoSpaceAgent begin
    egt_agent::Agent{N}
    utility::Float64
end

# A wrapper around the EGT "SystemParameters" with ABM specific information.
mutable struct ABMProperties
    reputations::BitArray
    judge::Judge
    params::SystemParameters
    selection_intensity::Float64
    mutation_probability::Float64
    reproduce_every::Int64
    step_counter::Int64
    cooperation_counter::Int64
end

@inline function evaluate(egt_agent::Agent{N}, information::NTuple{N,Bool}, model) where {N}
    perceived_information =
        map(information, egt_agent.pm.rates) do infobyte, misperception_rate
            rand(model.rng) > misperception_rate ? infobyte : !infobyte
        end
    intention = egt_agent.rule(perceived_information...)
    outcome = rand(model.rng) > egt_agent.em.rate ? intention : !intention
    return outcome
end

@inline evaluate(egt_agent::Agent{1}, information::Bool, model) =
    evaluate(egt_agent, Tuple(information), model)

function agent_step!(donor::ABMAgent{N}, model) where {N}
    # Find interaction partner from all other agents (well-mixed)
    recipient_id = rand(1:length(model.reputations)) # Possible optimisation, precalculate all recipeints
    while recipient_id == donor.id
        recipient_id = rand(1:length(model.reputations))
    end
    recipient = model.agents[recipient_id]
    pairwise_agent_step!(donor, recipient, model)
    return nothing
end

function pairwise_agent_step!(donor, recipient, model)
    is_good = model.reputations[recipient.id]
    # # model is passed to evaluate to use model.rng to ensure reproducibility
    outcome = evaluate(donor.egt_agent, is_good, model) # outcome is strictly Boolean
    model.properties.cooperation_counter += outcome
    # model.cooperation_counter += outcome # This causes a string allocation, bug?
    # Judge the donor's action and therewith update the reputations
    judgement = evaluate(model.judge, (is_good, outcome), model) # judgement also boolean
    model.reputations[donor.id] = judgement
    if outcome
        donor.utility -= model.params.cost
        recipient.utility += model.params.benefit
    end
    return nothing
end

function model_step!(model)
    # Choose two agents and let one reproduce with the Fermi update rule
    imitator_id, imitated_id = sample(1:length(model.reputations), 2)
    imitator = model.agents[imitator_id]
    imitated = model.agents[imitated_id]
    imitation_probability =
        1 / (
            1 + exp(
                -model.selection_intensity *
                ((imitated.utility - imitator.utility) / model.reproduce_every),
            )
        )
    if rand(model.rng) < imitation_probability
        imitator.egt_agent = imitated.egt_agent
    end
    # Reset utilities
    for abm_agent in allagents(model)
        abm_agent.utility = 0.0
    end
    # Randomly mutate an abm_agent
    if rand(model.rng) < model.mutation_probability
        mutated = random_agent(model)
        mutated.egt_agent = Player(
            Strategy(rand(model.rng, 0:3)),
            mutated.egt_agent.pm,
            mutated.egt_agent.em,
        )
    end
    return nothing
end

#TODO: Profile complex_step!
function complex_step!(model)
    model.cooperation_counter = 0
    model.step_counter += 1
    for id in Schedulers.fastest(model)
        agent_step!(model[id], model)
    end
    if model.step_counter % model.reproduce_every == 0
        model_step!(model)
    end
end

function initialize(;
    numagents = 50,
    norm::Norm = Norm(0),
    strategies = [Strategy(rand(model.rng, 0:3)) for i = 1:numagents],
    player_pm::PerceptionMistake{1},
    player_em::ExecutionMistake,
    judge_pm::PerceptionMistake{2},
    judge_em::ExecutionMistake,
    reputations = trues(numagents),
    params::SystemParameters,
    selection_intensity = 1.0,
    mutation_probability = 0.01,
    reproduce_every = 100,
    seed = 125,
)
    judge = Judge(norm, judge_pm, judge_em)
    properties = ABMProperties(
        reputations,
        judge,
        params,
        selection_intensity,
        mutation_probability,
        reproduce_every,
        0,
        0.0,
    )
    rng = Random.MersenneTwister(seed)
    model = ABM(
        ABMAgent{1};
        properties,
        rng,
        scheduler = Schedulers.Randomly(), #TODO: fastest?
    )
    for n = 1:numagents
        egt_agent = Player(strategies[n], player_pm, player_em)
        abm_agent = ABMAgent{1}(n, egt_agent, 0.0)
        add_agent!(abm_agent, model)
    end
    return model
end
