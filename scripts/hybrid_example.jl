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

    params = SystemParameters(
        benefit,
        cost
    )

    numagents = 500
    strategies = [Strategy(rand(1:3)) for i in 1:numagents]
    # strategies = [Strategy(15) for i in 1:numagents]

    model = initialize(
        numagents=numagents,
        strategies=strategies,
        norm=Norm(0),
        player_pm=player_pm,
        player_em=player_em,
        judge_pm=judge_pm,
        judge_em=judge_em,
        params=params,
        mutation_probability=0.1,
        reproduce_every=2 * numagents
    )
end

@benchmark agent_step!($model.agents[1], $model)
@benchmark model_step!($model)

nsteps = 50;
nsaves = min(1000, nsteps)
save_every = nsteps ÷ nsaves
step_axis = save_every:save_every:nsteps # Used for plotting later
begin # Run model and collect data
    # Data to collect:
    strat_freq = zeros(nsaves, 16)
    rep_freq = zeros(nsaves, 2)
    coop_freq = zeros(nsaves)
    Profile.clear()
    @time for step in 1:nsteps
        ## Progress bar :)
        # step % (nsteps // 10) == 0 && print("-")
        # Step model
        model.cooperation_counter = 0
        model.step_counter += 1
        for id in 1:numagents
            agent_step!(model.agents[id], model)
        end
        model.step_counter % model.reproduce_every == 0 && model_step!(model)

        # Collect data
        # if step % save_every == 0
        #     save_step = step ÷ save_every
        #     # println("Completion: $(100*save_step/nsaves)%")
        #     coop_freq[save_step] = model.cooperation_counter / numagents
        #     for strat in 0:15
        #         strat_freq[save_step, strat+1] = count(ag -> ag.egt_agent.rule.int == strat, allagents(model)) / numagents
        #     end
        #     for (i, group) in enumerate((red, blue))
        #         grouped = Iterators.filter(ag -> ag.group == group, allagents(model))
        #         rep_freq[save_step, i] = mean(ag -> model.reputations[ag.id], grouped)
        #     end
        # end
    end
end