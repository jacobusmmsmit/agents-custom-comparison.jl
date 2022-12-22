# tags: evolutionary game theory, agent based modelling, indirect reciprocity

# This project is concerned with the development of cooperation through the
# mechanism of indirect reciprocity. Indirect reciprocity is best described as a
# complement to direct reciprocity which says "You help me, I help you. You are
# mean to me, I am mean to you". In contrast indirect reciprocity asks what one
# should do when interacting with an individual with whom you have never
# interacted before.

# An agent could simply choose to always help or not help individuals they have
# never met before, but this is overly simplistic and does not reflect real life
# where you may have more information about someone. In this setup, we assume
# that everyone has a reputation which they get based on their previous
# interaction. The reputation is either Good or Bad and is assigned to an
# individual by an independent third party called a judge. Judges make decisions
# using (social) norms, players make decisions using strategies. These are
# implemented in the same way as we now show:

# Social rules are functions of N boolean inputs, which return boolean outputs.
# Given this, we can simplify their implementation by representing them as
# integers encoding what every combination of N boolean inputs means.
struct SocialRule{N}
    int::Int
    function SocialRule{N}(int::I) where {N,I<:Integer}
        int >= 2^(2^N) && throw(ArgumentError("$int >= $(2^(2^N)) == 2^(2^$N)"))
        return new{N}(int)
    end
end

# Evaluating a social rule is as simple as combining the information into a
# binary number by concatenation, which we then use to "index" into the social
# rule by reading the ith digit. NB: 0 indexing
function read_bit(x, pos)
    pos < 0 && throw(ArgumentError("Bit position $pos must be >=0"))
    return (x >> pos) & 1 != 0
end

(rule::SocialRule{N})(xs::Vararg{I,N}) where {N,I<:Integer} =
    read_bit(rule.int, evalpoly(2, xs))

# Social rules are infallible, abstract concepts, but Agents are humans and make
# mistakes. A mistake in this context means flipping a bit in one of two ways:

# 1. Execution mistakes are mistakes in the output of a function i.e. I intend
#    to return true given the input  but I have a %chance of returning false
#    instead.
struct ExecutionMistake
    rate::Float64
    function ExecutionMistake(rate)
        0 <= rate <= 1 || throw(ArgumentError("Rate must be between 0 and 1."))
        new(rate)
    end
end

# 2. Perception mistakes are mistakes in the input of a function i.e. I think
#    one of the inputs is true, but I'm wrong with some rate and may then
#    misperceive it as false.
struct PerceptionMistake{N}
    rates::NTuple{N,Float64}
    function PerceptionMistake(rates)
        all(0 .<= rates .<= 1) || throw(ArgumentError("All rates must be between 0 and 1."))
        N = length(rates)
        return new{N}(rates)
    end
end

# And so an Agent of order N takes N bits of information, feeds it (possibly
# making mistakes) to its order N social rule, receives back an outcome, and
# then carries it out (again possibly making a mistake).
struct Agent{N}
    rule::SocialRule{N}
    pm::PerceptionMistake{N}
    em::ExecutionMistake
    function Agent{N}(
        rule::SocialRule{N},
        pm = PerceptionMistake(Tuple(0.0 for _ = 1:N)),
        em = ExecutionMistake(0.0),
    ) where {N}
        new{N}(rule, pm, em)
    end
end

# Now interpreting what these outputs mean in context: If a social rule/agent
# evaluates input information as true, it will donate to the receiving agent
# which incurs a cost to itself and a benefit to the receiver. If the evaluation
# is instead false, no benefit or cost is incurred to either player as no
# donation takes place.
struct SystemParameters
    cost::Float64
    benefit::Float64
end

# In this project, we take a very simple setup where the only information agents
# use to make decisions is whether the person they are potentially donating to
# has a good reputation.
Player = Agent{1}
Judge = Agent{2}
Strategy = SocialRule{1}
Norm = SocialRule{2}
