module SuperConvergence

using Metalhead, Statistics, Random
using Flux.Data.MNIST
using Base.Iterators: partition
using Images: channelview
using Flux: onehotbatch, onecold
using Flux, CuArrays
using ProgressMeter

export TriangularCycle, ConstantCycle, find_eta!, one_cycle!

include("data.jl")
include("models.jl")
include("cycle.jl")

"""
find_eta!!(loss, params, train_data, val_data, opt;)

Flux.train!(loss, params, data, opt)
"""
function find_eta!(loss, ps, train_data, val_data, opt; 
                  start_eta=1e-8, 
                  end_eta=100.0, 
                  β=0.98, 
                  threshold=4.0)
    @info("Finding best learning rate ... range $start_eta : $end_eta")
    mult = (end_eta / start_eta)^(1 / (length(train_data)-1))
    batch_idx = 1
    opt.eta = start_eta

    @info(mult)

    train_avg_loss = 0.0
    val_avg_loss = 0.0
    best_loss = 0.0

    log_etas = Float64[]
    train_losses = Float64[]
    val_losses = Float64[]

    #= @showprogress for (x, y) in train_data =#
    for (x, y) in train_data
        l = loss(x, y)
        train_loss = l |> Flux.data
        val_loss = loss(val_data[1], val_data[2]) |> Flux.data

        if isnan(train_loss) || isnan(val_loss)
            @info("Returning, loss is NaN ... current learning rate = $(opt.eta)")
            return log_etas, train_losses, val_losses
        end

        train_avg_loss = β * train_avg_loss + (1-β) * train_loss
        train_smoothed_loss = train_avg_loss / (1 - β^batch_idx)
        val_avg_loss = β * val_avg_loss + (1-β) * val_loss
        val_smoothed_loss = val_avg_loss / (1 - β^batch_idx)

        if batch_idx > 1 && val_smoothed_loss > threshold * best_loss
            @info("Returning, loss diverging ... current learning rate = $(opt.eta)")
            return log_etas, train_losses, val_losses
        end

        if val_smoothed_loss < best_loss || batch_idx == 1
            best_loss = val_smoothed_loss
        end

        push!(log_etas, log10(opt.eta))
        push!(train_losses, train_smoothed_loss)
        push!(val_losses, val_smoothed_loss)

        # update model
        Flux.back!(l)
        Flux.Optimise._update_params!(opt, ps)

        opt.eta *= mult
        batch_idx += 1
    end

    return log_etas, train_losses, val_losses
end

"""
Flux.train!(loss, params, data, opt)

    eta_cycle = TriangularCycle(min_eta, max_eta, iters)
    rho_cycle = TriangularCycle(max_rho, min_rho, iters)
"""
function one_cycle!(loss, ps, data, opt, epochs, min_eta, max_eta; 
                    min_rho=0.0,
                    max_rho=0.0,
                    cycle_split=0.87,
                    cb = () -> ())
    iters = length(data) * epochs
    i1 = Int(iters * cycle_split)
    i2 = iters - i1
    etas = Iterators.flatten((TriangularCycle(min_eta, max_eta, i1), LinRange(min_eta, 0.0, i1)))
    if min_rho != 0.0 && max_rho != 0.0 && opt isa Union{Momentum, Nesterov, RMSProp, ADADelta}
        rhos = Iterators.flatten((TriangularCycle(max_rho, min_rho, i1), ConstantCycle(max_rho, i2)))
    else
        rhos = ConstantCycle(0.0, iters)
    end
    one_cycle!(loss, ps, data, opt, etas, rhos, cb = cb)
end

function one_cycle!(loss, ps, data, opt, etas, rhos; cb = () -> ())
    for (i, ((x, y), eta, rho)) in enumerate(zip(Iterators.cycle(data), etas, rhos))
        # update hyperparameters
        opt.eta = eta
        rho != 0.0 && (opt.rho = rho)
        l = loss(x, y)

        # update model
        Flux.back!(l)
        Flux.Optimise._update_params!(opt, ps)

        # call callback at the end of each epoch
        i % length(data) == 0 && cb()
    end
end


end # module
