function get_opt(opt)
    try
        getproperty(opt, :eta) !== nothing && (return opt)
    catch
        for x in opt
            try
                getproperty(x, :eta) !== nothing && (return x)
            catch
            end
        end
    end
    error("Optimiser $opt must have property :eta or contain an element which has field :eta")
end

accuracy = (m, labels) -> (x, y) -> mean(Flux.onecold(m(x), labels) .== Flux.onecold(y, labels))
loss = (m) -> (x, y) -> Flux.crossentropy(m(x), y)

function batchf(f::Function, batches::AbstractVector)
    res = 0.0
    for (x, y) in batches
        res += f(x, y)
    end
    res / length(batches)
end

