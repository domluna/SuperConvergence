"""
"""
struct BasicBlock
    conv1
    conv2
    shortcut
    bias1
    bias2
    bias3
    bias4
    mults
end

Flux.@treelike BasicBlock

function BasicBlock(ch::Pair{Int,Int}, stride::Int)
    conv1 = Conv((3, 3), ch, stride=stride, pad=1)
    conv2 = Conv((3, 3), ch[2] => ch[2], stride=1, pad=1)
    mults = Flux.param(ones(Float32, (1, 1, 1, 1)))
    bias1 = Flux.param(zeros(Float32, (1, 1, 1, 1)))
    bias2 = Flux.param(zeros(Float32, (1, 1, 1, 1)))
    bias3 = Flux.param(zeros(Float32, (1, 1, 1, 1)))
    bias4 = Flux.param(zeros(Float32, (1, 1, 1, 1)))
    #= biases = Flux.param(zeros(Float32, (4, 4))) =#
    #= biases = [Flux.param(zeros(Float32, (0, 0, 0, 0))) for _ in 1:4] =#
    shortcut = ch[1] == ch[2] ? identity : Conv((1, 1), ch, stride=stride, pad=0)
    #= BasicBlock(conv1, conv2, shortcut, biases, mults) =#
    BasicBlock(conv1, conv2, shortcut, bias1, bias2, bias3, bias4, mults)
end

function (b::BasicBlock)(x::AbstractArray)
    sc = b.shortcut(x)
    x = x .+ b.bias1
    x = relu.(b.conv1(x) .+ b.bias2) .+ b.bias3
    x = b.mults .* b.conv2(x) .+ b.bias4
    relu.(x .+ sc)
end

"""
Initialize the weights using the He init
rescaled by a Fixup multiplier.

Fixup by Hongyi Zhang, Yann N. Dauphin, and Tengyu Ma (ICLR 2019)

1. Initialize the classification layer and the last layer of each residual branch to 0.
2. Initialize every other layer using a standard method (e.g., He et al. (2015)), and scale only
the weight layers inside residual branches by L^(−1 / 2m−2).
3. Add a scalar multiplier (initialized at 1) in every branch and a scalar bias (initialized at
0) before each convolution, linear, and element-wise activation layer.

2) on its own is sufficient to compete with ResNet. 1) and 3) further improve regularization.
"""
function fixup_init!(b::BasicBlock, layer_index::Int)
    n = prod((size(b.conv1.weight)[1:2]..., size(b.conv1.weight)[end]))
    @info "fixup" n layer_index sqrt(2 / n) layer_index^-0.5 * sqrt(2 / n)
    b.conv1.weight.data .= randn(size(b.conv1.weight)) * (layer_index^-0.5 * sqrt(2 / n))
    b.conv2.weight.data .= 0
    if b.shortcut !== identity 
        n = prod((size(b.shortcut.weight)[1:2]..., size(b.shortcut.weight)[end]))
        b.shortcut.weight.data .= randn(size(b.shortcut.weight)) * sqrt(2 / n)
    end
    return nothing
end

function fixup_init!(blocks, layer_index::Int)
    for b in blocks
        fixup_init!(b, layer_index)
    end
    return nothing
end

function make_block_group(depth::Int, ch::Pair{Int,Int}, stride::Int)
    group = BasicBlock[]
    for i in 1:depth
        ch = i == 1 ? ch : ch[2] => ch[2]
        stride = i == 1 ? stride : 1
        push!(group, BasicBlock(ch, stride))
    end
    group
end

struct WideResNet
    layers::Chain
end

Flux.@treelike WideResNet


"""
    WideResNet(depth, width, in_channels=3)

Popular sizes (depth-width), in increasing representation ability
are 40-4, 16-10, and 28-10.

Note: Increasing the width (1:12) has positive results until a certain depth (40)
in which case increases the width has adverse results, i.e. 40-8 performs worse
than 28-8.
"""
function WideResNet(depth::Int, width::Int, n_classes::Int, in_channels::Int=3)
    # This calculation is from the initial source
    # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py#L7-L8
    #
    # I'm not entirely sure why we subtract 4 from `depth`. 
    #
    # A basic block has 2 convolutional layers so modulus by 6 
    # makes sense since there are 3 groups.
    @assert (depth - 4) % 6 == 0
    n = Int((depth - 4) / 6)

    @info "Creating a $depth-$width WideResNet"

    # output channels for each group of blocks
    out_channels = [16, 16*width, 32*width, 64*width]

    # initial convolutional layer
    conv1 = Conv((3, 3), in_channels => out_channels[1], stride=1, pad=1)

    conv2 = make_block_group(n, out_channels[1] => out_channels[2], 1)
    conv3 = make_block_group(n, out_channels[2] => out_channels[3], 2)
    conv4 = make_block_group(n, out_channels[3] => out_channels[4], 2)

    # don't count initial conv group `conv1`
    layer_index = n * 3
    fixup_init!(conv2, layer_index)
    fixup_init!(conv3, layer_index)
    fixup_init!(conv4, layer_index)

    pool = MeanPool((8, 8), stride=1)
    fc = Dense(out_channels[4], n_classes)
    fc.W.data .= 0
    fc.b.data .= 0

    WideResNet(Chain(conv1, conv2..., conv3..., conv4..., 
                     pool, x -> reshape(relu.(x), :, size(x, 4)), fc, softmax))
end

function (wrn::WideResNet)(x::AbstractArray)
    wrn.layers(x)
end


