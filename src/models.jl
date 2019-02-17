#= fcm = Chain( =#
#=     # reshape into a matrix =#
#=     x -> reshape(x, :, size(x, 4)), =#
#=     Dense(28*28, 100, relu), =#
#=     Dense(100, 10), =#
#=     softmax, =#
#= ) |> gpu =#
#=  =#
#=  =#
#= convm = Chain( =#
#=     # First convolution, operating upon a 28x28 image =#
#=     Conv((3, 3), 1=>16, pad=(1,1), relu), =#
#=     x -> maxpool(x, (2,2)), =#
#=  =#
#=     # Second convolution, operating upon a 14x14 image =#
#=     Conv((3, 3), 16=>32, pad=(1,1), relu), =#
#=     x -> maxpool(x, (2,2)), =#
#=  =#
#=     # Third convolution, operating upon a 7x7 image =#
#=     Conv((3, 3), 32=>32, pad=(1,1), relu), =#
#=     x -> maxpool(x, (2,2)), =#
#=  =#
#=     # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N) =#
#=     # which is where we get the 288 in the `Dense` layer below: =#
#=     x -> reshape(x, :, size(x, 4)), =#
#=     Dense(288, 10), =#
#=  =#
#=     # Finally, softmax to get nice probabilities =#
#=     softmax, =#
#= ) |> gpu =#

# ResNet && Fixup/Mixup whatever it's called

#= struct BasicBlock =#
#=     shortcut =#
#=     conv1 =#
#=     bn1 =#
#=     conv2 =#
#=     bn2 =#
#=  =#
#=     function BasicBlock(channels::Pair{Int, Int}, stride=1) =#
#=         conv1 = Conv((3, 3), channels, stride=stride, pad=1) =#
#=         conv2 = Conv((3, 3), channels, stride=1, pad=1) =#
#=         bn1 = BatchNorm(channels[2]) =#
#=         bn2 = BatchNorm(channels[2]) =#
#=         if stride != 1 || channels[1] != channels[2] =#
#=             shortcut = Chain(Conv((1, 1), channels, stride=1), BatchNorm(channels[2])) =#
#=         else =#
#=             shortcut = identity =#
#=         end =#
#=         new(shortcut, conv1, bn1, conv2, bn2) =#
#=     end =#
#= end =#
#=  =#
#= Flux.@treelike BasicBlock =#
#=  =#
#= function (b::BasicBlock)(x::AbstractArray) =#
#=     sc = b.shortcut(x) =#
#=     x = b.conv1(x) =#
#=     x = relu.(b.bn1(x)) =#
#=     x = b.conv2(x) =#
#=     x = b.bn2(x) =#
#=     relu.(x) + sc =#
#= end =#
#=  =#
#= struct ResNet =#
#=     conv1 =#
#=     bn1 =#
#=     layer1 =#
#=     layer2 =#
#=     layer3 =#
#=     dense1 =#
#= end =#
#=  =#
#= function make_group_layer() =#
#= end =#

struct ResidualBlock
    conv_layers
    norm_layers
    shortcut
end

Flux.@treelike ResidualBlock

function ResidualBlock(filters, kernels::Array{Tuple{Int,Int}}, pads::Array{Tuple{Int,Int}}, strides::Array{Tuple{Int,Int}}, shortcut = identity)
    local conv_layers = []
    local norm_layers = []
    for i in 2:length(filters)
        push!(conv_layers, Conv(kernels[i-1], filters[i-1]=>filters[i], pad = pads[i-1], stride = strides[i-1]))
        push!(norm_layers, BatchNorm(filters[i]))
    end
    ResidualBlock(Tuple(conv_layers), Tuple(norm_layers), shortcut)
end

ResidualBlock(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, shortcut = identity) =
  ResidualBlock(filters, [(i,i) for i in kernels], [(i,i) for i in pads], [(i,i) for i in strides], shortcut)

function (block::ResidualBlock)(input)
    local value = copy.(input)
    for i in 1:length(block.conv_layers)-1
        value = relu.((block.norm_layers[i])((block.conv_layers[i])(value)))
    end
    relu.(((block.norm_layers[end])((block.conv_layers[end])(value))) + block.shortcut(input))
end

function BasicBlock(filters::Int, downsample::Bool = false, res_top::Bool = false)
    # NOTE: res_top is set to true if this is the first residual connection of the architecture
    # If the number of channels is to be halved set the downsample argument to true
    if !downsample || res_top
        return ResidualBlock([filters for i in 1:3], [3,3], [1,1], [1,1])
    end
    shortcut = Chain(Conv((3,3), filters÷2=>filters, pad = (1,1), stride = (2,2)), BatchNorm(filters))
    ResidualBlock([filters÷2, filters, filters], [3,3], [1,1], [1,2], shortcut)
end

function load_resnet(Block, layers, initial_filters::Int = 64, nclasses::Int = 1000)
    local top = []
    local residual = []
    local bottom = []

    push!(top, Conv((7,7), 3=>initial_filters, pad = (3,3), stride = (2,2)))
    push!(top, MaxPool((3,3), pad = (1,1), stride = (2,2)))

    for i in 1:length(layers)
        push!(residual, Block(initial_filters, true, i==1))
        for j in 2:layers[i]
            push!(residual, Block(initial_filters))
        end
        initial_filters *= 2
    end

    push!(bottom, MeanPool((7,7)))
    push!(bottom, x -> reshape(x, :, size(x,4)))
    push!(bottom, (Dense(512, nclasses)))
    push!(bottom, softmax)

    Chain(top..., residual..., bottom...)
end

struct ResNet18
    layers::Chain

    ResNet18(nclasses::Int) = new(load_resnet(BasicBlock, [2, 2, 2, 2], 64, nclasses) |> gpu)
end

Flux.@treelike ResNet18

(m::ResNet18)(x) = m.layers(x)

struct ResNet34
    layers::Chain

    ResNet34(nclasses::Int) = new(load_resnet(BasicBlock, [3, 4, 6, 3], 64, nclasses) |> gpu)
end

Flux.@treelike ResNet34

(m::ResNet34)(x) = m.layers(x)

