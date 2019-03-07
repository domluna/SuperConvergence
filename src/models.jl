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

