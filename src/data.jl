getarray(x) = Float32.(permutedims(channelview(x), (2, 3, 1)))

function make_minibatch(fx, fy, X, Y, idxs)
    sz = size(fx(X[1]))
    sz = length(sz) == 3 ? sz : (sz..., 1)
    #= @info sz =#
    X_batch = Array{Float32}(undef, sz..., length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = fx(X[idxs[i]])
    end
    Y_batch = fy(Y[idxs])
    return X_batch, Y_batch
end

function load_cifar10(;batch_size=128, train_val_split=0.95)
    @info("Loading CIFAR10 ...")
    fx = x -> getarray(x)
    fy = y -> onehotbatch(y, 1:10)

    imgs = trainimgs(CIFAR10)
    X = map(x -> x.img, imgs)
    Y = map(x -> x.ground_truth.class, imgs)

    idxs = Random.shuffle(1:length(X))
    train_idx_range = 1:Int(length(idxs) * train_val_split)
    val_idx_range = train_idx_range[end]+1:length(idxs)
    train_idxs = partition(train_idx_range, batch_size)
    train = [make_minibatch(fx, fy, X, Y, i) for i in train_idxs]

    val_idxs = partition(val_idx_range, batch_size)
    val = [make_minibatch(fx, fy, X, Y, i) for i in val_idxs]

    imgs = valimgs(CIFAR10)
    X = map(x -> x.img, imgs)
    Y = map(x -> x.ground_truth.class, imgs)
    test_idxs = partition(1:length(X), batch_size)
    test = [make_minibatch(fx, fy, X, Y, i) for i in test_idxs]

    return (train=gpu.(train), val=gpu.(val), test=gpu.(test))
end

function load_mnist(;batch_size=128, train_val_split=0.95)
    @info("Loading MNIST ...")
    fx = x -> Float32.(x)
    fy = y -> onehotbatch(y, 0:9)
    X = MNIST.images()
    Y = MNIST.labels()

    idxs = Random.shuffle(1:length(X))
    train_idx_range = 1:Int(length(idxs) * train_val_split)
    val_idx_range = train_idx_range[end]+1:length(idxs)
    train_idxs = partition(train_idx_range, batch_size)
    train = [make_minibatch(fx, fy, X, Y, i) for i in train_idxs]

    val_idxs = partition(val_idx_range, batch_size)
    val = [make_minibatch(fx, fy, X, Y, i) for i in val_idxs]

    X = MNIST.images(:test)
    Y = MNIST.labels(:test)
    test_idxs = partition(1:length(X), batch_size)
    test = [make_minibatch(fx, fy, X, Y, i) for i in test_idxs]

    return (train=gpu.(train), val=gpu.(val), test=gpu.(test))
end

