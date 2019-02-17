getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))

function load_cifar10(;batch_size=128, train_val_split=0.95)
    @info("Loading CIFAR10 ...")
    X = trainimgs(CIFAR10)
    img_size = size(getarray(X[1].img))
    idxs = Random.shuffle(1:length(X))
    train_idxs = idxs[1:(Int(length(X) * train_val_split))]

    # training set
    train_imgs = [getarray(X[i].img) for i in train_idxs]
    train_labels = onehotbatch([X[i].ground_truth.class for i in train_idxs], 1:10)
    train = gpu.([(cat(train_imgs[i]..., dims = 4), train_labels[:, i]) for i in partition(1:length(train_imgs), batch_size)])

    # validation set
    val_idxs = idxs[length(train_idxs)+1:length(X)]
    val_imgs = Array{Float32}(undef, img_size..., length(val_idxs))
    for (i, idx) in enumerate(val_idxs)
        val_imgs[:, :, :, i] = getarray(X[idx].img)
    end
    val_labels = onehotbatch([X[i].ground_truth.class for i in val_idxs], 1:10)
    val = gpu.((val_imgs, val_labels))

    # test set
    X = valimgs(CIFAR10)
    test_imgs = Array{Float32}(undef, img_size..., length(X))
    for i in 1:length(X)
        test_imgs[:, :, :, i] = getarray(X[i].img)
    end
    test_labels = onehotbatch([X[i].ground_truth.class for i in 1:length(X)], 1:10)
    test = gpu.((test_imgs, test_labels))
    return (train=train, val=val, test=test)
end

function load_mnist(;batch_size=128)
    @info("Loading MNIST ...")
    X = MNIST.images()
    y = MNIST.labels()
    train_imgs = [Float32.(X[i]) for i in 1:length(X)]
    train_labels = onehotbatch(y, 0:9)
    train = gpu.([(cat(train_imgs[i]..., dims = 4), train_labels[:, i]) for i in partition(1:length(train_imgs), batch_size)])


    X = MNIST.images(:test)
    y = MNIST.labels(:test)
    test_imgs = Array{Float32}(undef, size(X[1])..., 1, length(X))
    for i in 1:length(X)
        test_imgs[:, :, :, i] = Float32.(X[i])
    end
    test_labels = onehotbatch(y, 0:9)
    test = gpu.((test_imgs, test_labels))
    return (train=train, test=test)
end

