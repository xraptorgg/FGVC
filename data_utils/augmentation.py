# Code to create data augmentation transforms



MEANS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.485, 0.456, 0.406),
    'tinyin': (0.4802, 0.4481, 0.3975),
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5070, 0.4865, 0.4409),
    'svhn': (0.4377, 0.4438, 0.4728),
    'cub': (0.3659524, 0.42010019, 0.41562049)
}

STDS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.229, 0.224, 0.225),
    'tinyin': (0.2770, 0.2691, 0.2821),
    'cifar10': (0.2470, 0.2435, 0.2616),
    'cifar100': (0.2673, 0.2564, 0.2762),
    'svhn': (0.1980, 0.2010, 0.1970),
    'cub': (0.07625843, 0.04599726, 0.06182727)
}



def standard_transform(args, is_train):
    mean = MEANS[args.dataset]
    std = STDS[args.dataset]

    t = []

    if is_train:
        # square resize + random crop
        t.append(transforms.Resize((args.resize_size, args.resize_size), interpolation = transforms.InterpolationMode.BICUBIC))
        t.append(transforms.RandomCrop(args.image_size))
        
        #horizontal flip
        t.append(transforms.RandomHorizontalFlip())

        #trivial augmentation
        t.append(transforms.TrivialAugmentWide())
    else:
        # square resize + center crop
        t.append(transforms.Resize((args.test_resize_size, args.test_resize_size), interpolation = transforms.InterpolationMode.BICUBIC))
        t.append(transforms.CenterCrop(args.image_size))

    t.append(transforms.ToTensor())

    t.append(transforms.Normalize(mean = mean, std = std))
    
    if is_train and args.re_prob > 0:
        t.append(transforms.RandomErasing(p = args.re_prob, scale = (0.02, args.re_max_area), ratio = (args.re_min_ratio, 3.3)))

    transform = transforms.Compose(t)

    print(transform)
    return transform