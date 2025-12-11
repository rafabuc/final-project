"""Image transformation utilities."""

from torchvision import transforms


def get_image_transforms(mode: str = "train", image_size: int = 224):
    """
    Get image transformation pipeline for EfficientNet-B0.

    Args:
        mode: 'train' or 'val' - determines augmentation strategy
        image_size: Target image size (default: 224 for EfficientNet-B0)

    Returns:
        torchvision.transforms.Compose object
    """
    # EfficientNet-B0 ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        transform = transforms.Compose([

            # Critical change  RandomResizedCrop instead of Resize  from v 1.8 
            # Esto hace zoom in/out aleatorio y recortes.
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),

            #transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation(degrees=15),
            #increase rotation
            transforms.RandomRotation(degrees=20),

            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            #jitter more difficult
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # sometimes (20%) image in gray scale for not dependiong on color
            transforms.RandomGrayscale(p=0.2),


            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:  # val or test
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return transform
