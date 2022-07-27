class Pepper_diseases(Dataset):
    def __init__(self, img_path, label, transform):
        self.img_path = img_path
        self.transform = transform
        self.label = label

    def __getitem__(self, index):
        if self.img_path[index] not in train_dict:
            img = np.load(self.img_path[index])
        else:
            img = train_dict[self.img_path[index]]
        if self.transform is not None:
            img = self.transform(image=img)["image"]
            
        label = torch.tensor(self.label[index])
        
        return img, label - 1

    def __len__(self):
        return len(self.img_path)
    
    
    
SIZE_TRAIN = 512 # 448 # 512
SIZE_TEST = 512  # 448 # 512


def build_transforms(is_train=True):
    if is_train:
        transform = A.Compose([
                 # A.Resize(SIZE_TRAIN, SIZE_TRAIN),
                 A.RandomResizedCrop(SIZE_TRAIN, SIZE_TRAIN, p=0.3),
                 A.IAAPerspective(p=0.1, scale=(0.05, 0.15)),
                 A.VerticalFlip(p=0.5),
                 A.HorizontalFlip(p=0.5),
                 A.RandomRotate90(p=0.2),
                 A.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
                 A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=20,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.3,
                 ),
                 A.CoarseDropout(p=0.2),
                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                 ToTensorV2(),
                ])
    else:
        transform = A.Compose([
            # A.Resize(SIZE_TEST, SIZE_TEST),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    return transform
