import unittest
import torch
import torchvision
import torchvision.transforms as transforms
from modules.cut_out import Cutout

from matplotlib import pyplot as plt
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        train_transform = transforms.Compose([])
        # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        # train_transform.transforms.append(normalize)
        train_transform.transforms.append(Cutout(n_holes=1, length=16))

        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                                download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                  shuffle=True, num_workers=4)
        ds_train, l = next(iter(trainloader))


        fig = plt.figure(figsize=(8, 8));
        columns = 4
        rows = 2
        for i in range(1, columns * rows + 1):
            img_xy = np.random.randint(len(ds_train));
            img = ds_train[img_xy][:, :, :].numpy()
            img=np.transpose(img,[1,2,0])
            fig.add_subplot(rows, columns, i)
            # plt.title(labels_map[int(ds_train[img_xy][1].numpy())])
            plt.axis('off')
            plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    unittest.main()
