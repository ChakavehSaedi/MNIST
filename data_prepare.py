import torch
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt


class data_handler():
    def __init__(self):
        super(data_handler, self).__init__()

        self.fig = plt.figure()

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        """transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        training = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(training, batch_size=4, shuffle=True, num_workers=2)

        testing = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(testing, batch_size=4, shuffle=False, num_workers=2)"""

        self.train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True,
                            transform=torchvision.transforms.Compose(
                            [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                            batch_size=16, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=False, download=True,
                            transform=torchvision.transforms.Compose(
                            [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                            batch_size=16, shuffle=True)



        data_iterator = iter(self.train_loader)
        self.train_images, self.train_labels = data_iterator.next()
        #self.plot(self.train_images, self.train_labels, [])

        data_iterator = iter(self.test_loader)
        self.test_images, self.test_labels = data_iterator.next()


    def plot(self, images, labels, prediction):
        # plot 4 images to visualize the data
        rows = 2
        columns = 2
        for i in range(4):
            self.fig.add_subplot(rows, columns, i + 1)
            if prediction == []:
                plt.title(self.classes[labels[i]])
            else:
                plt.title('truth ' + self.classes[labels[i]] + ': predict ' + self.classes[prediction[i]])
            img = images[i] / 2 + 0.5  # this is for unnormalize the image
            img = torchvision.transforms.ToPILImage()(img)
            plt.imshow(img)
        plt.show()
        plt.gcf().clear()

