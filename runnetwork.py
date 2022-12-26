from copy import deepcopy
import elasticdeform
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from customdataset import CustomDataset
from spatialtransformernetwork import SpatialTransformerNetwork

class RunNetwork:
    def __init__(
        self,
        batch_size=64,
        num_workers=4,
        elastic_deform=True,
        sigma=30,
        points=3,
        zoom=4,
        epochs=20,
        learning_rate=0.01
    ) -> None:
        self.__batch_size = batch_size
        self.__num_workers = num_workers
        self.__elastic_deform = elastic_deform
        self.__sigma = sigma
        self.__points = points
        self.__zoom = zoom
        self.__epochs = epochs
        self.__learning_rate = learning_rate
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.__train_set = datasets.MNIST(root='.', train=True, download=True,
                                transform=self.__transform)
        self.__test_set = datasets.MNIST(root='.', train=False, download=True,
                                transform=self.__transform)
                            
        # Get dataloaders
        if elastic_deform:
            self.__train_loader = self.__get_deform_loader(train=True, mode=None)
            self.__unaltered_test_loader = self.__get_deform_loader(train=False, mode=0)
            self.__elastic_deform_test_loader = self.__get_deform_loader(train=False, mode=1)
            self.__zoom_test_loader = self.__get_deform_loader(train=False, mode=2)
            self.__test_loaders = [self.__unaltered_test_loader, 
                            self.__elastic_deform_test_loader, 
                            self.__zoom_test_loader]
        else:
            self.__train_loader = torch.utils.data.DataLoader(self.__train_set,
                            batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.__test_loaders = [torch.utils.data.DataLoader(
                datasets.MNIST(root='.', train=False, transform=self.__transform), 
                                batch_size=batch_size, shuffle=True, num_workers=num_workers)]

    def run(self):
        model = SpatialTransformerNetwork().to(self.__device)
        optimizer = optim.SGD(model.parameters(), lr=self.__learning_rate)

        train_loss = list()
        train_accuracy = list()
        test_loss = list()
        test_accuracy = list()

        for epoch in range(1, epochs + 1):
            (train_l, train_acc) = self.__train(model, optimizer, epoch)
            train_loss.append(train_l.item())
            train_accuracy.append(train_acc)
            (test_l, test_acc) = self.__test(model)
            test_loss.append(test_l)
            test_accuracy.append(test_acc)
        
        print(train_accuracy)
        print(train_loss)
        print(test_accuracy)
        print(test_loss)

        if elastic_deform:
            acc_unaltered = list()
            acc_elastic_deform = list()
            acc_zoom = list()
            for acc in test_accuracy:
                acc_unaltered.append(acc[0])
                acc_elastic_deform.append(acc[1])
                acc_zoom.append(acc[2])
            
            loss_unaltered = list()
            loss_elastic_deform = list()
            loss_zoom = list()
            for l in test_loss:
                loss_unaltered.append(l[0])
                loss_elastic_deform.append(l[1])
                loss_zoom.append(l[2])
            
            print(acc_unaltered)
            print(acc_elastic_deform)
            print(acc_zoom) 
            print(loss_unaltered)
            print(loss_elastic_deform)
            print(loss_zoom)

    def __get_deform_loader(self, train, mode):
        image_index = 0
        label_index = 1

        dataset = CustomDataset([], [])

        if train:
            for data in self.__train_set:
                # first add the unaltered data
                dataset.data.append(deepcopy(data[image_index]))
                dataset.targets.append(deepcopy(data[label_index]))
                # add the random grid deform to the dataset
                image = data[image_index].squeeze().numpy()
                label = data[label_index]
                image_ed = elasticdeform.deform_random_grid(
                    image, 
                    sigma=self.__sigma, 
                    points=self.__points)
                image_ed = torch.unsqueeze(torch.from_numpy(image_ed), 0)
                dataset.data.append(deepcopy(image_ed))
                dataset.targets.append(deepcopy(label))
            
                # add the zoom deform to the dataset
                displacement = np.full((2, 3, 3), 0)
                image_z = elasticdeform.deform_grid(
                    image,
                    displacement,
                    prefilter=False,
                    zoom=0.25)
                image_z = torch.unsqueeze(torch.from_numpy(image_z), 0)
                dataset.data.append(deepcopy(image_z))
                dataset.targets.append(deepcopy(label))
            
        else:
            for data in self.__test_set:
                image = data[image_index].squeeze().numpy()
                label = data[label_index]

                if mode == 0:
                    # add the unaltered data
                    dataset.data.append(deepcopy(data[image_index]))
            
                elif mode == 1:
                    # add the random grid deform to the dataset
                    image_ed = elasticdeform.deform_random_grid(
                        image, 
                        sigma=self.__sigma, 
                        points=self.__points)
                    image_ed = torch.unsqueeze(torch.from_numpy(image_ed), 0)
                    dataset.data.append(deepcopy(image_ed))
    
                else:
                    # add the zoom deform to the dataset
                    displacement = np.full((2, 3, 3), 0)
                    image_z = elasticdeform.deform_grid(
                        image,
                        displacement,
                        prefilter=False,
                        zoom=0.25)
                    image_z = torch.unsqueeze(torch.from_numpy(image_z), 0)
                    dataset.data.append(deepcopy(image_z))
                
                dataset.targets.append(deepcopy(label))
    
        loader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)
        
        return loader

    def __train(self, model, optimizer, epoch):
        model.train()
        correct = 0
        total = len(self.__train_set)

        for batch_idx, (data, target) in enumerate(self.__train_loader):
            data, target = data.to(self.__device), target.to(self.__device)

            optimizer.zero_grad()
            output = model(data)
            
            for i in range(len(target)):
                if(target[i] == torch.argmax(output[i])):
                    correct += 1
        
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 500 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.__train_loader.dataset),
                        100. * batch_idx / len(self.__train_loader), loss.item(), correct / total))

        return (loss, correct / total)

    def __test(self, model):
        with torch.no_grad():
            model.eval()
            test_losses = list()
            test_accuracies = list()

            for i in range (len(self.__test_loaders)): 
                test_loss = 0
                correct = 0
                for data, target in self.__test_loaders[i]:
                    data, target = data.to(self.__device), target.to(self.__device)
                    output = model(data)
        
                    # sum up batch loss
                    test_loss += F.nll_loss(output, target, size_average=False).item()
                    # get the index of the max log-probability
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
        
                test_loss /= len(self.__test_loaders[i].dataset)
                test_accuracy = correct / len(self.__test_loaders[i].dataset)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
                print('\nTest set: {:n}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                        .format(i, test_loss, correct, len(self.__test_loaders[i].dataset),
                                test_accuracy * 100))
                
        return (test_losses, test_accuracies)

    def __convert_image_np(self, inp):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp


    def __visualize_stn(self, model):
        '''We want to visualize the output of the spatial transformers layer
        after the training, we visualize a batch of input images and
        the corresponding transformed batch using STN.'''
        image_index = 0
        label_index = 1

        with torch.no_grad():
            for test_loader in self.__test_loaders:
                data = next(iter(test_loader))
                num = 0
            
                for image in data[0]:
                    if num < 10:
                        in_grid = self.__convert_image_np(
                            torchvision.utils.make_grid(image))
                        plt.imshow(in_grid)
                        plt.show()
                    
                        # input_tensor = image.todevice()
                        transformed_input_tensor = model.stn(torch.unsqueeze(image.to(self.__device), 0))
                        out_grid = self.__convert_image_np(
                            torchvision.utils.make_grid(transformed_input_tensor).cpu())
                        plt.imshow(out_grid)
                        plt.show()
                        print("\n")
                        num += 1

    def __visualize_elastic_deformation(self):
        image_index = 0
        label_index = 1
        
        with torch.no_grad():
                data = next(iter(self.__test_loader))[0]
                print(data.size())
                
                for image in data:
                    print()
                    in_grid = self.__convert_image_np(
                        torchvision.utils.make_grid(image))
                    plt.imshow(in_grid)
                    plt.show()
                
                    input_tensor = image.cpu()
                    image = image.squeeze().numpy()
                
                    displacement = np.full((2, 3, 3), 0)
                    image_z = elasticdeform.deform_grid(
                        image,
                        displacement,
                        prefilter=False,
                        zoom=0.25)
                    image_z = torch.unsqueeze(torch.from_numpy(image_z), 0)
                
                
                    out_grid = self.__convert_image_np(
                        torchvision.utils.make_grid(image_z))
                    plt.imshow(out_grid)
                    plt.show()
                    print()
    
RunNetwork().run()