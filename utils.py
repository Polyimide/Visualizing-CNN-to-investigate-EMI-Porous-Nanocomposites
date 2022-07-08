from torchvision import transforms
import matplotlib.pyplot as plt

image_size = [224, 224]
data_transform = {'train': transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                  'test': transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                    ),
                  'vision': transforms.Compose([
                    # transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                    )
                }


def show_examples(test_loader, data_train):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_targets)
    print(example_data.shape)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data_train[-i][0].transpose(0, 2), interpolation='none')
        plt.title("Ground Truth: {}".format(data_train[-i][1]))
        plt.xticks([])
        plt.yticks([])
    plt.show()