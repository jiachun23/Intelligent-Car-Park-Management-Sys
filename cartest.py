import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from pathlib import Path
from labels import labels_dict
import easyocr
from PIL import ImageDraw



def default_device():
    '''Indicate availablibity of GPU, otherwise return CPU'''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(tensorss, device):
    '''Move tensor to chosen device'''
    if isinstance(tensorss, (list, tuple)):
        return [to_device(x, device) for x in tensorss]
    return tensorss.to(device, non_blocking=True)


class DeviceDataloader():
    '''Wrap DataLoader to move the model to device'''

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        '''Yield batch of data after moving the data to a device'''
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        '''Return number of batches'''
        return len(self.dl)


# Check available device type
device = default_device()
device


class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        image, label = batch
        out = self(image)  # prediction generated
        loss = F.cross_entropy(out, label)  # Calculate loss using cross_entropy
        return loss

    def validation_step(self, batch):
        image, label = batch
        out = self(image)  # predictioon generated
        loss = F.cross_entropy(out, label)  # Calculate loss using cross_entropy
        acc = accuracy(out, label)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, output):
        '''at the end of epoch, return average score (accuracy and cross entropy loss)'''
        batch_loss = [x['val_loss'] for x in output]
        epoch_loss = torch.stack(batch_loss).mean()

        batch_accs = [x['val_acc'] for x in output]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        '''Print out the score (accuracy and cross entropy loss) at the end of the epoch'''
        # result recorded using evaluate function
        print("Epoch [{}], train_loss: {:}, val_loss: {:}, val_acc: {:}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class Resnet50(MultilabelImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 196)

    def forward(self, xb):
        return self.network(xb)


model = to_device(Resnet50(), device)

path = "car_model.pt"
car_model = torch.load(path, map_location=torch.device('cpu'))

trans = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(inplace=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


SOURCE_PATH = "C:/Users/USER1/Desktop/smart intelligent car park/testing_sample"



def car_recogniser():
    root = Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilename(initialdir=SOURCE_PATH, title="Select a file",
                                               filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
    img_path = root.filename
    car_image = Image.open(Path(img_path))

    # preprocessing for prediction image
    input = trans(car_image)
    input = input.view(1, 3, 400, 400)

    output = car_model(input)

    prediction = int(torch.max(output.data, 1)[1].numpy())


    reader = easyocr.Reader(['en', 'en'])

    # Doing OCR. Get bounding boxes.
    bounds = reader.readtext(img_path)


    def draw_boxes(image, bounds, color='red', width=2):
        draw = ImageDraw.Draw(image)
        for bound in bounds:
            p0, p1, p2, p3 = bound[0]
            draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
        return image


    #draw_boxes(car_image, bounds)
    # canvas = Canvas(root, width=300, height=300)
    # canvas.pack()
    # img_canvas = ImageTk.PhotoImage(Image.open(img_path))
    # canvas.create_image(20, 20, anchor=NW, image=img_canvas)
    print(bounds)
    # return prediction label
    print(([value for value in labels_dict.values()][prediction]))
    root.mainloop()



car_recogniser()
