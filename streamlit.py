import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import PIL
from PIL import Image
import numpy as np
from labels import labels_dict
import easyocr

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




def car_recogniser(our_img):
    car_image = our_img

    trans = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(inplace=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # preprocessing for prediction image
    input = trans(car_image)
    input = input.view(1, 3, 400, 400)


    output = car_model(input)

    prediction = int(torch.max(output.data, 1)[1].numpy())

    # return prediction label
    predicted_val = ([value for value in labels_dict.values()][prediction])
    predicted_val

    #converting PIL object into numpy array for ocr
    new_array = np.array(car_image)
    reader = easyocr.Reader(['en'], gpu=False)
    bounds = reader.readtext(new_array, detail=0)
    bounds



def main():


    st.title("Car Model + License Plate Recognition")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Intelligent Car Park Management System</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = PIL.Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)


    if st.button("Recognise"):
        car_recogniser(our_image)







if __name__ == '__main__':
    main()