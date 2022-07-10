
import streamlit as st
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor

st.sidebar.selectbox("Menu", ("Parameter weights", "Reuse Evaluation", "Machine Learning Model"))

header = st.container()
parameter_weight = st.container()
logistic_feasibility = st.container()
image_classification = st.container()
structural_performance = st.container()
life_cycle_assessment = st.container()
general_decision = st.container()

def yes_or_no(userinput):
    if userinput == 'Yes':
        input_value = 1
    else:
        input_value = 0
    return input_value

def no_or_yes(userinput):
    if userinput == 'No':
        input_value = 1
    else:
        input_value = 0
    return input_value

def status(performance, treshold):
    if performance >= treshold:
        return 'Pass'
    else:
        return 'Not passed'

def show_batch(images):
    for image in images:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=10).permute(1, 2, 0))
        break

## To seamlessly use a GPU, if one is available

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def predict_image(img, model):
    # Convert to a batch of 1
    # xb = to_device(img.unsqueeze(0), device)
    xb = img.unsqueeze(0)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return preds[0].item()



with header:
    st.header('Decision making framework for reuse of structural steel elements')
    st.write('This dashboard evaluates the reuse potential of existing steel structures considering '
             'the logistic feasibility, the structural visual inspection, and the structural performance.')

with logistic_feasibility:
    st.header('Logistic Feasibility')

    selec_col, displ_col = st.columns(2)

    # st.select_slider('Is the element easy to handle?', options=['very easy', 'easy', 'difficult', 'very difficult'], index=1)
    item_weight = selec_col.selectbox('Weight of the structural element',
                               options=['Very light [< 0.1 ton]', 'Light [0.1 - 0.2 ton]',
                                        'Heavy [0.2 - 0.5 ton]', 'Very heavy [> 0.5 ton]'], index=1)

    if item_weight == 'Very light [< 0.1 ton]':
        item_weight_value = 1
    elif item_weight == 'Light [0.1 - 0.2 ton]':
        item_weight_value = 2
    elif item_weight == 'Heavy [0.2 - 0.5 ton]':
        item_weight_value = 3
    else:
        item_weight_value = 4

    easy_handle = selec_col.selectbox('Ease to handle, transport, store, and process?', options=['Yes', 'No'], index=0)
    exist_infrastructure = selec_col.selectbox('Availability of dismantle-sort-repair infrastructure',
                                               options=['Yes', 'No'], index=0)

    special_protection = displ_col.selectbox('Special protection is needed for transportation?',
                                             options=['Yes', 'No'], index=0)

    dismantle_phase = displ_col.selectbox('Dismantle phase is compatible with demolition work', options=['Yes', 'No'], index=0)

    storage_availability = displ_col.selectbox('Availability of storage', options=['Yes', 'No'], index=0)

    logistic_performance = (3*(1-(item_weight_value/4)) + 3*yes_or_no(easy_handle) + 4*yes_or_no(exist_infrastructure) + \
                            1*no_or_yes(special_protection) + 3*yes_or_no(dismantle_phase) + 3*yes_or_no(storage_availability))

    #displ_col.write('The logistic feasibility performance is: {:0.2f}%'.format((logistic_performance/4)*100))
    #displ_col.write(f'The logistic feasibility status: {status((logistic_performance / 3)*100, 75)}')

with image_classification:
    st.header('Structural Visual Inspection')
    st.write('This criteria is supplemented with an automated image classification CNN tool. Images can be uploaded and parameters '
             'related to corrosion, connection types and damage status are evaluated automatically.')

    input_image = st.radio('How would you like to perform the structural visual inspection?', ('I do have image files', 'I do not have image files'), index=1)

    if input_image == 'I do have image files':

        image_file = st.file_uploader("Please upload structural steel images here", type=["jpg", "png", "jpeg", "webp"], accept_multiple_files=True)

        from itertools import cycle
        cols = cycle(st.columns(4))

        if image_file == []:
            pass
            #corrosion = st.selectbox('Is the structural element corroded?', options=['Yes', 'No'], index=0)
        else:
            # instantiate the CNN models
            # reloading the trained model
            st.set_option('deprecation.showfileUploaderEncoding', False)

            #@st.cache(allow_output_mutation=True)

            # Deep Convolutional Neural Network (CNN)
            # Define ImageClassificationBase class which contains helper methods for training & validation

            class ImageClassificationBase(nn.Module):
                def training_step(self, batch):
                    images, labels = batch
                    out = self(images)  # Generate predictions
                    loss = F.cross_entropy(out, labels)  # Calculate loss
                    return loss

                def validation_step(self, batch):
                    images, labels = batch
                    out = self(images)  # Generate predictions
                    loss = F.cross_entropy(out, labels)  # Calculate loss
                    acc = accuracy(out, labels)  # Calculate accuracy
                    return {'val_loss': loss.detach(), 'val_acc': acc}

                def validation_epoch_end(self, outputs):
                    batch_losses = [x['val_loss'] for x in outputs]
                    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
                    batch_accs = [x['val_acc'] for x in outputs]
                    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
                    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

                def epoch_end(self, epoch, result):
                    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                        epoch + 1, result['train_loss'], result['val_loss'], result['val_acc']))

            def accuracy(outputs, labels):
                _, preds = torch.max(outputs, dim=1)
                return torch.tensor(torch.sum(preds == labels).item() / len(preds))

            # Use nn.Sequential to chain the layers and activation functions into a single network architecture

            class CorrosionModel(ImageClassificationBase):
                def __init__(self):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                        nn.ReLU(),  # activation function doesn't change the size
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

                        nn.Flatten(),
                        nn.Linear(256 * 4 * 4, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 2))  # binary classification: corroded, not-corroded

                def forward(self, xb):
                    return self.network(xb)

            class ConnectionModel(ImageClassificationBase):
                def __init__(self):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                        nn.ReLU(),  # activation function doesn't change the size
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

                        nn.Flatten(),
                        nn.Linear(256 * 4 * 4, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 2))  # binary classification: corroded, not-corroded

                def forward(self, xb):
                    return self.network(xb)

            class DamageModel(ImageClassificationBase):
                def __init__(self):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                        nn.ReLU(),  # activation function doesn't change the size
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

                        nn.Flatten(),
                        nn.Linear(256 * 4 * 4, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 2))  # binary classification: corroded, not-corroded

                def forward(self, xb):
                    return self.network(xb)

            # instantiate corrosion model
            model_1 = CorrosionModel()
            model_1.load_state_dict(torch.load('CNN_Corrosion.pth', map_location='cpu'))
            model_1.eval()

            # instantiate connection model
            model_2 = ConnectionModel()
            model_2.load_state_dict(torch.load('CNN_Connection_type.pth', map_location='cpu'))
            model_2.eval()

            # instantiate damage model
            model_3 = DamageModel()
            model_3.load_state_dict(torch.load('CNN_damage.pth', map_location='cpu'))
            model_3.eval()

            transform = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
            from numpy import array
            classes_1 = array(['Corroded', 'Not corroded'])
            classes_2 = array(['Bolted', 'Welded'])
            classes_3 = array(['Damaged', 'Not damaged'])

            from PIL import Image

            cols = cycle(st.columns(4))

            no_of_images = 0
            no_of_corroded_images = 0
            no_of_bolted_images = 0
            no_of_damaged_images = 0

            for image in image_file:
                no_of_images += 1
                img = Image.open(image)
                img = transform(img)

                label_1 = predict_image(img, model_1)
                label_2 = predict_image(img, model_2)
                label_3 = predict_image(img, model_3)

                label_1 = int(label_1)
                label_2 = int(label_2)
                label_3 = int(label_3)

                label_1 = classes_1[label_1]
                label_2 = classes_2[label_2]
                label_3 = classes_3[label_3]

                if label_1 == 'Corroded': no_of_corroded_images += 1
                if label_2 == 'Bolted': no_of_bolted_images += 1
                if label_3 == 'Damaged': no_of_damaged_images += 1
                next(cols).image(image, width=150, caption=label_1+'/'+label_2+'/'+label_3)

            selec_col, displ_col = st.columns(2)
            selec_col.write('Depending on the loaded images:')
            selec_col.write('{:0.2f}% of the images are corroded.'.format((no_of_corroded_images /no_of_images) * 100))
            selec_col.write('{:0.2f}% of the joints are bolted connection.'.format((no_of_bolted_images /no_of_images)*100))
            selec_col.write('{:0.2f}% of the images are damaged.'.format((no_of_damaged_images /no_of_images)*100))

            selec_col, displ_col = st.columns(2)

            composite_connection = selec_col.selectbox('Are there steel-concrete composite connections?',
                                                       options=['Yes', 'No'], index=0)
            fire_protection = selec_col.selectbox('Is there fire protection on the element?',
                                                  options=['Yes', 'No'], index=0)
            sufficient_amount = displ_col.selectbox('Availability of sufficient amount of potential reusable elements?',
                                                    options=['Yes', 'No'], index=0)

            geometry_check = displ_col.selectbox('Does the element pass standard geometric check without modification?',
                                                 options=['Yes', 'No'], index=0)

            image_classification_performance = (
                        3 * (no_of_bolted_images /no_of_images) + \
                        4 * (no_of_corroded_images /no_of_images) + \
                        4 * (no_of_damaged_images /no_of_images) + \
                        1 * no_or_yes(composite_connection) + 1 * yes_or_no(fire_protection) + \
                        2 * yes_or_no(sufficient_amount) + 2 * yes_or_no(geometry_check))

    else:
        selec_col, displ_col = st.columns(2)

        connection_type = selec_col.slider('Connection type [fully welded - fully bolted]?', min_value=0.25, max_value=1.0, value=0.5, step=0.25)

        #st.write('Please select the distribution of the connections type.')

        #box1, box2, box3, box4 = selec_col.columns(4)
        #option1 = box1.checkbox('all welded joints')
        #option2 = box2.checkbox('mostly welded joints')
        #option3 = box3.checkbox('mostly bolted joints')
        #option4 = box4.checkbox('all bolted joints')


        corrosion = selec_col.selectbox('Is the element corroded?', options=['Yes', 'No'], index=0)
        damage = selec_col.selectbox('Is the element damaged or distorted?', options=['Yes', 'No'], index=0)
        composite_connection = selec_col.selectbox('Are there steel-concrete composite connections?',
                                                   options=['Yes', 'No'], index=0)
        fire_protection = displ_col.selectbox('Is there fire protection on the element?',
                                              options=['Yes', 'No'], index=0)
        sufficient_amount = displ_col.selectbox('Availability of sufficient amount of potential reusable elements?', options=['Yes', 'No'], index=0)

        geometry_check = displ_col.selectbox('Does the element pass standard geometric check without modification?', options=['Yes', 'No'], index=0)

        image_classification_performance = (3*(connection_type) + 4*no_or_yes(corrosion) + 4*no_or_yes(damage) + \
                                           1*no_or_yes(composite_connection) + 1*yes_or_no(fire_protection) + \
                                           2*yes_or_no(sufficient_amount) + 2*yes_or_no(geometry_check))

        #displ_col.write('The structural element condition is: {:0.2f}%'.format((image_classification_performance/5)*100))
        #displ_col.write(f'The structural element status: {status((image_classification_performance/5)*100, 75)}')

with structural_performance:
    st.header('Structural performance')

    selec_col, displ_col = st.columns(2)

    data_quality = selec_col.selectbox('Quality of available data?',
                                options=['No documentation',
                                         'Only drawings available',
                                         'Drawings and calculation report available',
                                         'All detailed documentation available'], index=1)

    if data_quality == 'No documentation':
        data_quality_value = 1
    elif data_quality == 'Only drawings available':
        data_quality_value = 2
    elif data_quality == 'Drawings and calculation report available':
        data_quality_value = 3
    else:
        data_quality_value = 4

    construction_period = selec_col.selectbox('Is the building designed and constructed after year 2005?', options=['Yes', 'No'], index=0)
    maintenance = selec_col.selectbox('Did the structure has maintenance before?', options=['Yes', 'No'], index=0)
    purpose = displ_col.selectbox('Is the structural element unique for its purpose?', options=['Yes','No'], index=0)
    testing = displ_col.selectbox('Is it possible to conduct sample testing?', options=['Yes','No'], index=0)

    structural_performance_value = (4*(data_quality_value/4) + 2*yes_or_no(construction_period) + 3*yes_or_no(maintenance) + \
                                  3*no_or_yes(purpose) + 3*yes_or_no(testing))

    #displ_col.write('The structural performance is: {:0.2f}%'.format((structural_performance_value/3)*100))
    #displ_col.write(f'The structural performance status: {status((structural_performance_value/3)*100, 85)}')

with life_cycle_assessment:
    st.header('Life cycle assessment (LCA)')
    st.write('Please select how you prefer to input the weight of the elements.')

    box1, box2, box3 = st.columns(3)
    option1 = box1.checkbox('Weight of single element')
    option2 = box2.checkbox('Dimension of single element')
    option3 = box3.checkbox('Bulk material weight')

    if option1:
        box1, box2 = st.columns([3,1])
        weight = box1.number_input('Weight of single element (kg)', help='Please insert valid value')
        items = box2.number_input('Number of items', help='Please insert valid value')

        total_weight = int(weight) * int(items)

    elif option2:
        box1, box2, box3, box4, box5 = st.columns(5)
        height = box1.text_input('Height of the element [mm]', value='0')
        width = box2.text_input('Width of the element [mm]', value='0')
        length = box3.text_input('Length of the element [m]', value='0')
        unit_weight = box4.text_input('Material unit weight [kg/m3]', value='0')
        items = box5.text_input('Number of items', value='0')

        total_weight = (int(height)/1000 * int(width)/1000 * int(length)) * int(unit_weight) * int(items)

    elif option3:
        total_weight = st.text_input('Bulk weight of the material [kg]')

    checked = st.button('Compute the embodied carbon, kgCO2 e')

    if checked:
        st.write('According to the cradle-to-cradle life cycle assessment, the embodied carbon at different stage is reported below.')
        st.write('Total weight of structural element:', float(total_weight), 'kg')
        st.write('Product stage A1-A3:', float(total_weight) * 1.13, 'kgCO2 e')
        #st.write('Construction process stage A4-A5 is:', total_weight* )
        #st.write('Usage stage B1-B7 is:', total_weight *)
        st.write('End of life stage C1-C4:', float(total_weight) * 0.018, 'kgCO2 e')
        st.write('Reuse, recycle and recovery stage D:', float(total_weight) * -0.413, 'kgCO2 e')

with general_decision:
    st.header('General suggestion')
    # logistic feasibility
    st.write('The logistic feasibility performance is: {:0.2f}%'.format((logistic_performance/17)*100))
    st.write(f'The logistic feasibility status: {status((logistic_performance/17)*100, 75)}')

    # structural image classification task
    st.write('The structural element condition is: {:0.2f}%'.format((image_classification_performance/17)*100))
    st.write(f'The structural element status: {status((image_classification_performance/17)*100, 70)}')

    # structural performance
    st.write('The structural performance is: {:0.2f}%'.format((structural_performance_value/15)*100))
    st.write(f'The structural performance status: {status((structural_performance_value/15)*100, 80)}')


    if(status((image_classification_performance/17)*100, 75) == 'Pass' and status((logistic_performance / 17)*100, 75) == 'Pass' and status((structural_performance_value/15)*100, 80) == 'Pass' ):
        st.success('Based on the input evaluation, the final suggestion is to: Dismantle - Reuse')
        #st.success('The overall reusability analysis shows {:0.3f}%'.format(((structural_performance_value/3)*100 + (image_classification_performance/5)*100 + (logistic_performance / 3)*100)/3))
    else:
        st.success('Based on the input evaluation, the final suggestion is to: Demolition - Recycle')