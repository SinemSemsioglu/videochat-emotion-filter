import torch
import numpy as np
from processing.emonet.models import EmoNet
from processing.emonet.evaluation import evaluate, evaluate_flip
from torchvision import transforms 

net = None
# at beginning of the script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init():
    global net

    # Loading the model 
    state_dict_path = "./processing/pretrained/emonet_5.pth"
    print(f'Loading the model from {state_dict_path}.')
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
    net = EmoNet(n_expression=5).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def getValueResultTensor(results, key):
    return results[key].cpu().detach().numpy()[0]

"""
The function below identifies the device which is availabe to make the prediction and uses it to load and infer the frame. Once it has results it will extract the labels and cordinates(Along with scores) for each object detected in the frame.
"""
def score_frame(frame):
    w = frame.shape[0]
    h = frame.shape[1]

    frame = np.ascontiguousarray(frame) # from the original code: Fix for PyTorch currently not supporting negative stric
    transform_image = transforms.Compose([transforms.ToTensor()])
    frame = transform_image(frame)
    frame = frame.reshape(1,3,w,h) # seems like needs to be fixed to 256x256?
    frame = frame.to(device)

    with torch.no_grad():
        results = net(frame)

    categorical = softmax(getValueResultTensor(results, 'expression')) # cpu in case it is in gpu?
    dimensional = (getValueResultTensor(results, 'valence'), getValueResultTensor(results, 'arousal'))
   
    return {'categorical': categorical, 'dimensional': dimensional}