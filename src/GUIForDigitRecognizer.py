# install pre-reqs
# brew install python-tk@3.9
# To run 'python3 ./gui_digit_recognizer.py'

from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

# for the keras NN implementation
model = load_model('mnist.h5')

# for the deafult impl 
#myModel = nn.Sequential(nn.Linear(784, 128),
#                      nn.ReLU(),
#                      nn.Linear(128, 64),
#                      nn.ReLU(),
#                      nn.Linear(64, 10),
#                      nn.LogSoftmax(dim=1))
#myModel = torch.load('torch.h5')

# for the pytorch NN implementation 
import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x

# initialize the NN
myModel = Net()
myModel.load_state_dict(torch.load('torch1.h5'))
myModel.eval()

def predict_digit_keras(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    print(f"Keras predicted result: {res}")
    return np.argmax(res), max(res)

def predict_digit_torch(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #predicting the class
    imgTensor = transforms.ToTensor()(img)
    imgTensor = imgTensor.view(1, 784)
    with torch.no_grad():
        logps = myModel(imgTensor)
    res = torch.exp(logps)
    print(f"PyTorch predicted result: {res}")
    return np.argmax(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0
        
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "black", cursor="cross")
        self.label = tk.Label(self, text="Draw a digit...", font=("Arial", 48))
        self.classify_btn = tk.Button(self, text = "Predict the digit", command = self.classify_handwriting, background='#345', foreground='black')   
        self.button_clear = tk.Button(self, text = "Clear Screen", command = self.clear_all, background='#345', foreground='black')
       
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        
        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        
        x0 = self.canvas.winfo_rootx()
        y0 = self.canvas.winfo_rooty()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        # print(f"x0: {x0}, y0: {y0}")
        # Rect(left, top, right, bottom)
        rect = (x0+60, y0+60, x0+width+300, y0+height+300)
        
        im = ImageGrab.grab(rect)
        #convert rgb to grayscale
        im = im.convert('L')
        im.save('digit_drawing.png')
        
        # implement prediction using Keras
        digit, acc = predict_digit_keras(im)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%' + ' confidence')
        
        # implement prediction using pytorch
        # digit = predict_digit_torch(im)
        # self.label.configure(text= str(digit))
        
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=16
        self.canvas.create_rectangle(self.x-r, self.y-r, self.x + r, self.y + r, fill='white')
       
app = App()
mainloop()
