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

model = load_model('mnist.h5')

myModel = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
myModel = torch.load('torch.h5')
myModel.eval()

def predict_digit_keras(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #img.save('test2.png')
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
    #img.save('test2.png')
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
        self.label = tk.Label(self, text="Don't Draw..", font=("Arial", 48))
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
        #digit = predict_digit_torch(im)
        #self.label.configure(text= str(digit))
        
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=16
        self.canvas.create_rectangle(self.x-r, self.y-r, self.x + r, self.y + r, fill='white')
       
app = App()
mainloop()
