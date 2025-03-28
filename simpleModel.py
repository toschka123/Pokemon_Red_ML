import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_gen1 import GameWrapperPokemonGen1
import random 
import math

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(18*20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Linear(512, 7),
            nn.Linear(512, 6), # remove start for the time being
            nn.Softmax(dim=1)  # Adding Softmax activation function
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    """def losfunction(self, hist,current):
        loss = torch.tensor(0.0, requires_grad=True)

        if len(hist)>0:
            for i in hist:
                dist = np.linalg.norm(i-current)
                if loss < dist:
                    loss = dist
        loss = - loss"""
    def losfunction(self, hist, current):
        # Compute the loss
        loss = math.inf
        #Find minimum distance
        for i in hist:
            dist = np.linalg.norm(i-current)
            if dist < loss:
                    loss = dist
        if loss > 2*math.e**24:
            reward = 1
            print("TASty!")
        elif 2*math.e**24 > loss > 2*math.e**23:
            reward = 0.01
        else:
            reward = 0
        return reward

    
  
        # Store the current tensor (detach to prevent computation graph issues)
        

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.1, 0.1)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -0.1, 0.1)


def predict(save):
    save2 = save.reshape(360)
    save2_reshaped = save2.reshape(1, -1)

    # Convert the numpy array to a torch tensor
    input_tensor = torch.tensor(save2_reshaped, dtype=torch.float32)

    # Run the model on the input tensor
    output = model(input_tensor)

    # Print the output
    return(output)

"""def losfunction(hist, current):
    loss = 0
    for i in hist:
        dist = np.linalg.norm(i-current)
        if loss < dist:
            loss = dist
    hist.append(current)
    return -loss"""

alpha = 0.05
model = NeuralNetwork()
model.apply(initialize_weights)
hist = []
pyboy = PyBoy("Pokemon Red.gb")  # Replace with your ROM filename

with open("state_file.state", "rb") as f:
    pyboy.load_state(f)

    
optimizer = optim.SGD(model.parameters(), lr=0.0001) # try adam
optimizer.zero_grad()
frames = 0
while True:
    frames+=1
    save = pyboy.game_area()
    actions = predict(save)
    if random.random() > alpha:
        action_index = torch.argmax(actions)
    else:
        #print("random")
        action_index = random.randint(0, 6)
    if action_index == 0:
        pyboy.button('a')
    elif action_index == 1:
        pyboy.button('b')
    elif action_index == 2:
        pyboy.button('up')
    elif action_index == 3:
        pyboy.button('down')
    elif action_index == 4:
        pyboy.button('left')
    elif action_index == 5:
        pyboy.button('right')
    # elif action_index == 6:
        # pyboy.button('start')
    pyboy.tick(1)
    current =pyboy.game_area()
    #loss = model.losfunction(hist, current)
    #hist.append(current)
    loss= model.losfunction(hist, current)
    hist.append(current)
    loss_tensor = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
    optimizer.step()
    print(loss_tensor)
    loss_tensor.backward()
    
    """if ticks%200 == 0:
        loss= losfunction(hist, current)
        hist.append(current)
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
        #optimizer.zero_grad()
        loss_tensor = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
        #print(loss_tensor, action_index)
        loss_tensor.backward()
        optimizer.step()
        print('abc',hist)"""

  