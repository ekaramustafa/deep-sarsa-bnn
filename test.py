import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from layers.vbl_base import VBLinear  
class FunctionApproxTester:
 
    def __init__(self, seed=42):
        self.seed = seed
    
    def test(self, module, true_function, num_epochs=400, num_points=100):
        print("=======================")
        print(f"Model : {module.layer_name}")
        print("Training&Testing Starts")
        print("========================")
        torch.manual_seed(self.seed)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(module.parameters())

        x_train = torch.linspace(0, 2 * np.pi, num_points).reshape(-1, 1)
        y_train = true_function(x_train) + torch.randn_like(x_train) * 0.1  # Add noise to simulate real-world data

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = module(x_train)
            loss = criterion(output, y_train)

            kl_loss = 0
            for layer in module.children():
                if isinstance(layer, VBLinear):
                    kl, n = layer.kl_loss()
                    kl_loss += kl / n

            loss += kl_loss
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Generate predictions for the test data
        x_test = torch.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
        with torch.no_grad():
            y_pred = module(x_test)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.scatter(x_train, y_train, color='blue', label='Training Data')
        plt.plot(x_test, true_function(x_test), color='green', linestyle='--', label='True Function')
        plt.plot(x_test, y_pred, color='red', label='Model Prediction')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Bayesian Neural Network Approximation')
        plt.legend()
        plt.grid(True)
        plt.show()
        print("=======================")
        print(f"Model : {module.layer_name}")
        print("Training&Testing Ends")
        print("========================\n")
