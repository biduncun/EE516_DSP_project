import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import signal_tools


save_folder = "ADC_data/TI_ADC_input_data_v1.1"
output_folder = "ADC_data/TI_ADC_output_data_v1.1"

non_ideal_output_in_bit = np.load(f"{save_folder}/inputs.npy")
ideal_output_in_double = np.load(f"{save_folder}/targets.npy")
double_conversion = np.load(f"{save_folder}/before_calibration.npy")

inputs = torch.tensor(non_ideal_output_in_bit)
targets = torch.tensor(ideal_output_in_double)

print(inputs)
print(targets)

sample_rate = 1_000_000 


n_bits_ = 14


# this is to create mini batch
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

dataset = CustomDataset(inputs, targets)

# Create the DataLoader with mini-batch size (e.g., batch size = 4)
batch_size = ideal_output_in_double.size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



#model = nn.Linear(n_bits_*4+4, 1)

model = nn.Sequential(
    nn.Linear(n_bits_ * 4,4),  # First layer: Fully connected with 4 output 
)

#initialize edge weight to reduce training time
model[0].bias.data.zero_()
model[0].weight.data.zero_()

initial_weight=np.zeros((4,56))

for i in range(n_bits_):
    temp = 1/(2**(1+i))
    initial_weight[0,i] = temp
    initial_weight[1,i+n_bits_] = temp
    initial_weight[2,i+n_bits_*2] = temp
    initial_weight[3,i+n_bits_*3] = temp

model[0].weight.data = torch.tensor(initial_weight, dtype=torch.float32)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 500


outputs = model(inputs)

#breakpoint()

for epoch in range(epochs):
    if epoch == 2999:
        cumulative_loss = 0.0
    for inputs_batch, targets_batch in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs_batch)  # [batch_size, 1]
        # Compute the loss
        loss = criterion(outputs.squeeze(), targets_batch)
        if epoch == 2999:
            cumulative_loss += loss.item()
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
       

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.10f}")


outputs = model(inputs)

NN_output = outputs.detach().numpy().squeeze().flatten()


signal_fft = signal_tools.spectrum._compute_spectrum(double_conversion,2**16)

# Sample data
x = np.abs(np.fft.fftfreq(2**16)[:signal_fft.shape[0]]) #* sample_rate*4
y = 20*np.log10(signal_fft)

# Create a line plot
plt.plot(x, y)  # Label for legend
plt.ylabel("Magnitude(db)")          # Label for y-axis
plt.xlabel("Normalized Frequency")
#plt.title("Simple Line Graph") # Title of the graph
plt.legend()                   # Show legend
plt.grid(True)                 # Add grid
plt.show()                     # Display the graph
plt.savefig("Image/TI_ADC_CALIBRATION__v1.1/non ideal output TI-SAR-ADC fft .png", dpi=300, bbox_inches="tight")  # Save with high resolution
plt.close()  # Close the plot to free memory



signal_fft = signal_tools.spectrum._compute_spectrum(NN_output,2**16)

# Sample data
x = np.abs(np.fft.fftfreq(2**16)[:signal_fft.shape[0]]) #* sample_rate*4
y = 20*np.log10(signal_fft)

# Create a line plot
plt.plot(x, y)  # Label for legend
plt.ylabel("Magnitude(db)")          # Label for y-axis
plt.xlabel("Normalized Frequency")

#plt.title("Simple Line Graph") # Title of the graph
plt.legend()                   # Show legend
plt.grid(True)                 # Add grid
plt.show()                     # Display the graph
plt.savefig("Image/TI_ADC_CALIBRATION__v1.1/NN output TI-SAR-ADC fft .png", dpi=300, bbox_inches="tight")  # Save with high resolution
plt.close()  # Close the plot to free memory


simulation_bit = 16

print("non_ideal")
print(f"enob: [{signal_tools.spectrum.compute_enob(double_conversion,nfft = 2**simulation_bit)}]")
print(f"sfdr: [{signal_tools.spectrum.compute_sfdr(double_conversion,nfft = 2**simulation_bit)}]")
print(f"sndr: [{signal_tools.spectrum.compute_sndr(double_conversion,nfft = 2**simulation_bit)}]")
print(f"sndr: [{signal_tools.spectrum.compute_snr(double_conversion,nfft = 2**simulation_bit)}]")

print("NN output")
print(f"enob: [{signal_tools.spectrum.compute_enob(NN_output,nfft = 2**simulation_bit)}]")
print(f"sfdr: [{signal_tools.spectrum.compute_sfdr(NN_output,nfft = 2**simulation_bit)}]")
print(f"sndr: [{signal_tools.spectrum.compute_sndr(NN_output,nfft = 2**simulation_bit)}]")
print(f"sndr: [{signal_tools.spectrum.compute_snr(NN_output,nfft = 2**simulation_bit)}]")

breakpoint()
