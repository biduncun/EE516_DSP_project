import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import signal_tools


save_folder = "ADC_data/Single_channel_ADC_input_data"
output_folder = "ADC_data/Single_channel_ADC_output_data"

n_bits_ = 14
simulation_bit = 16
sample_rate = 1_000_000
simulation_bit = 16 

non_ideal_output_in_bit = np.load(f"{save_folder}/inputs.npy")
ideal_output_in_double = np.load(f"{save_folder}/targets.npy")
double_conversion = np.load(f"{save_folder}/before_calibration.npy")



inputs = torch.tensor(non_ideal_output_in_bit)
targets = torch.tensor(ideal_output_in_double)

print(inputs)
print(targets)

# this is to create mini batch just in case the pc is too weak
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
batch_size = 2^(simulation_bit)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



model = nn.Linear(n_bits_, 1) #n bit in 1 double out
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 20


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

NN_output = outputs.detach().numpy().squeeze()


np.save(f"{output_folder}/NN_output.npy",NN_output)

def quantize_to_14bit(arr, signed=False):
    arr = np.clip(arr, 0, 1.0)  # Assuming input is in range [-1,1]
    
    if signed:
        scale = 8191  # 14-bit signed range: [-8192, 8191]
        quantized_arr = np.round(arr * scale).astype(np.int16)
    else:
        scale = 16383  # 14-bit unsigned range: [0, 16383]
        quantized_arr = np.round((arr + 1) / 2 * scale).astype(np.uint16)
    
    return quantized_arr

NN_output_quantized = quantize_to_14bit(NN_output,False)
np.save(f"{output_folder}/NN_output_Quantized.npy",NN_output_quantized)


# plotting for the non ideal output
signal_fft = signal_tools.spectrum._compute_spectrum(double_conversion,2**simulation_bit)
#breakpoint()

    
# Sample data
x = np.abs(np.fft.fftfreq(2**simulation_bit)[:signal_fft.shape[0]]) #* sample_rate
y = 20*np.log10(signal_fft)

##np.save("non_ideal_output_fft.npy",y)


# Create a line plot
plt.plot(x, y, label="Line")  # Label for legend
plt.ylabel("Y-axis")          # Label for y-axis
#plt.title("Simple Line Graph") # Title of the graph
plt.legend()                   # Show legend
plt.grid(True)                 # Add grid
plt.savefig("Image/ADC_CALIBRATION/non_ideal.png", dpi=300, bbox_inches="tight")  # Save with high resolution
plt.close()  # Close the plot to free memory


# plotting
signal_fft = signal_tools.spectrum._compute_spectrum(NN_output_quantized,2**simulation_bit)
#breakpoint()


# Sample data
x = np.abs(np.fft.fftfreq(2**simulation_bit)[:signal_fft.shape[0]]) #* sample_rate
y = 20*np.log10(signal_fft)

##np.save("Quantized_NN_output_FFT.npy",y)

# Create a line plot
plt.plot(x, y, label="Line")  # Label for legend
plt.ylabel("Y-axis")          # Label for y-axis
#plt.title("Simple Line Graph") # Title of the graph
plt.legend()                   # Show legend
plt.grid(True)                 # Add grid
plt.savefig("Image/ADC_CALIBRATION/NN_SAR_output.png", dpi=300, bbox_inches="tight")  # Save with high resolution
plt.close()  # Close the plot to free memory


print("non_ideal")
#effective number of bit
print(f"enob: [{signal_tools.spectrum.compute_enob(double_conversion,nfft = 2**simulation_bit)}]")
#Spurious-Free Dynamic Range Signal vs Harmonic
print(f"sfdr: [{signal_tools.spectrum.compute_sfdr(double_conversion,nfft = 2**simulation_bit)}]")
#Signal-to-Noise-and-Distortion Ratio: power of signal / (noise + distortion)
print(f"sndr: [{signal_tools.spectrum.compute_sndr(double_conversion,nfft = 2**simulation_bit)}]")
#Signal-to-Noise Ratio: power of signal / (noise)
print(f"sndr: [{signal_tools.spectrum.compute_snr(double_conversion,nfft = 2**simulation_bit)}]")

print("NN output")
print(f"enob: [{signal_tools.spectrum.compute_enob(NN_output,nfft = 2**simulation_bit)}]")
print(f"sfdr: [{signal_tools.spectrum.compute_sfdr(NN_output,nfft = 2**simulation_bit)}]")
print(f"sndr: [{signal_tools.spectrum.compute_sndr(NN_output,nfft = 2**simulation_bit)}]")
print(f"sndr: [{signal_tools.spectrum.compute_snr(NN_output,nfft = 2**simulation_bit)}]")
