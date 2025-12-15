import numpy as np
import adc_models
import signal_tools

# This code is to generate both the output of the non ideal sar_adc as well as the truth label for the output

save_folder = "ADC_data/Single_channel_ADC_input_data"
v_ref_ = 1 #volt
n_bits_ = 14

# the non ideal sar that we trying to calibrate
sar_non_ideal = adc_models.sar.BinarySingleEnded(n_bits  = n_bits_, mode = "nonideal", v_ref = v_ref_)


simulation_bit = 16 # 2^simulation_bit number of point use for training

sample_rate = 1_000_000 #sample rate of the SAR_ADC
input_frequency = 51_111 #input frequency of the signal
T_input_frequency = 1/input_frequency
T_sample = 1/sample_rate 

#input signal to the SAR_ADC
signal_gen = signal_tools.signal_generator.sine_wave(n_points = 2**simulation_bit, sample_rate = sample_rate,fin = 51_111, vpp = v_ref_, offset = v_ref_/2)


non_ideal_output_in_bit = np.zeros((2**simulation_bit, n_bits_ ), dtype=np.float32)
ideal_output_in_double  = np.zeros(2**simulation_bit, dtype=np.float32)

#voltage for each level of ADC
adc_step = v_ref_ / (2**n_bits_)

for i in range(2** simulation_bit):
    #this input will be feed into 14 bit neural network
    non_ideal_output_in_bit[i] = sar_non_ideal.digitize(signal_gen[i])

    # truth label
    ideal_output_in_double[i] = np.floor(signal_gen[i] / adc_step)*adc_step



# non linear ouput to double for comparison at the end of the simulation:

binary_array = np.round(non_ideal_output_in_bit).astype(int)
integers = [int(''.join(map(str, row)), 2) for row in binary_array]
#print(f"Binary array:\n{binary_array}")

#print(f"Integer values: {integers}")
integers_array = np.array(integers)
double_conversion = integers_array * adc_step

#input to the neural network
np.save(f"{save_folder}/inputs.npy", non_ideal_output_in_bit)
np.save(f"{save_folder}/targets.npy", ideal_output_in_double)

#before input for compar
np.save(f"{save_folder}/before_calibration.npy", double_conversion)
