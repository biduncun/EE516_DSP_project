import signal_tools
import matplotlib.pyplot as plt
import numpy as np
import adc_models


save_folder = "ADC_data/TI_ADC_input_data_v1.1"

v_ref_ = 1

sample_rate_low = 250_000 # sample rate of SAR_ADC
simulation_bit = 22
n_bits_ = 14

sar_non_ideal_1 = adc_models.sar.BinarySingleEnded(n_bits = n_bits_, mode = "nonideal", v_ref = v_ref_)
sar_non_ideal_2 = adc_models.sar.BinarySingleEnded(n_bits = n_bits_, mode = "nonideal", v_ref = v_ref_)
sar_non_ideal_3 = adc_models.sar.BinarySingleEnded(n_bits = n_bits_, mode = "nonideal", v_ref = v_ref_)
sar_non_ideal_4 = adc_models.sar.BinarySingleEnded(n_bits = n_bits_, mode = "nonideal", v_ref = v_ref_)

sample_rate = 1_000_000 #for signal gen
input_frequency = 51_111
T_input_frequency = 1/input_frequency
T_sample = 1/sample_rate 



delta_1 = 1  #skew timer 1 in degree
delta_2 = -2 #sket timer 2
delta_3 = 2  #sket timer 3

phase_shift_1 = T_sample/T_input_frequency*360   + delta_1
phase_shift_2 = 2*T_sample/T_input_frequency*360 + delta_2
phase_shift_3 = 3*T_sample/T_input_frequency*360 + delta_3



input_signals_1 = signal_tools.signal_generator.sine_wave(n_points = 2**simulation_bit, sample_rate = sample_rate,fin = input_frequency, vpp = v_ref_, offset = v_ref_/2)
input_signals_2 = signal_tools.signal_generator.sine_wave(n_points = 2**simulation_bit, sample_rate = sample_rate,fin = input_frequency, vpp = v_ref_, offset = v_ref_/2,phase=phase_shift_1)
input_signals_3 = signal_tools.signal_generator.sine_wave(n_points = 2**simulation_bit, sample_rate = sample_rate,fin = input_frequency, vpp = v_ref_, offset = v_ref_/2,phase=phase_shift_2)
input_signals_4 = signal_tools.signal_generator.sine_wave(n_points = 2**simulation_bit, sample_rate = sample_rate,fin = input_frequency, vpp = v_ref_, offset = v_ref_/2,phase=phase_shift_3)


phase_shift_ideal_1 = T_sample/T_input_frequency*360
#phase_shift_ideal_2 = 2*T_sample/T_input_frequency*360
#phase_shift_ideal_3 = 3*T_sample/T_input_frequency*360

input_signals_ideal_1 = signal_tools.signal_generator.sine_wave(n_points = 2**simulation_bit, sample_rate = sample_rate,fin = input_frequency, vpp = v_ref_, offset = v_ref_/2)
#input_signals_ideal_2 = signal_tools.signal_generator.sine_wave(n_points = 2**simulation_bit, sample_rate = sample_rate,fin = input_frequency, vpp = v_ref_, offset = v_ref_/2,phase=phase_shift_ideal_1)
#input_signals_ideal_3 = signal_tools.signal_generator.sine_wave(n_points = 2**simulation_bit, sample_rate = sample_rate,fin = input_frequency, vpp = v_ref_, offset = v_ref_/2,phase=phase_shift_ideal_2)
#input_signals_ideal_4 = signal_tools.signal_generator.sine_wave(n_points = 2**simulation_bit, sample_rate = sample_rate,fin = input_frequency, vpp = v_ref_, offset = v_ref_/2,phase=phase_shift_ideal_3)

#only store 2^16/4 sample. or 14*4 input 4 output
non_ideal_output_in_bit = np.zeros((2**simulation_bit//64//4, n_bits_*4), dtype=np.float32)
ideal_output_in_double  = np.zeros((2**simulation_bit//64//4,4), dtype=np.float32)

adc_step = v_ref_ / (2**n_bits_)



temp = np.zeros(n_bits_*4, dtype=np.float32)
non_ideal_output_bit_seperate = np.zeros((2**simulation_bit//64*4, n_bits_), dtype=np.float32)

for i in range(2** simulation_bit//64//4):
    temp[0:n_bits_] = sar_non_ideal_1.digitize(input_signals_1[i*4])
    temp[n_bits_:2*n_bits_] = sar_non_ideal_2.digitize(input_signals_2[i*4])
    temp[2*n_bits_:3*n_bits_] = sar_non_ideal_3.digitize(input_signals_3[i*4])
    temp[3*n_bits_:4*n_bits_] = sar_non_ideal_4.digitize(input_signals_4[i*4])

    non_ideal_output_bit_seperate[i*4] = temp[0:n_bits_]
    non_ideal_output_bit_seperate[i*4+1] = temp[n_bits_:2*n_bits_]
    non_ideal_output_bit_seperate[i*4+2] = temp[2*n_bits_:3*n_bits_]
    non_ideal_output_bit_seperate[i*4+3] = temp[3*n_bits_:4*n_bits_]
    
    ideal_output_in_double[i,0] = np.floor(input_signals_ideal_1[i*4] / adc_step)*adc_step
    ideal_output_in_double[i,1] = np.floor(input_signals_ideal_1[i*4+1] / adc_step)*adc_step
    ideal_output_in_double[i,2] = np.floor(input_signals_ideal_1[i*4+2] / adc_step)*adc_step
    ideal_output_in_double[i,3] = np.floor(input_signals_ideal_1[i*4+3] / adc_step)*adc_step

    non_ideal_output_in_bit[i] = temp

trimmed_array = non_ideal_output_bit_seperate
binary_array = np.round(trimmed_array[:][0:]).astype(int)
# Convert each row of the binary array to an integer
integers = [int(''.join(map(str, row)), 2) for row in binary_array]
#print(f"Binary array:\n{binary_array}")

integers_array = np.array(integers)
double_conversion = integers_array * adc_step
print(f"Output of nonideal values: {double_conversion}")

x = double_conversion[0:500]
plt.plot(x, label='x values')
plt.title("Visualization of x Array")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid()


plt.savefig("Image/TI_ADC_CALIBRATION__v1.1/nonideal_output_signal.png", dpi=300, bbox_inches="tight")
plt.close()  # Close the plot to free memory

np.save(f"{save_folder}/inputs.npy", non_ideal_output_in_bit)
np.save(f"{save_folder}/targets.npy", ideal_output_in_double)
np.save(f"{save_folder}/before_calibration.npy", double_conversion)

#breakpoint()