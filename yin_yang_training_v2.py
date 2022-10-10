import numpy as np
import matplotlib.pyplot as plt
import math
from dataset import YinYangDataset
from torch.utils.data import DataLoader




def gating_variables_hh(V, m, n, h, dt):

    gNa = 0.12
    gK = 0.036
    gL = 0.0003
    El = -54.3
    ENa = 50
    Ek = -77

    # for compartment in range(0, len(V)):
    alpha_m = (0.1 * (V + 40)) / (1 - np.exp(-0.1 * (V + 40)))
    beta_m = 4 * np.exp(-0.0556 * (V + 65))
    tau_m = 1 / (alpha_m + beta_m)
    m_inf = alpha_m / (alpha_m + beta_m)
    m = m_inf + (m - m_inf) * np.exp(-dt / tau_m)

    alpha_h = 0.07 * np.exp(-0.05 * (V + 65))
    beta_h = 1 / (1 + np.exp(-0.1 * (V + 35)))
    tau_h = 1 / (alpha_h + beta_h)
    h_inf = alpha_h / (alpha_h + beta_h)
    h = h_inf + (h - h_inf) * np.exp(-dt / tau_h)

    alpha_n = (0.01 * (V + 55)) / (1 - np.exp(-0.1 * (V + 55)))
    beta_n = 0.125 * np.exp(-0.0125 * (V + 65))
    tau_n = 1 / (alpha_n + beta_n)
    n_inf = alpha_n / (alpha_n + beta_n)
    n = n_inf + (n - n_inf) * np.exp(-dt / tau_n)

    totGi = gL + gK*(n**4) + gNa*(m**3)*h
    totGiE = gL*El + gK*(n**4)*Ek + gNa*(m**3)*h*ENa

    return totGi*1000, totGiE*1000, m, n, h


def dCaAP(v, t_dCaAP, dCaAP_count, A, B, t, dt, K, w):
    # define parameters
    # print(t)
    # w = 0
    v_th = -36
    refract_period = 30
    tauA = 3
    tauB = 0.4
    D = 0.3
    sigma_diff = 21
    v_rest = -75
    denom = -1 / ((v_th - v_rest)*D)
    # t_dCaAP = -refract_period
    # w = w# change to 3
    # print(B)
    x = dt * (1/tauA)
    y = 2 * x

    x2 = dt * (1/tauB)
    y2 = 2 * x2

    A = A * np.exp(x - y*A)
    B = B * (np.exp(x2 - y2*B))

    if B == 0 and t > t_dCaAP + sigma_diff and w > 0:
        B = 0.001

    if t > t_dCaAP + refract_period and v > v_th and w > 0:
        t_dCaAP = t
        A = 0.001
        B = 0
        K = np.exp((v - v_th) * denom)
        if K > 1:
            K = 1
        dCaAP_count += 1

    i = -(A - B) * w * K

    return A, B, i, dCaAP_count, t_dCaAP, K, w


def vtrap(x,y):
    # Traps for 0 in denominator of rate eqns.
    if abs(x / y) < 1e-6:
        vtrap = y*(1 - x/y/2)
    else:
        vtrap = x/(np.exp(x/y) - 1)
    return vtrap


def gating_variables_passive(V):

    gL_bar = 0.001
    eL = -70
    i = gL_bar*(V - eL)

    totG = gL_bar
    totGE = gL_bar*eL

    return totG*1000, totGE*1000, i


def BE_voltage_update_twoComp(V_dend, V_soma, dt, totG_dend, totGE_dend, totG_soma, totGE_soma, g, Ie):
    cm = 1
    # update somatic
    B_soma = 0
    C_soma = -(1 / cm) * (totG_soma + g[0, 1])
    D_soma = (1/cm) * g[0, 1]
    F_soma = (1/cm) * (totGE_soma + Ie[1])

    b_soma = 0
    c_soma = C_soma * dt
    d_soma = D_soma * dt
    f_soma = (F_soma + C_soma*V_soma + D_soma*V_dend) * dt

    # update dendrite
    B_dend = (1/cm) * g[1, 0]
    C_dend = -(1 / cm) * (totG_dend + g[1, 0])
    D_dend = 0
    F_dend = (1/cm) * (totGE_dend + Ie[0])

    b_dend = B_dend * dt
    c_dend = C_dend * dt
    d_dend = 0
    f_dend = (F_dend + B_dend*V_soma + C_dend*V_dend) * dt

    c_x_soma = c_soma
    f_x_soma = f_soma
    c_x_dend = c_dend + (b_dend*d_soma)/(1-c_x_soma)
    f_x_dend = f_dend + (b_dend*f_soma)/(1-c_x_soma)

    dV_dend = f_x_dend / (1 - c_x_dend)
    dV_soma = (d_soma*dV_dend + f_x_soma) / (1 - c_x_soma)

    V_dend += dV_dend
    V_soma += dV_soma

    return V_dend, V_soma


def update_synapse(ps_vals, timestep, spike_times, V, syn_weights):


    # num_synapses = 35
    exp_decay = np.exp(-0.1/12)
    # test_exp = np.exp(-0.1/1.8) - np.exp(-0.1/0.3)
    num_inputs = 2
    # for input in range(2):
    #     these_weights = syn_weights[input]
    current = np.zeros([num_inputs, np.size(V)])

    for input in range(num_inputs):
        these_weights = syn_weights[input]

        for synapse in range(np.size(ps_vals,1)):
            these_spike_times = spike_times[input][synapse][:]

            ps_vals[input, synapse] *= exp_decay
            # these_spike_times = spike_times[synapse]

            if timestep in these_spike_times:
                ps_vals[input, synapse] += (1 - ps_vals[input, synapse])

            # if synapse < 25:
            #     these_weights = syn_weights[0]
            # else:
            #     these_weights = syn_weights[1]

            for output in range(np.size(V)):

                # current[synapse, output] = (these_weights[output] / 1000) * ps_vals[input, synapse] * (0 - V[output])
                current[input, output] += (these_weights[output] / 1000) * ps_vals[input, synapse] * (0 - V[output])

            # current[synapse] = (syn_weights[1]/1000) * ps_vals[synapse] * (0 - V)

    total_current = np.sum(current, 0)

    return ps_vals, total_current


def BREAKPOINT(m,h,n,v):
    gnabar = 0.12
    gkbar = 0.036
    gl = 0.0003
    el = -54.3
    ena = 50
    ek = -77

    totG = gl + gkbar * (n ** 4) + gnabar * (m ** 3) * h
    totGE = gl * el + gkbar * (n ** 4) * ek + gnabar * (m ** 3) * h * ena


    return totG*1000, totGE*1000


def lookup_table_hh():
    v_values = np.arange(-100, 100, 1)
    table = np.empty([7, len(v_values)])
    table[0, :] = v_values
    count = 0

    for vt in v_values:
        alpha_m = .1 * vtrap(-(vt + 40), 10)
        beta_m = 4 * np.exp(-(vt + 65) / 18)
        tau_m = 1 / (alpha_m + beta_m)
        m_inf = alpha_m / (alpha_m + beta_m)

        alpha_h = 0.07 * np.exp(-(vt+65)/20)
        beta_h = 1 / (np.exp(-(vt+35)/10) + 1)
        tau_h = 1 / (alpha_h + beta_h)
        h_inf = alpha_h / (alpha_h + beta_h)

        alpha_n = .01*vtrap(-(vt+55), 10)
        beta_n = .125*np.exp(-(vt+65)/80)
        tau_n = 1 / (alpha_n + beta_n)
        n_inf = alpha_n / (alpha_n + beta_n)

        table[1, count] = tau_m
        table[2, count] = m_inf
        table[3, count] = tau_h
        table[4, count] = h_inf
        table[5, count] = tau_n
        table[6, count] = n_inf

        count = count + 1

    return table


def hh_lookup(v, m, n, h, table):
    gnabar = 0.12
    gkbar = 0.036
    gl = 0.0003
    el = -54.3
    ena = 50
    ek = -77


    rem = (v - -100) - int(v - -100)
    indices = [int(v - -100), int(v - -100)+1]
    tau_m = np.diff(table[1, indices]) * rem + table[1, indices[0]]
    m_inf = np.diff(table[2, indices]) * rem + table[2, indices[0]]
    tau_h = np.diff(table[3, indices]) * rem + table[3, indices[0]]
    h_inf = np.diff(table[4, indices]) * rem + table[4, indices[0]]
    tau_n = np.diff(table[5, indices]) * rem + table[5, indices[0]]
    n_inf = np.diff(table[6, indices]) * rem + table[6, indices[0]]

    m = m_inf[0] - (-m + m_inf[0]) * np.exp(-0.1 / tau_m[0])
    n = n_inf[0] - (-n + n_inf[0]) * np.exp(-0.1 / tau_n[0])
    h = h_inf[0] - (-h + h_inf[0]) * np.exp(-0.1 / tau_h[0])

    totG = gl + gkbar * (n ** 4) + gnabar * (m ** 3) * h
    totGE = gl * el + gkbar * (n ** 4) * ek + gnabar * (m ** 3) * h * ena

    return totG*1000, totGE*1000, m, n, h


def update_synapse_output(ps_val, spike , V, syn_weights):

    exp_decay = np.exp(-0.1/12)
    # test_exp = np.exp(-0.1/1.8) - np.exp(-0.1/0.3)
    current = np.empty(np.size(syn_weights))

    ps_val *= exp_decay

    if spike:
        ps_val += (1 - ps_val)

    for synapse in range(np.size(syn_weights)):
        current[synapse] = (syn_weights[synapse]/1000) * ps_val * (0 - V[synapse])

    return ps_val, current


def LIF_neuron_update(voltage_array, dt, current, timestep, refract):
    E = -65
    Rm = 10
    tau_exp = 0.990049833749168
    tau_m = 10
    # t_rest = 0  # this is calculated from exp(-dt/tau_m) with dt = 0.1 and tau_m = 10
    # dt = 0.025
    # V = -65
    v_reset = -65
    v_thresh = -60
    # current = 1
    refractory_period = 100
    spikes = [0,0,0]

    # print(E_l)
    # voltage = []

    # for timestep in range(0, 1000):
    #     voltage.append(V)

    count = 0
    for voltage in voltage_array:

        E_l = E + Rm * current[count]*100

        if timestep > refract[count]:
            voltage = E_l + (voltage - E_l) * np.exp(-dt / tau_m)

        if voltage > v_thresh:
            voltage = v_reset
            # print(voltage)
            refract[count] = timestep + refractory_period
            spikes[count] = 1

        voltage_array[count] = voltage

        count += 1

    return voltage_array, refract, spikes


def run_suimulation(freq1, freq2, syn_weights):

    num_neurons_hidden_layer = 5
    num_inputs = 2
    num_outputs = 3
    num_synapses = 50
    total_time = 1
    freq1 = 5
    freq2 = 4
    spike_time_array_1 = []#np.zeros([num_synapses, firing_frequency])
    spike_time_array_2 = []
    spike_time_array = []

    output_1_spike_count = 0
    output_2_spike_count = 0
    output_3_spike_count = 0
    # freq1 = 2
    # freq2 = 0

    for synapse in range(num_synapses):
        randomness1 = np.random.random(total_time * 10000 + 1) < (freq1 / 10000)
        randomness2 = np.random.random(total_time * 10000 + 1) < (freq2 / 10000)

        # spike_times = np.nonzero(randomness1)[0].tolist()
        # spike_times += np.nonzero(randomness2)[0].tolist()

        spike_time_array_1.append(np.nonzero(randomness1)[0].tolist())
        spike_time_array_2.append(np.nonzero(randomness2)[0].tolist())

        # spike_time_array_1.append(spike_times)
        # spike_times.append(1)

        # spike_time_array_2.append(spike_times)




    spike_time_array.append(spike_time_array_1)
    spike_time_array.append(spike_time_array_2)
    # spike_time_array = spike_time_array_1
    # synapse_ps = np.zeros([num_synapses])

    synapse_ps = np.zeros([num_inputs, num_synapses])

    output_ps = np.zeros(num_neurons_hidden_layer)

    num_compartments = 2
    g = np.empty((num_compartments, 2))
    g[0, 0] = 0
    g[0, 1] = 0.1
    g[1, 0] = 0
    g[1, 1] = 0

    dt = 0.1

    time = total_time * 1000
    t = time / dt

    soma_spike_count = np.zeros(num_neurons_hidden_layer)

    output_voltages = np.zeros(num_outputs)-65
    # start dendrite off
    V_dend = np.zeros(num_neurons_hidden_layer)-65
    dCaAP_count = np.zeros(num_neurons_hidden_layer)
    A_dCaAP = np.zeros(num_neurons_hidden_layer)
    B_dCaAP = np.zeros(num_neurons_hidden_layer)
    t_dCaAP = np.zeros(num_neurons_hidden_layer)-200
    K = np.zeros(num_neurons_hidden_layer)
    i = np.zeros(num_neurons_hidden_layer)
    w = 3
    for dend in range(num_neurons_hidden_layer):
        A_dCaAP[dend], B_dCaAP[dend], i[dend], dCaAP_count[dend], t_dCaAP[dend], K[dend], w = dCaAP(V_dend[dend], t_dCaAP[dend], dCaAP_count[dend], A_dCaAP[dend], B_dCaAP[dend], -dt / 2,
                                                            dt, K[dend], w)

    # start soma off
    V_soma = np.zeros(num_neurons_hidden_layer)-65
    m = np.zeros(num_neurons_hidden_layer)+0.0529
    n = np.zeros(num_neurons_hidden_layer)+0.3177
    h = np.zeros(num_neurons_hidden_layer)+0.5961
    # totG_soma, totGE_soma, m, n, h = gating_variables_hh(V_soma, m, n, h, dt)
    dend_voltage_array = []
    synaptic_current = []
    voltage_array = []
    table = lookup_table_hh()
    totG_soma = np.empty(num_neurons_hidden_layer)
    totGE_soma = np.empty(num_neurons_hidden_layer)

    refract = np.zeros(num_outputs)

    for neuron in range(num_neurons_hidden_layer):
        totG_soma[neuron], totGE_soma[neuron] = BREAKPOINT(m[neuron], h[neuron], n[neuron], V_soma[neuron])
        totG_soma[neuron], totGE_soma[neuron], m[neuron], n[neuron], h[neuron] = hh_lookup(V_soma[neuron], m[neuron], n[neuron], h[neuron], table)
        # totG_soma[neuron], totGE_soma[neuron], m[neuron], n[neuron], h[neuron] = gating_variables_hh(V_soma[neuron],
        #                                                                                              m[neuron],
        #                                                                                              n[neuron],
        #                                                                                              h[neuron], dt)
    totGE_dend = np.zeros(num_neurons_hidden_layer)-70
    totG_dend = np.zeros(num_neurons_hidden_layer)+1

    timesteps = np.linspace(0, total_time * 1000, total_time * 10000)


    for timestep in range(1, int(t)):
        t = round(timesteps[timestep], 5)
        old_V_somas = V_soma

        # update synaptic current
        synapse_ps, i_synapses = update_synapse(synapse_ps, timestep, spike_time_array, V_dend, syn_weights[0])
        # Ie = [0,0]
        i_output = np.zeros([num_neurons_hidden_layer, num_outputs])

        for neuron in range(num_neurons_hidden_layer):
            this_old_v = old_V_somas[neuron]
            spike = 0
            Ie = [0, 0]

            Ie[0] += i_synapses[neuron] * 100
            # Ie[0] += i_synapses[1, neuron] * 100

            if neuron == 0:
                dend_voltage_array.append(V_dend[neuron])
                synaptic_current.append(i[neuron])
                voltage_array.append(V_soma[neuron])


            A_dCaAP[neuron], B_dCaAP[neuron], i[neuron], dCaAP_count[neuron], t_dCaAP[neuron], K[neuron], w = dCaAP(V_dend[neuron], t_dCaAP[neuron], dCaAP_count[neuron], A_dCaAP[neuron], B_dCaAP[neuron],  t - (dt / 2), dt, K[neuron], w)
            Ie[0] -= i[neuron] * 100

            # update somatic current
            totG_soma[neuron], totGE_soma[neuron] = BREAKPOINT(m[neuron], h[neuron], n[neuron], V_soma[neuron])
            totG_soma[neuron], totGE_soma[neuron], m[neuron], n[neuron], h[neuron] = hh_lookup(V_soma[neuron],
                                                                                               m[neuron], n[neuron],
                                                                                               h[neuron], table)
            #
            # totG_soma[neuron], totGE_soma[neuron], m[neuron], n[neuron], h[neuron] = gating_variables_hh(V_soma[neuron], m[neuron], n[neuron], h[neuron], dt)

            # print(V_dend)
            # update voltages

            V_dend[neuron], V_soma[neuron] = BE_voltage_update_twoComp(V_dend[neuron], V_soma[neuron], dt, totG_dend[neuron], totGE_dend[neuron], totG_soma[neuron], totGE_soma[neuron],
                                                       g, Ie)

            # print(Ie)
            if V_dend[neuron] < -300:
                V_dend[neuron] = -300

            if V_soma[neuron] > 20 and this_old_v < 20:
                soma_spike_count[neuron] += 1
                spike = 1

            output_ps[neuron], i_output[neuron, :] = update_synapse_output(output_ps[neuron], spike, output_voltages, syn_weights[1][neuron])

        i_output = np.mean(i_output, 0)

        output_voltages, refract, spikes = LIF_neuron_update(output_voltages, dt, i_output, timestep, refract)

        output_1_spike_count += spikes[0]
        output_2_spike_count += spikes[1]
        output_3_spike_count += spikes[2]


    # soma_firing_frequency = soma_spike_count / total_time
    # average_calcium = np.mean(ca_current)

    output_firing_frequencies = [output_1_spike_count/total_time, output_2_spike_count/total_time, output_3_spike_count/total_time]


    return output_firing_frequencies


# # load dataset/generate dataset
# dataset = np.loadtxt("xor_dataset_shifted_right.txt")
# load dataset
BATCH_SIZE = 1

yy_dataset = YinYangDataset()
dataloader = DataLoader(yy_dataset, batch_size=BATCH_SIZE, shuffle=True)

training_inputs = []
training_labels = []
for train_features, train_labels in dataloader:
    # print(train_features.numpy(), train_labels.numpy())
    training_inputs.append(train_features.numpy())
    training_labels.append(train_labels.numpy())

inputs = 2
hidden_neurons = 5
outputs = 3

# initialise run parameters
lr = 0.1
dw = 0.7
learning_steps = 40
number_of_inputs = 130
max_firing_rate = 60

weights = []
loss_array = []
gradient_array = []
output_array = []
mean_gradient_array = []

weights.append(np.zeros([inputs, hidden_neurons])+2)
weights.append(np.zeros([hidden_neurons, outputs])+2)
loss_array.append(np.zeros([inputs, hidden_neurons, number_of_inputs]))
loss_array.append(np.zeros([hidden_neurons, outputs, number_of_inputs]))
gradient_array.append(np.zeros([inputs, hidden_neurons, number_of_inputs]))
gradient_array.append(np.zeros([hidden_neurons, outputs, number_of_inputs]))
output_array.append(np.zeros([inputs, hidden_neurons, outputs]))
output_array.append(np.zeros([hidden_neurons, outputs, outputs]))
mean_gradient_array.append(np.zeros([inputs, hidden_neurons, outputs]))
mean_gradient_array.append(np.zeros([hidden_neurons, outputs, outputs]))



# initialise empty arrays
# weights = np.empty([learning_steps+1, len(syn_weights)])
trainingFreqs = np.empty([number_of_inputs,2])
testFreqs = np.empty([int(number_of_inputs / 4), 2])

loss = []
# loss = np.empty([len(number_of_weights)+1, number_of_inputs])
# gradients = np.empty([len(syn_weights), number_of_inputs])
total_loss = []

# loss = np.empty([10, number_of_inputs])

# for i in range(number_of_inputs):
#     trainingFreqs[i, :] = np.random.randint(0, 15, 2)
#
# for i in range(int(number_of_inputs / 4)):
#     testFreqs[i, :] = np.random.randint(0, 15, 2)


# weights[0, :] = syn_weights

for trial in range(number_of_inputs):

    freq1 = (training_inputs[trial][0][0]*15)
    freq2 = (training_inputs[trial][0][1]*15)
    correct_output = training_labels[trial][0]

    if correct_output == 0:
        target_freq = [1, 0, 0]
    if correct_output == 1:
        target_freq = [0, 1, 0]
    if correct_output == 2:
        target_freq = [0, 0, 1]

    output = run_suimulation(freq1, freq2, weights)
    # denom = np.exp(output[0]) + np.exp(output[1]) + np.exp(output[2])
    # output = [(output[0] / denom), (output[1] / denom), (output[2] / denom)]
    loss = 0
    output = [(output[0]/max_firing_rate), (output[1]/max_firing_rate), (output[2]/max_firing_rate)]

    for out in range(outputs):
        # loss -= target_freq[out] * np.log(output[out])
        loss += (target_freq[out] - output[out])

    total_loss.append(loss)

    for layer in range(np.size(weights)):
        these_weights = weights[layer]
        for input in range(np.size(these_weights,0)):
            for synapse in range(np.size(these_weights,1)):
                print("Trial = ", trial, "Layer = ", layer, "Input = ", input, "Synapse = ", synapse)
                weights[layer][input][synapse] += dw
                these_outputs = run_suimulation(freq1, freq2, weights)
                normalised_output = [(these_outputs[0]/max_firing_rate), (these_outputs[1]/max_firing_rate), (these_outputs[2]/max_firing_rate)]
                output_array[layer][input][synapse][:] = normalised_output
                this_loss = 0

                for out in range(outputs):
                    # this_loss -= target_freq[out] * np.log(normalised_output[out])
                    this_loss += (target_freq[out] - normalised_output[out])
                    loss_array[layer][input][synapse] = this_loss

                # if math.isnan((this_loss - loss)/dw):
                #     print("broken")

                gradient_array[layer][input][synapse][trial] = (this_loss - loss)/dw

                weights[layer][input][synapse] -= dw



# #
total_loss_array = []
mean_gradient_array[0] = np.mean(gradient_array[0],2)
mean_gradient_array[1] = np.mean(gradient_array[1],2)
total_loss_array.append(np.mean(total_loss))


for ls in range(learning_steps):
    print("Learning step = ", ls, "loss = ", np.mean(total_loss), "learning rate = ", lr, "dw = ", dw)

    for layer in range(np.size(weights)):
        these_weights = weights[layer]
        for input in range(np.size(these_weights,0)):
            for synapse in range(np.size(these_weights,1)):
                weights[layer][input][synapse] -= mean_gradient_array[layer][input][synapse] * lr


    loss_array = []
    gradient_array = []
    output_array = []
    total_loss = []
    loss_array.append(np.zeros([inputs, hidden_neurons, number_of_inputs]))
    loss_array.append(np.zeros([hidden_neurons, outputs, number_of_inputs]))
    gradient_array.append(np.zeros([inputs, hidden_neurons, number_of_inputs]))
    gradient_array.append(np.zeros([hidden_neurons, outputs, number_of_inputs]))
    output_array.append(np.zeros([inputs, hidden_neurons, outputs]))
    output_array.append(np.zeros([hidden_neurons, outputs, outputs]))

    for trial in range(number_of_inputs):

        freq1 = (training_inputs[trial][0][0] * 15)
        freq2 = (training_inputs[trial][0][1] * 15)
        correct_output = training_labels[trial][0]

        if correct_output == 0:
            target_freq = [1, 0, 0]
        if correct_output == 1:
            target_freq = [0, 1, 0]
        if correct_output == 2:
            target_freq = [0, 0, 1]

        output = run_suimulation(freq1, freq2, weights)
        loss = 0
        output = [(output[0] / max_firing_rate), (output[1] / max_firing_rate), (output[2] / max_firing_rate)]

        for out in range(outputs):
            # loss -= target_freq[out] * np.log(output[out])
            loss += (target_freq[out] - output[out])
        total_loss.append(loss)

        for layer in range(np.size(weights)):
            these_weights = weights[layer]
            for input in range(np.size(these_weights, 0)):
                for synapse in range(np.size(these_weights, 1)):
                    print("Trial = ", trial, "Layer = ", layer, "Input = ", input, "Synapse = ", synapse)
                    weights[layer][input][synapse] += dw
                    these_outputs = run_suimulation(freq1, freq2, weights)
                    normalised_output = [(these_outputs[0] / max_firing_rate), (these_outputs[1] / max_firing_rate), (these_outputs[2] / max_firing_rate)]
                    output_array[layer][input][synapse][:] = normalised_output
                    this_loss = 0
                    for out in range(outputs):
                        this_loss += (target_freq[out] - normalised_output[out])
                        loss_array[layer][input][synapse] = this_loss

                    if math.isnan((this_loss - loss) / dw):
                        print("broken")

                    gradient_array[layer][input][synapse][trial] = (this_loss - loss) / dw

                    weights[layer][input][synapse] -= dw

    mean_gradient_array[0] = np.mean(gradient_array[0], 2)
    mean_gradient_array[1] = np.mean(gradient_array[1], 2)
    total_loss_array.append(np.mean(total_loss))


np.savetxt("weights_23092022_lr0.1_dw0.3", weights)
np.savetxt("loss_23092022_lr0.1_dw0.3", total_loss_array)
print("Done")
