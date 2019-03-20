// TODO: remove in production.
#![allow(dead_code)]

use std::collections::HashMap;

// Has fields: input range, hidden neurons,
// and the output neurons.
struct Network {
    hidden_neurons: HashMap<u32, Neuron>,
    output_neurons: Box<[Neuron]>,
}

// Has fields: bias, input connections, and hidden
// connections.
struct Neuron {
    bias: f64,
    input_connections: HashMap<usize, f64>,
    hidden_connections: HashMap<u32, f64>,
}

enum NeuronEnum<'a> {
    HiddenId(u32),
    OutputNeuron(&'a Neuron),
}

// Computes the output of an organism.
fn output(network: &Network, xs: &[f64]) -> (HashMap<u32, f64>, Box<[f64]>) {
    // Sets up the cortex.
    let mut cortex: HashMap<u32, f64> = HashMap::with_capacity(network.hidden_neurons.len());
    let output = network
        .output_neurons
        .iter()
        .map(|neuron| {
            fire(
                NeuronEnum::OutputNeuron(neuron),
                xs,
                &network.hidden_neurons,
                &mut cortex,
            )
        })
        .collect::<Vec<f64>>()
        .into_boxed_slice();
    (cortex, output)
}

// Fires a neuron.
fn fire(
    neuron_enum: NeuronEnum,
    xs: &[f64],
    hidden_neurons: &HashMap<u32, Neuron>,
    cortex: &mut HashMap<u32, f64>,
) -> f64 {
    let (update_cortex, neuron) = match neuron_enum {
        NeuronEnum::OutputNeuron(neuron) => (None, neuron),
        NeuronEnum::HiddenId(id) => match cortex.get(&id) {
            Some(y) => return *y,
            None => (Some(id), hidden_neurons.get(&id).unwrap()),
        },
    };

    let mut potential: f64 = 0.0;

    for (other_id, weight) in neuron.input_connections.iter() {
        if let Some(x) = xs.get(*other_id) {
            potential += x * *weight;
        }
    }

    for (other_id, weight) in neuron.hidden_connections.iter() {
        let y = fire(NeuronEnum::HiddenId(*other_id), xs, hidden_neurons, cortex);
        potential += y * *weight;
    }

    let y = potential.tanh();

    match update_cortex {
        None => return y,
        Some(id) => {
            cortex.insert(id, y);
            return y;
        }
    }
}

// TODO: unit tests.
// #[cfg(test)]
// mod tests {
//    use super::*;
//}
