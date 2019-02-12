mod copperhorn {

    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::FromEntropy;
    use rand::Rng;
    use std::collections::HashMap;
    use std::collections::HashSet;

    // Real value numbers.
    type Real = f64;

    // Identifiers for hidden neurons.
    type NeuronId = u32;

    // Holds the weights coming from the input
    // and the weights coming from the hidden neurons.
    pub struct Neuron {
        bias: Real,
        input: HashMap<usize, Real>,
        hidden: HashMap<NeuronId, Real>,
    }

    impl Neuron {
        // Computes the output of a neuron. TODO: vectorize.
        fn fire(&self, cache: &HashMap<NeuronId, Real>, xs: &[Real]) -> Real {
            let mut acc = self.bias;
            for (i, w) in self.input.iter() {
                if let Some(x) = xs.get(*i) {
                    acc += x * w;
                }
            }
            for (i, w) in self.hidden.iter() {
                if let Some(x) = cache.get(i) {
                    acc += x * w;
                }
            }
            acc
        }
    }

    // Holds the hidden neurons and the output layer.
    pub struct Organism(HashMap<NeuronId, Neuron>, Vec<Neuron>);

    impl Organism {
        // Generates a new organism. Generates random connections between output layer
        // and input layer with no hidden neurons.
        pub fn new(x: usize, y: usize) -> Self {
            let mut rng = SmallRng::from_entropy();
            let inputs: &[usize] = &(0..(x - 1)).collect::<Vec<_>>()[..];
            let output_neurons = (0..(y - 1))
                .map(|_| {
                    let n: usize = rng.gen_range(1, x);
                    let mut ws = HashMap::with_capacity(n);
                    for j in inputs.choose_multiple(&mut rng, n) {
                        ws.insert(*j, rng.gen());
                    }
                    Neuron {
                        bias: rng.gen(),
                        input: ws,
                        hidden: HashMap::new(),
                    }
                })
                .collect();
            Organism(HashMap::new(), output_neurons)
        }

        // Computes output of neural network.
        pub fn act(&self, xs: &[Real]) -> Vec<Real> {
            // Sets up a cache to hold already fired neurons.
            let mut cache = HashMap::with_capacity(self.0.len());

            // Computes the topological ordering of the hidden neurons
            // and computes neuron firing and stores the results
            // in the cache. TODO: support firing neurons in parallel.
            for (i, n) in top_sort(&self.0) {
                let y = n.fire(&cache, xs);
                cache.insert(*i, y);
            }

            // Sets up output vector.
            let mut ys = Vec::with_capacity(self.1.len());

            // Fills the output vector. TODO:
            // support firing neurons in parallel.
            for n in self.1.iter() {
                let y = n.fire(&cache, xs);
                ys.push(y);
            }
            ys
        }
    }

    // Utility functions.

    // See issue with visit.
    fn top_sort<'a>(graph: &'a HashMap<NeuronId, Neuron>) -> Vec<(NeuronId, &'a Neuron)> {
        let mut stack = Vec::with_capacity(graph.len());
        let mut visited = HashSet::with_capacity(graph.len());
        for i in graph {
            visit(*i, graph, &mut stack, &mut visited);
        }
        stack
    }

    // TODO: needs proper error handling if graph has a loop
    // this will overflow the stack.
    fn visit<'a>(
        i: NeuronId,
        graph: &'a HashMap<NeuronId, Neuron>,
        stack: &mut Vec<(NeuronId, &'a Neuron)>,
        visited: &mut HashSet<NeuronId>,
    ) {
        if !(visited.contains(i)) {
            if let Some(n) = graph.get(i) {
                for j in n.hidden.keys() {
                    visit(*j, graph, stack, visited);
                }
                visited.insert(*j);
                stack.push((*j, n))
            }
        }
    }

    // TODO: unit tests.
    #[cfg(test)]
    mod tests {
        use super::*;

    }
}
