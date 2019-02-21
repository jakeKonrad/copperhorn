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

    // In order from the left: the bias, the input connections, and the hidden
    // connections.
    pub struct Neuron(Real, HashMap<usize, Real>, HashMap<NeuronId, Real>);

    impl Neuron {
        // Computes the output of a neuron. TODO: vectorize.
        fn output(&self, xs: &[Real], cache: &HashMap<NeuronId, Real>) -> Real {
            let mut potential = self.0;

            for (i, w) in self.1.iter() {
                if let Some(x) = xs.get(*i) {
                    potential += *x * w;
                }
            }

            for (i, w) in self.2.iter() {
                if let Some(x) = cache.get(i) {
                    potential += *x * w;
                }
            }

            potential.tanh()
        }

        // Updates the weights according to Oja's rule.
        // TODO: vectorize.
        fn learn(&mut self, eta: Real, y: Real, xs: &[Real], cache: &HashMap<NeuronId, Real>) {
            let coefficient = eta * y;

            for (i, w) in self.1.iter_mut() {
                if let Some(x) = xs.get(*i) {
                    *w += coefficient * (*x - y * *w);
                }
            }

            for (i, w) in self.2.iter_mut() {
                if let Some(x) = cache.get(i) {
                    *w += coefficient * (*x - y * *w);
                }
            }

            y
        }
    }

    // Has fields: hidden, contains the hidden neurons;
    // output, holds the output neurons.
    pub struct Organism {
        hidden: HashMap<NeuronId, Neuron>,
        output: Box<[Neuron]>,
    }

    impl Organism {
        // Generates a new organism of 'minimalistic' topology. Generates random connections
        // between output layer and input layer with no hidden neurons.
        pub fn new(x: usize, y: usize) -> Self {
            let mut rng = SmallRng::from_entropy();

            let input: &[usize] = &(0..(x - 1)).collect::<Vec<_>>()[..];

            let hidden = HashMap::new();

            let output = (0..(y - 1))
                .map(|_| {
                    let n = rng.gen_range(1, x);
                    let mut ws = HashMap::with_capacity(n);
                    for j in input.choose_multiple(&mut rng, n) {
                        ws.insert(*j, rng.gen());
                    }
                    Neuron(rng.gen(), ws, HashMap::new())
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();

            Organism { hidden, output }
        }

        // Computes output of neural network.
        pub fn output(&self, xs: &[Real]) -> Box<[Real]> {
            // Sets up a cache to hold already fired neurons.
            let mut cache = HashMap::with_capacity(self.hidden.len());

            // Computes the topological ordering of the hidden neurons
            // and computes neuron firing and stores the results
            // in the cache. TODO: support firing neurons in parallel.
            for i in top_sort(self.hidden).iter() {
                if let Some(n) = self.hidden.get(i) {
                    let y = n.output(xs, &cache);
                    cache.insert(*i, y);
                }
            }

            // Sets up output vector.
            let mut ys = Vec::with_capacity(self.output.len());

            // Fills the output vector. TODO:
            // support firing neurons in parallel.
            for n in self.output.iter() {
                let y = n.output(xs, &cache);
                ys.push(y);
            }

            ys.into_boxed_slice()
        }

        pub fn learn(&mut self, eta: Real, xs: &[Real]) {
            // Sets up a cache to hold already fired neurons.
            let mut cache = HashMap::with_capacity(self.hidden.len());

            // Computes the topological ordering of the hidden neurons
            // and computes neuron firing and stores the results
            // in the cache. TODO: support firing neurons in parallel.
            for i in top_sort(self.hidden).iter() {
                if let Some(n) = self.hidden.get(i) {
                    let y = n.output(xs, &cache);
                    n.learn(eta, y, xs, &cache);
                    cache.insert(*i, y);
                }
            }

            for n in self.output.iter_mut() {
                let y = n.output(xs, &cache);
                n.learn(eta, y, xs, &cache);
            }
        }
    }

    // Utility functions.

    // See issue with visit.
    fn top_sort(graph: &HashMap<NeuronId, Neuron>) -> Box<[NeuronId]> {
        let mut stack = Vec::with_capacity(graph.len());
        let mut visited = HashSet::with_capacity(graph.len());
        for i in graph.keys() {
            visit(*i, graph, &mut stack, &mut visited);
        }
        stack.into_boxed_slice()
    }

    // TODO: needs proper error handling if graph has a loop
    // this will overflow the stack.
    fn visit(
        i: NeuronId,
        graph: &HashMap<NeuronId, Neuron>,
        stack: &mut Vec<NeuronId>,
        visited: &mut HashSet<NeuronId>,
    ) {
        if !(visited.contains(&i)) {
            if let Some(n) = graph.get(&i) {
                for j in n.2.keys() {
                    visit(*j, graph, stack, visited);
                }
                visited.insert(i);
                stack.push(i)
            }
        }
    }

    // TODO: unit tests.
    #[cfg(test)]
    mod tests {
        use super::*;

    }
}
