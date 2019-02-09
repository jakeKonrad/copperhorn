mod copperhorn {

    use std::collections::HashMap;
    use std::collections::HashSet;

    // Real value numbers.
    type Real = f64;

    // The input space are vectors of real numbers
    // represented by the Weight type.
    type InputIndex = usize;

    // Identifiers for hidden neurons. 
    type NeuronId = u32;

    // Holds the weights coming from the input
    // and the weights coming from the hidden neurons.
    pub struct Weights {
        input: HashMap<InputIndex, Real>,
        hidden: HashMap<NeuronId, Real>,
    }

    // Holds the connections and the weights 
    // of the neuron.
    pub struct Neuron(Weights);

    // Holds the hidden neurons and the output layer.
    pub struct Organism(HashMap<NeuronId, Neuron>, Vec<Neuron>);

    // Outputs of neurons that have already
    // fired.
    type Signals = HashMap<NeuronId, Real>;

    // The input vector.
    type Input = Vec<Real>;

    // Computes the output of a "neuron".
    fn fire(cache: &Signals, xs: &Input, n: &Neuron) -> Real {
        let mut acc = 0.0;
        for (i, w) in n.0.input.iter() {
            if let Some(x) = xs.get(i) {
                acc += x * w;
            }
        }
        for (i, w) in n.0.hidden.iter() {
            if let Some(x) = cache.get(i) {
                acc += x * w;
            }
        }
        acc
    }

    // Output vector.
    type Output = Vec<Real>;
    
    impl Organism {

        // Single threaded algorithm. Computes output of neural network.
        pub fn act(&self, xs: &Input) -> Output {
            // Sets up a cache to hold already fired neurons.
            let mut cache = HashMap::with_capacity(self.0.len());

            // Populates the cache. Computes the topological ordering of 
            // the hidden neurons, then computes the output of each neuron
            // and caches it.
            for (i, n) in top_sort(&self.0) {
                let y = fire(&cache, xs, n);
                cache.insert(i, y);
            }
            
            // Computes output vector.
            let mut ys = Vec::with_capacity(self.1.len());
            for n in self.1.iter() {
                let y = fire(&cache, xs, n);
                ys.push(y);
            }
            ys
        }
    }

    // Utility functions.

    fn top_sort<'a>(graph: &'a HashMap<u32, Neuron>) -> Vec<(u32, &'a Neuron)> {
        let mut stack = Vec::with_capacity(graph.len());
        let mut visited = HashSet::with_capacity(graph.len());
        for i in graph.keys() {
            visit(*i, graph, &mut stack, &mut visited);
        }
        stack
    }

    fn visit<'a>(i: u32, graph: &'a HashMap<u32, Neuron>, stack: &mut Vec<(u32, &'a Neuron)>, visited: &mut HashSet<u32>) {
        if !(visited.contains(&i)) {
            if let Some(n) = graph.get(&i) {
                for j in n.hidden.keys() {
                    visit(*j, graph, stack, visited);
                }
                visited.insert(i);
                stack.push((i, n));
            }
        }
    }

}
