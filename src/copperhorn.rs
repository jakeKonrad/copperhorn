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
            if let Some(x) = xs.get(*i) {
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
        // Fails if there is a loop in the hidden neurons.
        pub fn act(&self, xs: &Input) -> Option<Output> {
            // Computes the topological ordering of the hidden neurons
            // , if the topological ordering encounters a loop
            // the whole function fails.
            match top_sort(&self.0) {
                None => None,
                Some(ns) => {
                    // Sets up a cache to hold already fired neurons.
                    let mut cache = HashMap::with_capacity(self.0.len());

                    for (i, n) in ns {
                        let y = fire(&cache, xs, n);
                        cache.insert(i, y);
                    }

                    // Computes output vector.
                    let mut ys = Vec::with_capacity(self.1.len());
                    for n in self.1.iter() {
                        let y = fire(&cache, xs, n);
                        ys.push(y);
                    }
                    Some(ys)
                }
            }
        }
    }

    // Utility functions.

    fn top_sort<'a>(graph: &'a HashMap<NeuronId, Neuron>) -> Option<Vec<(NeuronId, &'a Neuron)>> {
        let mut stack = Vec::with_capacity(graph.len());
        let mut visited = HashMap::with_capacity(graph.len());
        match graph
            .keys()
            .try_for_each(|i| visit(*i, graph, &mut stack, &mut visited))
        {
            None => None,
            Some(_) => Some(stack),
        }
    }

    fn visit<'a>(
        i: NeuronId,
        graph: &'a HashMap<NeuronId, Neuron>,
        stack: &mut Vec<(NeuronId, &'a Neuron)>,
        visited: &mut HashMap<NeuronId, bool>,
    ) -> Option<()> {
        match visited.get(&i) {
            None => match graph.get(&i) {
                Some(n) => {
                    visited.insert(i, false);
                    match n
                        .0
                        .hidden
                        .keys()
                        .try_for_each(|j| visit(*j, graph, stack, visited))
                    {
                        None => None,
                        Some(_) => {
                            visited.insert(i, true);
                            stack.push((i, n));
                            Some(())
                        }
                    }
                }
                None => None,
            },
            Some(mark) => {
                if *mark {
                    Some(())
                } else {
                    None
                }
            }
        }
    }

}
