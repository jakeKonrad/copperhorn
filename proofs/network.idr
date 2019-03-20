-- The purpose of this file is to prove properties
-- of the network structure. Based on inductive graphs because
-- they are easier to think about.
module Network

data Context : Nat -> [(Nat, Nat)]-> Type where
  MkContext : (dendrites: [Nat])
           -> (id: Nat)
           -> (axons: [Nat])
           -> Context id (map (\d => (d, id)) dendrites ++ map (\a => (id, a)))

data Graph : [Nat] -> [(Nat, Nat)] -> Type where
  Empty: Graph [] []
  (&): Context newNode newArcs -> Graph nodes arcs -> Graph (newNode::nodes) (newArcs ++ arcs)
