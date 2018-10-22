#![feature(nll)]
//! The purpose of this crate is to maintain an topological order in the face
//! of single updates, like adding new nodes, adding new depedencies, deleting
//! dependencies, and deleting nodes.
//!
//! Adding nodes, deleting nodes, and deleting dependencies require a trivial
//! amount of work to perform an update, because those operations do not change
//! the topological ordering. Adding new dependencies does
//!
//! ## What is a Topological Order
//!
//! To define a topological order requires at least simple definitions of a
//! graph, and specifically a directed acyclic graph (DAG). A graph can be
//! described as a pair of sets, `(V, E)` where `V` is the set of all nodes in
//! the graph, and `E` is the set of edges. An edge is defined as a pair, `(m,
//! n)` where `m` and `n` are nodes. A directed graph means that edges only
//! imply a single direction of relationship between two nodes, as opposed to a
//! undirected graph which implies the relationship goes both ways. An example
//! of undirected vs. directed in social networks would be Facebook vs.
//! Twitter. Facebook friendship is a two way relationship, while following
//! someone on Twitter does not imply that they follow you back.
//!
//! A topological ordering, `ord_D` of a directed acyclic graph, `D = (V, E)`
//! where `x, y ∈ V`, is a mapping of nodes to priority values such that
//! `ord_D(x) < ord_D(y)` holds for all edges `(x, y) ∈ E`. This yields a total
//! ordering of the nodes in `D`.
//!
//! ## Examples
//!
//! ```
//! use incremental_topo::IncrementalTopo;
//! use std::{cmp::Ordering::*, collections::HashSet};
//!
//! let mut dag = IncrementalTopo::new();
//!
//! let dog = dag.add_node();
//! let cat = dag.add_node();
//! let mouse = dag.add_node();
//! let lion = dag.add_node();
//! let human = dag.add_node();
//! let gazelle = dag.add_node();
//! let grass = dag.add_node();
//!
//! assert_eq!(dag.size(), 7);
//!
//! dag.add_dependency(lion, human).unwrap();
//! dag.add_dependency(lion, gazelle).unwrap();
//!
//! dag.add_dependency(human, dog).unwrap();
//! dag.add_dependency(human, cat).unwrap();
//!
//! dag.add_dependency(dog, cat).unwrap();
//! dag.add_dependency(cat, mouse).unwrap();
//!
//! dag.add_dependency(gazelle, grass).unwrap();
//!
//! dag.add_dependency(mouse, grass).unwrap();
//!
//! let pairs = dag
//!     .descendants_unsorted(human)
//!     .unwrap()
//!     .collect::<HashSet<_>>();
//! let expected_pairs = [(4, cat), (3, dog), (5, mouse), (7, grass)]
//!     .iter()
//!     .cloned()
//!     .collect::<HashSet<_>>();
//!
//! assert_eq!(pairs, expected_pairs);
//!
//! assert!(dag.contains_transitive_dependency(lion, grass));
//! assert!(!dag.contains_transitive_dependency(human, gazelle));
//!
//! assert_eq!(dag.topo_cmp(cat, dog).unwrap(), Greater);
//! assert_eq!(dag.topo_cmp(lion, human).unwrap(), Less);
//! ```
//!
//! ## Sources
//!
//! The [paper by D. J. Pearce and P. H. J. Kelly] contains descriptions of
//! three different algorithms for incremental topological ordering, along with
//! analysis of runtime bounds for each.
//!
//! [paper by D. J. Pearce and P. H. J. Kelly]: http://www.doc.ic.ac.uk/~phjk/Publications/DynamicTopoSortAlg-JEA-07.pdf

extern crate failure;
#[macro_use]
extern crate failure_derive;
#[macro_use]
extern crate log;
extern crate generational_arena;

use generational_arena::{Arena, Index as ArenaIndex};
use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashSet},
    iter::Iterator,
};

/// Data structure for maintaining a topological ordering over a collection of
/// elements, in an incremental fashion.
///
/// See the [module-level documentation] for more information.
///
/// [module-level documentation]: index.html
#[derive(Debug, Clone)]
pub struct IncrementalTopo {
    nodes: Arena<NodeData>,
    next_topo_value: u32,
    node_count: usize,
    edge_count: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TopoKey(ArenaIndex);

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct NodeData {
    topo_order: u32,
    parents: HashSet<ArenaIndex>,
    children: HashSet<ArenaIndex>,
}

impl NodeData {
    fn new(topo_order: u32) -> Self {
        NodeData {
            topo_order,
            parents: HashSet::new(),
            children: HashSet::new(),
        }
    }
}

impl PartialOrd for NodeData {
    fn partial_cmp(&self, other: &NodeData) -> Option<Ordering> {
        self.topo_order.partial_cmp(&other.topo_order)
    }
}

impl Ord for NodeData {
    fn cmp(&self, other: &NodeData) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Different types of failures that can occur while updating or querying the
/// graph.
#[derive(Fail, Debug)]
pub enum Error {
    #[fail(display = "Node was not found in graph")]
    NodeMissing,
    #[fail(display = "Nodes may not transitively depend on themselves in a cyclic fashion")]
    CycleDetected,
}

pub type Result<T> = std::result::Result<T, Error>;

impl IncrementalTopo {
    /// Create a new IncrementalTopo graph.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// let dag = IncrementalTopo::<usize>::new();
    ///
    /// assert!(dag.is_empty());
    /// ```
    pub fn new() -> Self {
        IncrementalTopo {
            nodes: Arena::new(),
            next_topo_value: 0,
            edge_count: 0,
            node_count: 0,
        }
    }

    /// Request the creation of a new node.
    ///
    /// Initially this node will not have any order relative to the values
    /// that are already in the graph. Only when relations are added
    /// with [`add_dependency`] will the order begin to matter.
    ///
    /// Returns the key associated with the node. Do not use this key with any
    /// other instance of IncremenalTopo.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("dog"));
    /// assert!(dag.add_node("mouse"));
    /// assert!(dag.add_node("human"));
    ///
    /// assert!(!dag.add_node("cat"));
    /// ```
    ///
    /// [`add_dependency`]: struct.IncrementalTopo.html#method.add_dependency
    pub fn add_node(&mut self) -> TopoKey {
        let node_data = NodeData::new(self.next_topo_value);

        info!("Created node {:?}", node_data);

        let index = self.nodes.insert(node_data);

        self.next_topo_value += 1;
        self.node_count -= 1;

        TopoKey(index)
    }

    /// Returns true if the graph contains the node associated with the key.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("dog"));
    ///
    /// assert!(dag.contains_node("cat"));
    /// assert!(dag.contains_node("dog"));
    ///
    /// assert!(!dag.contains_node("horse"));
    /// assert!(!dag.contains_node("orc"));
    /// ```
    pub fn contains_node(&self, key: TopoKey) -> bool {
        let TopoKey(inner) = key;

        self.nodes.contains(inner)
    }

    /// Attempt to remove node from graph, returning true if the node was
    /// contained and removed.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("dog"));
    ///
    /// assert!(dag.delete_node("cat"));
    /// assert!(dag.delete_node(&"dog"));
    ///
    /// assert!(!dag.delete_node("horse"));
    /// assert!(!dag.delete_node(&"orc"));
    /// ```
    pub fn delete_node(&mut self, key: TopoKey) -> bool {
        if let Some(data) = self.nodes.remove(key.0) {
            for child in data.children {
                if let Some(child_data) = self.nodes.get_mut(child) {
                    child_data.parents.remove(&key.0);
                }
            }

            // Delete backward edges
            for parent in data.parents {
                if let Some(parent_data) = self.nodes.get_mut(parent) {
                    parent_data.children.remove(&key.0);
                }
            }

            // TODO Fix inefficient compaction step
            for (_, node_data) in self.nodes.iter_mut() {
                if node_data.topo_order > data.topo_order {
                    node_data.topo_order -= 1;
                }
            }

            // Decrement last topo order to account for shifted topo values
            self.next_topo_value -= 1;
            self.node_count -= 1;

            true
        } else {
            false
        }
    }

    /// Add a directed link between two nodes already present in the graph.
    ///
    /// This link indicates an ordering constraint on the two nodes, now `prec`
    /// must always come before `succ` in the ordering.
    ///
    /// The values of `prec` and `succ` may be any borrowed form of the graph's
    /// node type, but Hash and Eq on the borrowed form must match those for
    /// the node type.
    ///
    /// Returns `Ok(true)` if the graph did not previously contain this
    /// dependency. Returns `Ok(false)` if the graph did have a previous
    /// dependency between these two nodes.
    ///
    /// # Errors
    /// This function will return an `Err` if the dependency introduces a cycle
    /// into the graph or if either of the nodes passed is not found in the
    /// graph.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("dog"));
    /// assert!(dag.add_node("mouse"));
    /// assert!(dag.add_node("human"));
    ///
    /// assert!(dag.add_dependency("human", "dog").unwrap());
    /// assert!(dag.add_dependency("human", "cat").unwrap());
    /// assert!(dag.add_dependency("cat", "mouse").unwrap());
    /// ```
    pub fn add_dependency(&mut self, prec: TopoKey, succ: TopoKey) -> Result<bool> {
        if !(self.nodes.contains(prec.0) && self.nodes.contains(succ.0)) {
            return Err(Error::NodeMissing);
        }

        if prec == succ {
            // No loops to self
            return Err(Error::CycleDetected);
        }

        // Insert forward edge
        let mut no_prev_edge = self.nodes[prec.0].children.insert(succ.0);
        let upper_bound = self.nodes[prec.0].topo_order;

        // Insert backward edge
        no_prev_edge = no_prev_edge && self.nodes[succ.0].parents.insert(prec.0);
        let lower_bound = self.nodes[succ.0].topo_order;

        // If edge already exists short circuit
        if !no_prev_edge {
            return Ok(false);
        }

        info!("Adding edge from {:?} to {:?}", prec.0, succ.0);

        trace!(
            "Upper: Order({}), Lower: Order({})",
            upper_bound,
            lower_bound
        );
        // If the affected region of the graph has non-zero size (i.e. the upper and
        // lower bound are equal) then perform an update to the topological ordering of
        // the graph
        if lower_bound < upper_bound {
            trace!("Will change");
            let mut visited = HashSet::new();

            // Walk changes forward from the succ, checking for any cycles that would be
            // introduced
            let change_forward = self.dfs_forward(succ.0, &mut visited, upper_bound)?;
            trace!("Change forward: {:?}", change_forward);
            // Walk backwards from the prec
            let change_backward = self.dfs_backward(prec.0, &mut visited, lower_bound);
            trace!("Change backward: {:?}", change_backward);

            self.reorder_nodes(change_forward, change_backward);
        } else {
            trace!("No change");
        }

        self.edge_count += 1;

        Ok(true)
    }

    /// Returns true if the graph contains a dependency from `prec` to `succ`.
    ///
    /// Returns false if either node is not found, or if there is no dependency.
    ///
    /// The values of `prec` and `succ` may be any borrowed form of the graph's
    /// node type, but Hash and Eq on the borrowed form must match those for
    /// the node type.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("mouse"));
    /// assert!(dag.add_node("human"));
    ///
    /// assert!(dag.add_dependency("human", "cat").unwrap());
    /// assert!(dag.add_dependency("cat", "mouse").unwrap());
    ///
    /// assert!(dag.contains_dependency("cat", "mouse"));
    /// assert!(!dag.contains_dependency("human", "mouse"));
    /// assert!(!dag.contains_dependency("cat", "horse"));
    /// ```
    pub fn contains_dependency(&self, prec: TopoKey, succ: TopoKey) -> bool {
        if let Some(node_data) = self.nodes.get(prec.0) {
            node_data.children.contains(&succ.0)
        } else {
            false
        }
    }

    /// Returns true if the graph contains a transitive dependency from `prec`
    /// to `succ`.
    ///
    /// In this context a transitive dependency means that `succ` exists as a
    /// descendant of `prec`, with some chain of other nodes in between.
    ///
    /// Returns false if either node is not found in the graph, or there is no
    /// transitive dependency.
    ///
    /// The values of `prec` and `succ` may be any borrowed form of the graph's
    /// node type, but Hash and Eq on the borrowed form must match those for
    /// the node type.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("mouse"));
    /// assert!(dag.add_node("human"));
    /// assert!(dag.add_node("dog"));
    ///
    /// assert!(dag.add_dependency("human", "cat").unwrap());
    /// assert!(dag.add_dependency("human", "dog").unwrap());
    /// assert!(dag.add_dependency("cat", "mouse").unwrap());
    ///
    /// assert!(dag.contains_transitive_dependency("human", "mouse"));
    /// assert!(!dag.contains_transitive_dependency("dog", "mouse"));
    /// ```
    pub fn contains_transitive_dependency(&self, prec: TopoKey, succ: TopoKey) -> bool {
        // If either node is missing, return quick
        if !(self.nodes.contains(prec.0) && self.nodes.contains(succ.0)) {
            return false;
        }

        // A node cannot depend on itself
        if prec == succ {
            return false;
        }

        // Else we have to search the graph. Using dfs in this case because it avoids
        // the overhead of the binary heap, and this task doesn't really need ordered
        // descendants.
        let mut stack = Vec::new();
        let mut visited = HashSet::new();

        stack.push(prec.0);

        // For each node key popped off the stack, check that we haven't seen it
        // before, then check if its children contain the node we're searching for.
        // If they don't, continue the search by extending the stack with the children.
        while let Some(key) = stack.pop() {
            if visited.contains(&key) {
                continue;
            } else {
                visited.insert(key);
            }

            let children = &self.nodes[key].children;

            if children.contains(&succ.0) {
                return true;
            } else {
                stack.extend(children);

                continue;
            }
        }

        // If we exhaust the stack, then there is no transitive dependency.
        false
    }

    /// Attempt to remove a dependency from the graph, returning true if the
    /// dependency was removed.
    ///
    /// Returns false is either node is not found in the graph.
    ///
    /// The values of `prec` and `succ` may be any borrowed form of the graph's
    /// node type, but Hash and Eq on the borrowed form must match those for
    /// the node type.
    ///
    /// Removing a dependency from the graph is an extremely simple operation,
    /// which requires no recalculation of the topological order. The ordering
    /// before and after a removal is exactly the same.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("mouse"));
    /// assert!(dag.add_node("human"));
    /// assert!(dag.add_node("dog"));
    ///
    /// assert!(dag.add_dependency("human", "cat").unwrap());
    /// assert!(dag.add_dependency("human", "dog").unwrap());
    /// assert!(dag.add_dependency("cat", "mouse").unwrap());
    ///
    /// assert!(dag.delete_dependency("cat", "mouse"));
    /// assert!(dag.delete_dependency("human", "dog"));
    /// assert!(!dag.delete_dependency("human", "mouse"));
    /// ```
    pub fn delete_dependency(&mut self, prec: TopoKey, succ: TopoKey) -> bool {
        if !(self.nodes.contains(prec.0) && self.nodes.contains(succ.0)) {
            return false;
        }

        let prec_children = &mut self.nodes[prec.0].children;

        if !prec_children.contains(&succ.0) {
            return false;
        }

        prec_children.remove(&succ.0);
        self.nodes[succ.0].parents.remove(&prec.0);

        self.edge_count -= 1;

        true
    }

    /// Return the number of nodes within the graph.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("mouse"));
    /// assert!(dag.add_node("human"));
    /// assert!(dag.add_node("dog"));
    ///
    /// assert_eq!(dag.size(), 4);
    /// ```
    pub fn node_size(&self) -> usize {
        self.node_count
    }

    pub fn edge_size(&self) -> usize {
        self.edge_count
    }

    /// Return true if there are no nodes in the graph.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.is_empty());
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("mouse"));
    /// assert!(dag.add_node("human"));
    /// assert!(dag.add_node("dog"));
    ///
    /// assert!(!dag.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.node_size() == 0
    }

    /// Return an iterator over the nodes of the graph
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// use std::collections::HashSet;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("mouse"));
    /// assert!(dag.add_node("human"));
    /// assert!(dag.add_node("dog"));
    ///
    /// assert!(dag.add_dependency("human", "cat").unwrap());
    /// assert!(dag.add_dependency("human", "dog").unwrap());
    /// assert!(dag.add_dependency("cat", "mouse").unwrap());
    ///
    /// let pairs = dag.iter_unsorted().collect::<HashSet<_>>();
    ///
    /// let mut expected_pairs = HashSet::new();
    /// expected_pairs.extend(vec![(1, &"human"), (2, &"cat"), (3, &"mouse"), (4, &"dog")]);
    ///
    /// assert_eq!(pairs, expected_pairs);
    /// ```
    pub fn iter_unsorted(&self) -> impl Iterator<Item = (u32, TopoKey)> + '_ {
        self.nodes
            .iter()
            .map(|(key, data)| (data.topo_order, TopoKey(key)))
    }

    /// Return an iterator over the descendants of a node in the graph, in an
    /// unosrted order.
    ///
    /// The passed node may be any borrowed form of the graph's node type, but
    /// Hash and Eq on the borrowed form must match those for the node type.
    ///
    /// Accessing the nodes in an unsorted order allows for faster access using
    /// a iterative DFS search. This is opposed to the order descendants
    /// iterator which requires the use of a binary heap to order the values.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// use std::collections::HashSet;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("mouse"));
    /// assert!(dag.add_node("dog"));
    /// assert!(dag.add_node("human"));
    ///
    /// assert!(dag.add_dependency("human", "cat").unwrap());
    /// assert!(dag.add_dependency("human", "dog").unwrap());
    /// assert!(dag.add_dependency("dog", "cat").unwrap());
    /// assert!(dag.add_dependency("cat", "mouse").unwrap());
    ///
    /// let pairs = dag
    ///     .descendants_unsorted("human")
    ///     .unwrap()
    ///     .collect::<HashSet<_>>();
    ///
    /// let mut expected_pairs = HashSet::new();
    /// expected_pairs.extend(vec![(2, &"dog"), (3, &"cat"), (4, &"mouse")]);
    ///
    /// assert_eq!(pairs, expected_pairs);
    /// ```
    pub fn descendants_unsorted(&self, node: TopoKey) -> Result<DescendantsUnsorted> {
        let TopoKey(inner_key) = node;

        if let Some(node_data) = self.nodes.get(inner_key) {
            let mut stack = Vec::new();
            let visited = HashSet::new();

            // Add all children of selected node
            stack.extend(&node_data.children);

            Ok(DescendantsUnsorted {
                dag: self,
                stack,
                visited,
            })
        } else {
            Err(Error::NodeMissing)
        }
    }

    /// Return an iterator over descendants of a node in the graph, in a
    /// topologically sorted order.
    ///
    /// The passed node may be any borrowed form of the graph's node type, but
    /// Hash and Eq on the borrowed form must match those for the node type.
    ///
    /// Accessing the nodes in a sorted order requires the use of a BinaryHeap,
    /// so some performance penalty is paid there. If all is required is access
    /// to the descendants of a node, use [`descendants_unsorted`].
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("mouse"));
    /// assert!(dag.add_node("dog"));
    /// assert!(dag.add_node("human"));
    ///
    /// assert!(dag.add_dependency("human", "cat").unwrap());
    /// assert!(dag.add_dependency("human", "dog").unwrap());
    /// assert!(dag.add_dependency("dog", "cat").unwrap());
    /// assert!(dag.add_dependency("cat", "mouse").unwrap());
    ///
    /// let ordered_nodes = dag.descendants("human").unwrap().collect::<Vec<_>>();
    ///
    /// assert_eq!(ordered_nodes, vec![&"dog", &"cat", &"mouse"]);
    /// ```
    ///
    /// [`descendants_unsorted`]:
    /// struct.IncrementalTopo.html#method.descendants_unsorted
    pub fn descendants(&self, node: TopoKey) -> Result<Descendants> {
        if let Some(node_data) = self.nodes.get(node.0) {
            let mut queue = BinaryHeap::new();

            // Add all children of selected node
            queue.extend(node_data.children.iter().cloned().map(|child_key| {
                let child_order = self.nodes[child_key].topo_order;
                (Reverse(child_order), child_key)
            }));

            let visited = HashSet::new();

            Ok(Descendants {
                dag: self,
                queue,
                visited,
            })
        } else {
            Err(Error::NodeMissing)
        }
    }

    /// Compare two nodes present in the graph, topographically. Returns
    /// Err(...) if either node is missing from the graph.
    ///
    /// The values of `prec` and `succ` may be any borrowed form of the graph's
    /// node type, but Hash and Eq on the borrowed form must match those for
    /// the node type.
    ///
    /// # Examples
    /// ```
    /// use incremental_topo::IncrementalTopo;
    /// use std::cmp::Ordering::*;
    /// let mut dag = IncrementalTopo::new();
    ///
    /// assert!(dag.add_node("cat"));
    /// assert!(dag.add_node("mouse"));
    /// assert!(dag.add_node("dog"));
    /// assert!(dag.add_node("human"));
    ///
    /// assert!(dag.add_dependency("human", "cat").unwrap());
    /// assert!(dag.add_dependency("human", "dog").unwrap());
    /// assert!(dag.add_dependency("dog", "cat").unwrap());
    /// assert!(dag.add_dependency("cat", "mouse").unwrap());
    ///
    /// assert_eq!(dag.topo_cmp("human", "mouse").unwrap(), Less);
    /// assert_eq!(dag.topo_cmp("cat", "dog").unwrap(), Greater);
    /// assert!(dag.topo_cmp("cat", "horse").is_err());
    /// ```
    pub fn topo_cmp(&self, node_a: TopoKey, node_b: TopoKey) -> Result<Ordering> {
        if let Some(a_data) = self.nodes.get(node_a.0) {
            if let Some(b_data) = self.nodes.get(node_b.0) {
                Ok(a_data.topo_order.cmp(&b_data.topo_order))
            } else {
                Err(Error::NodeMissing)
            }
        } else {
            Err(Error::NodeMissing)
        }
    }

    fn dfs_forward(
        &self,
        start_key: ArenaIndex,
        visited: &mut HashSet<ArenaIndex>,
        upper_bound: u32,
    ) -> Result<HashSet<ArenaIndex>> {
        let mut stack = Vec::new();
        let mut result = HashSet::new();

        stack.push(start_key);

        while let Some(next_key) = stack.pop() {
            visited.insert(next_key);
            result.insert(next_key);

            for child_key in &self.nodes[next_key].children {
                let child_topo_order = self.nodes[*child_key].topo_order;

                if child_topo_order == upper_bound {
                    return Err(Error::CycleDetected);
                }

                if !visited.contains(child_key) && child_topo_order < upper_bound {
                    stack.push(*child_key);
                }
            }
        }

        Ok(result)
    }

    fn dfs_backward(
        &self,
        start_key: ArenaIndex,
        visited: &mut HashSet<ArenaIndex>,
        lower_bound: u32,
    ) -> HashSet<ArenaIndex> {
        let mut stack = Vec::new();
        let mut result = HashSet::new();

        stack.push(start_key);

        while let Some(next_key) = stack.pop() {
            visited.insert(next_key);
            result.insert(next_key);

            for parent_key in &self.nodes[next_key].parents {
                let parent_topo_order = self.nodes[*parent_key].topo_order;

                if !visited.contains(&parent_key) && lower_bound < parent_topo_order {
                    stack.push(*parent_key);
                }
            }
        }

        result
    }

    fn reorder_nodes(
        &mut self,
        change_forward: HashSet<ArenaIndex>,
        change_backward: HashSet<ArenaIndex>,
    ) {
        let mut change_forward: Vec<_> = change_forward
            .into_iter()
            .map(|key| (key, self.nodes[key].topo_order))
            .collect();
        change_forward.sort_unstable_by_key(|pair| pair.1);

        let mut change_backward: Vec<_> = change_backward
            .into_iter()
            .map(|key| (key, self.nodes[key].topo_order))
            .collect();
        change_backward.sort_unstable_by_key(|pair| pair.1);

        let mut all_keys = Vec::new();
        let mut all_topo_orders = Vec::new();

        for (key, topo_order) in change_backward {
            all_keys.push(key);
            all_topo_orders.push(topo_order);
        }

        for (key, topo_order) in change_forward {
            all_keys.push(key);
            all_topo_orders.push(topo_order);
        }

        all_topo_orders.sort_unstable();

        for (key, topo_order) in all_keys.into_iter().zip(all_topo_orders.into_iter()) {
            self.nodes[key].topo_order = topo_order;
        }
    }
}

pub struct DescendantsUnsorted<'a> {
    dag: &'a IncrementalTopo,
    stack: Vec<ArenaIndex>,
    visited: HashSet<ArenaIndex>,
}

impl<'a> Iterator for DescendantsUnsorted<'a> {
    type Item = (u32, TopoKey);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(key) = self.stack.pop() {
            if self.visited.contains(&key) {
                continue;
            } else {
                self.visited.insert(key);
            }

            let order = self.dag.nodes[key].topo_order;

            self.stack.extend(&self.dag.nodes[key].children);

            return Some((order, TopoKey(key)));
        }

        return None;
    }
}

pub struct Descendants<'a> {
    dag: &'a IncrementalTopo,
    queue: BinaryHeap<(Reverse<u32>, ArenaIndex)>,
    visited: HashSet<ArenaIndex>,
}

impl<'a> Iterator for Descendants<'a> {
    type Item = TopoKey;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((_, key)) = self.queue.pop() {
                if self.visited.contains(&key) {
                    continue;
                } else {
                    self.visited.insert(key);
                }

                for child in &self.dag.nodes[key].children {
                    let order = self.dag.nodes[*child].topo_order;
                    self.queue.push((Reverse(order), *child))
                }

                return Some(TopoKey(key));
            } else {
                return None;
            }
        }
    }
}
