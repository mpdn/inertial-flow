use fixedbitset::FixedBitSet;
use petgraph::csr::{Csr, EdgeReference, Edges};
use petgraph::graph::IndexType;
use petgraph::visit::{EdgeRef, IntoNeighbors};
use petgraph::EdgeType;
use std::collections::HashSet;
use std::hash::BuildHasher;
use std::hash::Hash;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum Side {
    Source,
    Mid,
    Sink,
}

pub trait Set<Ix>: Default
where
    for<'a> &'a Self: SetRef<Ix>,
{
    fn insert(&mut self, index: Ix) -> bool;
    fn remove(&mut self, index: &Ix) -> bool;
    fn contains(&self, index: &Ix) -> bool;
    fn clear(&mut self, size: usize);

    #[inline]
    fn extend(&mut self, indices: impl IntoIterator<Item = Ix>) {
        indices.into_iter().for_each(|x| {
            self.insert(x);
        })
    }
}

pub trait SetRef<Ix> {
    type Iter: Iterator<Item = Ix>;
    fn iter(self) -> Self::Iter;
}

impl<Ix, S> Set<Ix> for HashSet<Ix, S>
where
    Ix: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    #[inline]
    fn insert(&mut self, index: Ix) -> bool {
        self.insert(index)
    }

    #[inline]
    fn remove(&mut self, index: &Ix) -> bool {
        self.remove(index)
    }

    #[inline]
    fn contains(&self, index: &Ix) -> bool {
        self.contains(index)
    }

    #[inline]
    fn clear(&mut self, _: usize) {
        self.clear()
    }

    #[inline]
    fn extend(&mut self, indices: impl IntoIterator<Item = Ix>) {
        std::iter::Extend::extend(self, indices);
    }
}

impl<'a, Ix, S> SetRef<Ix> for &'a HashSet<Ix, S>
where
    Ix: Clone + Eq + Hash,
    S: BuildHasher + Default,
{
    type Iter = std::iter::Cloned<std::collections::hash_set::Iter<'a, Ix>>;
    fn iter(self) -> Self::Iter {
        HashSet::iter(self).cloned()
    }
}

impl<Ix> Set<Ix> for FixedBitSet
where
    Ix: IndexType,
{
    #[inline]
    fn insert(&mut self, index: Ix) -> bool {
        !self.put(index.index())
    }

    #[inline]
    fn remove(&mut self, index: &Ix) -> bool {
        let v = self[index.index()];
        self.set(index.index(), false);
        v
    }

    #[inline]
    fn contains(&self, index: &Ix) -> bool {
        self[index.index()]
    }

    #[inline]
    fn clear(&mut self, size: usize) {
        self.clear();

        if self.len() < size {
            self.grow(size);
        }
    }
}

pub struct FixedBitSetIter<'a, Ix>(fixedbitset::Ones<'a>, std::marker::PhantomData<Ix>);

impl<'a, Ix> Iterator for FixedBitSetIter<'a, Ix>
where
    Ix: IndexType,
{
    type Item = Ix;

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn next(&mut self) -> Option<Ix> {
        self.0.next().map(Ix::new)
    }
}

impl<'a, Ix> SetRef<Ix> for &'a FixedBitSet
where
    Ix: IndexType,
{
    type Iter = FixedBitSetIter<'a, Ix>;
    fn iter(self) -> Self::Iter {
        FixedBitSetIter(self.ones(), std::marker::PhantomData)
    }
}

struct Bfs<Ix, SN> {
    current: Vec<Ix>,
    visited: SN,
    next: Vec<Ix>,
}

impl<Ix, SN> Default for Bfs<Ix, SN>
where
    SN: Default,
{
    fn default() -> Self {
        Bfs {
            current: Vec::default(),
            visited: SN::default(),
            next: Vec::default(),
        }
    }
}

impl<Ix, SN> Bfs<Ix, SN>
where
    Ix: IndexType,
    SN: Set<Ix>,
    for<'x> &'x SN: SetRef<Ix>,
{
    fn bfs<N, E, Ty: EdgeType, SE>(
        &mut self,
        graph: &Csr<N, E, Ty, Ix>,
        source: impl IntoIterator<Item = Ix>,
        sink_set: &SN,
        edge_set: &mut SE,
        post_edges: &mut Vec<usize>,
    ) -> bool
    where
        SN: Set<Ix>,
        SE: Set<usize>,
        for<'x> &'x SN: SetRef<Ix>,
        for<'x> &'x SE: SetRef<usize>,
    {
        let Bfs {
            ref mut current,
            ref mut visited,
            ref mut next,
        } = self;

        visited.clear(graph.node_count());
        next.extend(source);

        let mut sink_reachable = false;

        while !current.is_empty() || !next.is_empty() {
            while let Some(node) = current.pop() {
                sink_reachable |= sink_set.contains(&node);

                for edge in graph.edges(node) {
                    if !edge_set.contains(&edge.id()) {
                        continue;
                    }

                    if visited.contains(&edge.target()) {
                        post_edges.push(edge.id());
                        edge_set.remove(&edge.id());
                    } else {
                        next.push(edge.target());
                    }
                }
            }

            current.extend(next.drain(..).filter(|&x| visited.insert(x)));
        }

        sink_reachable
    }
}

struct Dfs<'a, E, Ty, Ix> {
    stack: Vec<Edges<'a, E, Ty, Ix>>,
    path: Vec<EdgeReference<'a, E, Ty, Ix>>,
}

impl<'a, E, Ty, Ix> Default for Dfs<'a, E, Ty, Ix> {
    fn default() -> Self {
        Dfs {
            stack: Vec::new(),
            path: Vec::new(),
        }
    }
}

impl<'a, E, Ty, Ix> Dfs<'a, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn dfs<N, SN, SE>(
        &mut self,
        graph: &'a Csr<N, E, Ty, Ix>,
        sink_set: &mut SN,
        edge_set: &mut SE,
        post_edges: &mut Vec<usize>,
        node: Ix,
    ) -> bool
    where
        SN: Set<Ix>,
        SE: Set<usize>,
        for<'x> &'x SN: SetRef<Ix>,
        for<'x> &'x SE: SetRef<usize>,
    {
        let Dfs {
            ref mut stack,
            ref mut path,
        } = self;

        debug_assert!(stack.is_empty());
        debug_assert!(path.is_empty());

        stack.push(graph.edges(node));
        while let Some(edges) = stack.last_mut() {
            if let Some(edge) = edges.next() {
                if !edge_set.contains(&edge.id()) {
                    continue;
                }

                path.push(edge);

                if sink_set.remove(&edge.target()) {
                    stack.clear();
                    path.drain(..).for_each(|edge| {
                        let removed = edge_set.remove(&edge.id());

                        debug_assert!(removed);

                        post_edges.push(
                            graph
                                .edges(edge.target())
                                .find(|opposing_edge| opposing_edge.target() == edge.source())
                                .expect("Expected graph to have opposing edges")
                                .id(),
                        );
                    });

                    return true;
                } else {
                    stack.push(graph.edges(edge.target()));
                }
            } else {
                // Dead end, disable all edges from this node
                stack.pop();

                post_edges.extend(
                    graph
                        .edges(path.pop().map_or(node, |x| x.target()))
                        .filter(|edge| edge_set.remove(&edge.id()))
                        .map(|edge| edge.id()),
                );
            }
        }

        false
    }
}

pub struct Dinic<'a, E, Ty, Ix, SN, SE>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    node_set: SN,
    edge_set: SE,
    source_set: SN,
    sink_set: SN,

    bfs: Bfs<Ix, SN>,
    dfs: Dfs<'a, E, Ty, Ix>,

    source_vec: Vec<Ix>,
    post_edges: Vec<usize>,
}

impl<'a, E, Ty, Ix, SN, SE> Default for Dinic<'a, E, Ty, Ix, SN, SE>
where
    Ty: EdgeType,
    Ix: IndexType,
    SN: Default,
    SE: Default,
{
    fn default() -> Self {
        Dinic {
            node_set: SN::default(),
            edge_set: SE::default(),
            source_set: SN::default(),
            sink_set: SN::default(),
            bfs: Bfs::default(),
            dfs: Dfs::default(),
            source_vec: Vec::default(),
            post_edges: Vec::default(),
        }
    }
}

impl<'a, E, Ty, Ix, SN, SE> Dinic<'a, E, Ty, Ix, SN, SE>
where
    Ty: EdgeType,
    Ix: IndexType,
    SN: Set<Ix>,
    SE: Set<usize>,
    for<'x> &'x SN: SetRef<Ix>,
    for<'x> &'x SE: SetRef<usize>,
{
    pub fn vertex_dissect<'b, N>(
        &'b mut self,
        graph: &'a Csr<N, E, Ty, Ix>,
        nodes: impl IntoIterator<Item = (Ix, Side)>,
    ) -> VertexDissection<'b, 'a, N, E, Ty, Ix, SN, SE> {
        let &mut Dinic {
            ref mut node_set,
            ref mut edge_set,
            ref mut sink_set,
            ref mut source_set,
            ref mut bfs,
            ref mut dfs,
            ref mut source_vec,
            ref mut post_edges,
        } = self;

        source_vec.clear();
        post_edges.clear();

        node_set.clear(graph.node_count());
        edge_set.clear(graph.edge_count());
        source_set.clear(graph.node_count());
        sink_set.clear(graph.node_count());

        for (node, side) in nodes {
            assert!(
                node.index() < graph.node_count(),
                "Node index outside of node bound"
            );

            let new = node_set.insert(node);

            assert!(new, "A node must only appear once in the node list");

            match side {
                Side::Source => {
                    source_set.insert(node);
                    source_vec.push(node);
                }
                Side::Sink => {
                    sink_set.insert(node);
                }
                _ => {}
            }
        }

        edge_set.extend(
            node_set
                .iter()
                .flat_map(|node| graph.edges(node))
                .filter(|edge| node_set.contains(&edge.target()))
                .map(|edge| edge.id()),
        );

        while bfs.bfs(
            graph,
            source_vec.iter().cloned(),
            sink_set,
            edge_set,
            post_edges,
        ) {
            source_vec.retain(|&node| !dfs.dfs(graph, sink_set, edge_set, post_edges, node));

            edge_set.extend(post_edges.drain(..));
        }

        VertexDissection { dinic: self, graph }
    }
}

pub struct VertexDissection<'a, 'b, N, E, Ty, Ix, SN, SE>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    dinic: &'a Dinic<'a, E, Ty, Ix, SN, SE>,
    graph: &'b Csr<N, E, Ty, Ix>,
}

impl<'a, 'b, N, E, Ty, Ix, SN, SE> VertexDissection<'a, 'b, N, E, Ty, Ix, SN, SE>
where
    Ty: EdgeType,
    Ix: IndexType,
    SN: Set<Ix>,
    SE: Set<usize>,
    for<'x> &'x SN: SetRef<Ix>,
    for<'x> &'x SE: SetRef<usize>,
{
    pub fn side(&self, node: Ix) -> Side {
        if self.dinic.bfs.visited.contains(&node) {
            Side::Source
        } else if self.dinic.source_set.contains(&node)
            || self
                .graph
                .neighbors(node)
                .any(|neighbor| self.dinic.bfs.visited.contains(&neighbor))
        {
            Side::Mid
        } else {
            Side::Sink
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inertial_flow_graph;
    use petgraph::graph::Graph;
    use petgraph::visit::Dfs;
    use petgraph::visit::{IntoNodeIdentifiers, NodeFiltered, Visitable};
    use petgraph::Undirected;
    use quickcheck::quickcheck;

    quickcheck! {
        fn dissects_graph(graph: Graph<(), (), Undirected, usize>) -> bool {
            let csr = inertial_flow_graph(&graph);

            if csr.node_count() < 2 {
                return true
            }

            let mut dinic: Dinic<_, _, _, FixedBitSet, FixedBitSet> = Dinic::default();

            let src_index = 0;
            let sink_index = csr.node_count() - 1;

            let initial_sides = (0..csr.node_count()).map(|i| (i,
                if i == src_index {
                    Side::Source
                } else if i == sink_index {
                    Side::Sink
                } else {
                    Side::Mid
                }
            ));

            let dissection = dinic.vertex_dissect(&csr, initial_sides);

            assert!(dissection.side(src_index) != Side::Sink);
            assert!(dissection.side(sink_index) != Side::Source);
            assert!((0..csr.node_count()).any(|node| dissection.side(node) != Side::Mid));
            assert_dissection(&csr, |node| dissection.side(node));

            true
        }
    }

    fn assert_dissection<G>(graph: G, side: impl Fn(G::NodeId) -> Side)
    where
        G: IntoNeighbors + Visitable + IntoNodeIdentifiers,
    {
        let graph = NodeFiltered::from_fn(&graph, |node| (side)(node) != Side::Mid);
        let mut dfs = Dfs::empty(&graph);

        for node in graph.node_identifiers() {
            let node_side = (side)(node);

            if node_side != Side::Mid {
                dfs.move_to(node);

                while let Some(node) = dfs.next(&graph) {
                    assert!((side)(node) == node_side);
                }

                dfs.reset(&graph);
            }
        }
    }
}
