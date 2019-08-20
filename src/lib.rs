use arrayvec::ArrayVec;
use fixedbitset::FixedBitSet;
use fxhash::FxHashSet;
use num_rational::Ratio;
use ordered_float::OrderedFloat;
use partition::partition;
use pdqselect::select_by_key;
use petgraph::csr::Csr;
use petgraph::graph::IndexType;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};
use petgraph::Directed;
use petgraph::EdgeType;
use rayon;
use std::cell::RefCell;
use thread_local::ThreadLocal;

mod dinic;

use dinic::{Dinic, Side};
use dinic::{Set as DinicSet, SetRef as DinicSetRef};

/// Creates a graph suitable for computing a ND order from an arbitrary graph
pub fn inertial_flow_graph<Ix>(
    graph: impl IntoEdgeReferences + NodeIndexable,
) -> Csr<(), (), Directed, Ix>
where
    Ix: IndexType,
{
    assert!(graph.node_bound() < <Ix as IndexType>::max().index());

    let mut edges: Vec<_> = graph
        .edge_references()
        .flat_map(|edge| {
            let a = Ix::new(graph.to_index(edge.source()));
            let b = Ix::new(graph.to_index(edge.target()));
            ArrayVec::from([(a, b), (b, a)])
        })
        .collect();

    edges.sort();
    edges.dedup();

    Csr::from_sorted_edges(&edges).expect("Edges should be sorted and deduplicated")
}

/// Computes a nested-dissection order using inertial flow given a graph, a ratio of terminals, and
/// a function mapping nodes to 2D positions.
///
/// `graph` can be directed, but each edge must have a corresponding opposing edge. See
/// `inertial_flow_graph` for creating a suitable graph from an arbitrary graph.
///
/// `terminals` determines the ratio of source and sink nodes when computing the  intertial flow.
/// This must be less than or equal to 1/2.
///
/// `positions` must be an iterator of 2D positions, of same length as nodes in the graph. The first
/// positions in the iterator corresponds to the first node in the graph.
///
/// The function returns the node indices in nested-dissection order.
pub fn inertial_flow_order<N, E, Ty, Ix>(
    graph: &Csr<N, E, Ty, Ix>,
    terminals: Ratio<usize>,
    positions: impl IntoIterator<Item = (f32, f32)>,
) -> impl Iterator<Item = Ix>
where
    N: Sync,
    E: Sync,
    Ty: EdgeType + Sync + Send,
    Ix: IndexType + Sync + Send,
{
    inertial_flow_order_with_options(graph, terminals, 20000, positions)
}

#[cfg(feature = "bench")]
pub fn inertial_flow_order_with_bitset_cutoff<N, E, Ty, Ix>(
    graph: &Csr<N, E, Ty, Ix>,
    terminals: Ratio<usize>,
    bitset_cutoff: usize,
    positions: impl IntoIterator<Item = (f32, f32)>,
) -> impl Iterator<Item = Ix>
where
    N: Sync,
    E: Sync,
    Ty: EdgeType + Sync + Send,
    Ix: IndexType + Sync + Send,
{
    inertial_flow_order_with_options(graph, terminals, bitset_cutoff, positions)
}

fn inertial_flow_order_with_options<N, E, Ty, Ix>(
    graph: &Csr<N, E, Ty, Ix>,
    terminals: Ratio<usize>,
    bitset_cutoff: usize,
    positions: impl IntoIterator<Item = (f32, f32)>,
) -> impl Iterator<Item = Ix>
where
    N: Sync,
    E: Sync,
    Ty: EdgeType + Sync + Send,
    Ix: IndexType + Sync + Send,
{
    assert!(
        terminals <= Ratio::new(1, 2),
        "Terminals must less than or equal to 1/2"
    );

    assert!(
        graph.edge_references().all(|e1| graph
            .edges(e1.target())
            .any(|e2| e2.target() == e1.source())),
        "Every edge must have an opposing edge"
    );

    let mut nodes: Vec<_> = positions
        .into_iter()
        .enumerate()
        .map(|(i, (x, y))| (Ix::new(i), x, y, Side::Mid))
        .collect();

    assert!(
        nodes.len() == graph.node_count(),
        "Length of positions must match nodes in graph"
    );

    let nd = Nd {
        small: ThreadLocal::new(),
        large: ThreadLocal::new(),
    };

    nd.nested_dissect(graph, terminals, bitset_cutoff, &mut nodes);

    drop(nd);

    nodes.into_iter().map(|(i, ..)| i)
}

struct Nd<'a, E: Sync, Ty: EdgeType + Sync + Send, Ix: IndexType + Sync + Send> {
    small: ThreadLocal<RefCell<Dinic<'a, E, Ty, Ix, FxHashSet<Ix>, FxHashSet<usize>>>>,
    large: ThreadLocal<RefCell<Dinic<'a, E, Ty, Ix, FixedBitSet, FixedBitSet>>>,
}

impl<'a, E: Sync, Ty: EdgeType + Sync + Send, Ix: IndexType + Sync + Send> Nd<'a, E, Ty, Ix> {
    fn nested_dissect<N: Sync>(
        &self,
        graph: &'a Csr<N, E, Ty, Ix>,
        terminals: Ratio<usize>,
        bitset_cutoff: usize,
        nodes: &mut [(Ix, f32, f32, Side)],
    ) {
        if nodes.len() < 3 {
            return;
        }

        let (lo, hi) = if nodes.len() > bitset_cutoff {
            let dinic = &mut *self.large.get_default().borrow_mut();
            Self::dissect(dinic, graph, terminals, nodes)
        } else {
            let dinic = &mut *self.small.get_default().borrow_mut();
            Self::dissect(dinic, graph, terminals, nodes)
        };

        let (lo_slice, nodes) = nodes.split_at_mut(lo);
        let (hi_slice, sep_slice) = nodes.split_at_mut(hi - lo);

        rayon::join(
            || self.nested_dissect(graph, terminals, bitset_cutoff, lo_slice),
            || {
                rayon::join(
                    || self.nested_dissect(graph, terminals, bitset_cutoff, sep_slice),
                    || self.nested_dissect(graph, terminals, bitset_cutoff, hi_slice),
                )
            },
        );
    }

    fn dissect<N, SN, SE>(
        dinic: &mut Dinic<'a, E, Ty, Ix, SN, SE>,
        graph: &'a Csr<N, E, Ty, Ix>,
        terminals: Ratio<usize>,
        nodes: &mut [(Ix, f32, f32, Side)],
    ) -> (usize, usize)
    where
        SN: DinicSet<Ix>,
        SE: DinicSet<usize>,
        for<'x> &'x SN: DinicSetRef<Ix>,
        for<'x> &'x SE: DinicSetRef<usize>,
    {
        let n_nodes = nodes.len();

        let mut min_width = std::usize::MAX;

        for &(x_coeff, y_coeff) in &[(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0)] {
            let n_sources = std::cmp::max((terminals * n_nodes).to_integer(), 1);

            select_by_key(nodes, n_sources, |&(i, x, y, _)| {
                (i, OrderedFloat::from(x * x_coeff + y * y_coeff))
            });
            select_by_key(&mut nodes[n_sources..], n_sources, |&(i, x, y, _)| {
                (i, OrderedFloat::from(x * x_coeff + y * y_coeff))
            });

            let sources = n_sources;
            let sinks = n_sources * 2;

            let thing = nodes.iter().enumerate().map(|(i, &(node, ..))| {
                (
                    node,
                    if i < sources {
                        Side::Source
                    } else if i < sinks {
                        Side::Sink
                    } else {
                        Side::Mid
                    },
                )
            });

            let dissection = dinic.vertex_dissect(graph, thing);

            let width = nodes
                .iter()
                .filter(|&(node, ..)| dissection.side(*node) == Side::Mid)
                .count();

            if width < min_width {
                min_width = width;
                nodes
                    .iter_mut()
                    .for_each(|(i, .., sep)| *sep = dissection.side(*i));
            }
        }

        let (sources, nodes) = partition(nodes, |&(.., x)| x == Side::Source);
        let (sink, _) = partition(nodes, |&(.., x)| x == Side::Sink);

        let min_lo = sources.len();
        let min_hi = sources.len() + sink.len();

        (min_lo, min_hi)
    }
}
