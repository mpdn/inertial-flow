use inertial_flow::inertial_flow_order_with_bitset_cutoff;
use criterion::{black_box, Criterion};
use fxhash::FxHashMap;
use num_rational::Ratio;
use osmpbfreader::{OsmObj, OsmPbfReader};
use petgraph::{csr::Csr, Directed};
use std::fs::File;

fn load_osm(file: &str) -> (Csr<(), (), Directed, u32>, Vec<(f32, f32)>) {
    let f = File::open(file).unwrap();

    let mut pbf = OsmPbfReader::new(f);

    let mut nodes: Vec<_> = Vec::new();
    let mut edges = Vec::new();

    let coord_scale = 10000000.0;

    for obj in pbf.par_iter() {
        match obj.unwrap() {
            OsmObj::Way(way) => {
                if way.tags.contains_key("highway") {
                    edges.extend(way.nodes.windows(2).map(|w| (w[0], w[1])));
                    edges.extend(way.nodes.windows(2).map(|w| (w[1], w[0])));
                }
            }
            OsmObj::Node(node) => {
                nodes.push((
                    node.id,
                    (
                        (node.decimicro_lat as f64 / coord_scale) as f32,
                        (node.decimicro_lon as f64 / coord_scale) as f32,
                    ),
                ));
            }
            _ => {}
        }
    }

    let node_index_by_id: FxHashMap<_, _> = nodes
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, (id, point))| (id, (i as u32, point)))
        .collect();

    let mut edges: Vec<_> = edges
        .into_iter()
        .filter(|(a, b)| a != b)
        .map(|(n1, n2)| (node_index_by_id[&n1].0, node_index_by_id[&n2].0))
        .collect();

    drop(node_index_by_id);

    edges.sort();
    edges.dedup();

    let pos = nodes.into_iter().map(|(_, xy)| xy).collect();

    (Csr::from_sorted_edges(&edges).unwrap(), pos)
}

fn main() {
    Criterion::default()
        .sample_size(10)
        .bench_function_over_inputs(
            "osm_bench",
            |b, &&size| {
                let (csr, pos) = load_osm("example-latest.osm.pbf");

                b.iter(|| {
                    let _ = black_box(inertial_flow_order_with_bitset_cutoff(
                        black_box(&csr),
                        Ratio::new(1, 5),
                        size,
                        pos.iter().cloned(),
                    ));
                })
            },
            &[8192, 16384, 32768],
        );
}
