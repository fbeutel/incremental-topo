#[macro_use]
extern crate criterion;
extern crate incremental_topo;
extern crate rand;

use criterion::Criterion;
use incremental_topo::{IncrementalTopo, TopoKey};

const DEFAULT_DENSITY: f32 = 0.1;
const DEFAULT_SIZE: usize = 1000;

fn generate_random_dag(size: usize, density: f32) -> (Vec<TopoKey>, IncrementalTopo) {
    use rand::distributions::{Bernoulli, Distribution};
    assert!(0.0 < density && density <= 1.0);
    let mut rng = rand::thread_rng();
    let dist = Bernoulli::new(density.into());
    let mut topo = IncrementalTopo::new();

    let mut keys = vec![];
    for _node in 0..size {
        keys.push(topo.add_node());
    }

    for i in 0..size {
        for j in 0..size {
            if i != j && dist.sample(&mut rng) {
                let i_key = unsafe { keys.get_unchecked(i) };
                let j_key = unsafe { keys.get_unchecked(j) };
                // Ignore failures
                let _ = topo.add_dependency(*i_key, *j_key);
            }
        }
    }

    (keys, topo)
}

fn criterion_benchmark(c: &mut Criterion) {
    use rand::distributions::{Distribution, Uniform};
    c.bench_function_over_inputs(
        "single_insert_random_graph_different_density",
        |b, &&density| {
            let (keys, dag) = generate_random_dag(750, density);
            let mut rng = rand::thread_rng();
            let dist = Uniform::new(0, 750);

            let i = dist.sample(&mut rng);
            let j = dist.sample(&mut rng);

            let i_key = unsafe { keys.get_unchecked(i) };
            let j_key = unsafe { keys.get_unchecked(j) };

            b.iter(|| {
                let mut dag = dag.clone();
                let _ = dag.add_dependency(*i_key, *j_key);
            });
        },
        &[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    );

    c.bench_function("clone", |b| {
        let (_keys, dag) = generate_random_dag(DEFAULT_SIZE, DEFAULT_DENSITY);

        b.iter(|| dag.clone());
    });

    c.bench_function("contains_node", |b| {
        let (keys, dag) = generate_random_dag(DEFAULT_SIZE, DEFAULT_DENSITY);

        let mid_key = unsafe { keys.get_unchecked(500) };

        b.iter(|| dag.clone().contains_node(*mid_key));
    });

    c.bench_function("delete_node", |b| {
        let (keys, dag) = generate_random_dag(DEFAULT_SIZE, DEFAULT_DENSITY);

        let mid_key = unsafe { keys.get_unchecked(500) };

        b.iter(|| dag.clone().delete_node(*mid_key));
    });

    c.bench_function_over_inputs(
        "contains_dependency",
        |b, &&density| {
            let (keys, dag) = generate_random_dag(DEFAULT_SIZE, density);
            let mut rng = rand::thread_rng();
            let dist = Uniform::new(0, DEFAULT_SIZE);

            b.iter(|| {
                let i = dist.sample(&mut rng);
                let j = dist.sample(&mut rng);

                let i_key = unsafe { keys.get_unchecked(i) };
                let j_key = unsafe { keys.get_unchecked(j) };
                let _ = dag.contains_dependency(*i_key, *j_key);
            });
        },
        &[0.02, 0.04, 0.06, 0.08, 0.1],
    );

    c.bench_function("contains_transitive_dependency", |b| {
        let (keys, dag) = generate_random_dag(DEFAULT_SIZE, DEFAULT_DENSITY);
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(0, DEFAULT_SIZE);

        b.iter(|| {
            let i = dist.sample(&mut rng);
            let j = dist.sample(&mut rng);

            let i_key = unsafe { keys.get_unchecked(i) };
            let j_key = unsafe { keys.get_unchecked(j) };
            let _ = dag.contains_transitive_dependency(*i_key, *j_key);
        });
    });

    c.bench_function("delete_dependency", |b| {
        let (keys, dag) = generate_random_dag(DEFAULT_SIZE, DEFAULT_DENSITY);
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(0, DEFAULT_SIZE);

        b.iter(|| {
            let i = dist.sample(&mut rng);
            let j = dist.sample(&mut rng);

            let i_key = unsafe { keys.get_unchecked(i) };
            let j_key = unsafe { keys.get_unchecked(j) };
            let _ = dag.clone().delete_dependency(*i_key, *j_key);
        });
    });

    c.bench_function("descendants_unsorted", |b| {
        let (keys, dag) = generate_random_dag(DEFAULT_SIZE, DEFAULT_DENSITY);
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(0, DEFAULT_SIZE);

        b.iter(|| {
            let i = dist.sample(&mut rng);
            let i_key = unsafe { keys.get_unchecked(i) };
            let _ = dag
                .descendants_unsorted(*i_key)
                .unwrap()
                .collect::<Vec<_>>();
        });
    });

    c.bench_function("descendants", |b| {
        let (keys, dag) = generate_random_dag(DEFAULT_SIZE, DEFAULT_DENSITY);
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(0, DEFAULT_SIZE);

        b.iter(|| {
            let i = dist.sample(&mut rng);
            let i_key = unsafe { keys.get_unchecked(i) };
            let _ = dag.descendants(*i_key).unwrap().collect::<Vec<_>>();
        });
    });

    c.bench_function("topo_cmp", |b| {
        let (keys, dag) = generate_random_dag(DEFAULT_SIZE, DEFAULT_DENSITY);
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(0, DEFAULT_SIZE);

        b.iter(|| {
            let i = dist.sample(&mut rng);
            let j = dist.sample(&mut rng);

            let i_key = unsafe { keys.get_unchecked(i) };
            let j_key = unsafe { keys.get_unchecked(j) };
            let _ = dag.topo_cmp(*i_key, *j_key).unwrap();
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
