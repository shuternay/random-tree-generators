use criterion::{criterion_group, criterion_main, Criterion};

use rand::{thread_rng, Rng};
use random_tree::{
    sample_random_tree_fast_prufer_ps07, sample_random_tree_fast_prufer_www09,
    sample_random_tree_mapping, sample_random_tree_naive_prufer, sample_random_tree_walk,
    sample_random_tree_walk_reformulated,
};

fn random_sequence(c: &mut Criterion) {
    c.bench_function("sequence, n=1000_000", |b| {
        let n = 1000_000;
        b.iter(|| {
            (0..n)
                .map(|_| thread_rng().gen_range(0, n))
                .collect::<Vec<usize>>()
        })
    });
}

fn random_tree_mapping(c: &mut Criterion) {
    // c.bench_function("mapping, n=10", |b| {
    //     b.iter(|| sample_random_tree_mapping(10))
    // });

    c.bench_function("mapping, n=1000_000", |b| {
        b.iter(|| sample_random_tree_mapping(1000_000))
    });
}

fn random_tree_fast_prufer_ps07(c: &mut Criterion) {
    c.bench_function("fast_prufer [PS07], n=1000_000", |b| {
        b.iter(|| sample_random_tree_fast_prufer_ps07(1000_000))
    });
}

fn random_tree_fast_prufer_www09(c: &mut Criterion) {
    c.bench_function("fast_prufer [WWW09], n=1000_000", |b| {
        b.iter(|| sample_random_tree_fast_prufer_www09(1000_000))
    });
}

fn random_tree_naive_prufer(c: &mut Criterion) {
    c.bench_function("naive_prufer, n=1000_000", |b| {
        b.iter(|| sample_random_tree_naive_prufer(1000_000))
    });
}

fn random_tree_walk(c: &mut Criterion) {
    c.bench_function("random walk [Bro89, Ald90], n=1000_000", |b| {
        b.iter(|| sample_random_tree_walk(1000_000))
    });
}

fn random_tree_walk_reformulated(c: &mut Criterion) {
    c.bench_function(
        "random walk reformulated [Ald90, Algorithm 2], n=1000_000",
        |b| b.iter(|| sample_random_tree_walk_reformulated(1000_000)),
    );
}

criterion_group!(
    benches,
    random_sequence,
    random_tree_mapping,
    random_tree_fast_prufer_ps07,
    random_tree_fast_prufer_www09,
    random_tree_naive_prufer,
    random_tree_walk,
    random_tree_walk_reformulated,
);
criterion_main!(benches);
