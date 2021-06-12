use std::cmp::Reverse;
use std::collections::BinaryHeap;

use bit_set::BitSet;
use rand::prelude::*;

/// Sorts the edges list. This method is used to compare trees for equivalence
/// (but not for isomorphism)
pub fn normalise_tree(mut tree: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    for edge in tree.iter_mut() {
        if edge.0 > edge.1 {
            *edge = (edge.1, edge.0);
        }
    }
    tree.sort();
    tree
}

/// Decodes a Prufer code in a straightforward naive way.
///
/// Complexity: O(n log n)
fn naive_prufer_decode(n: usize, code: &Vec<usize>) -> Vec<(usize, usize)> {
    assert_eq!(code.len(), n - 2);

    let mut degrees = vec![1; n];
    for &v in code {
        degrees[v] += 1;
    }

    let mut leafs: BinaryHeap<Reverse<usize>> = degrees
        .iter()
        .enumerate()
        .filter_map(|(v, &d)| if d == 1 { Some(Reverse(v)) } else { None })
        .collect();

    let mut tree = Vec::new();
    for &current_neighbour in code {
        let minimal_leaf = leafs.pop().unwrap().0;
        tree.push((current_neighbour, minimal_leaf));

        degrees[current_neighbour] -= 1;
        if degrees[current_neighbour] == 1 {
            leafs.push(Reverse(current_neighbour));
        }
    }
    assert_eq!(leafs.len(), 2);
    tree.push((leafs.peek().unwrap().0, n - 1));

    tree
}

/// Decodes a Prufer code in a fast way, as described in [PS07].
///
/// Complexity: O(n)
fn fast_prufer_decode_ps07(n: usize, code: &Vec<usize>) -> Vec<(usize, usize)> {
    assert_eq!(code.len(), n - 2);

    let mut is_loose = vec![true; n];
    is_loose[n - 1] = false;
    let mut largest_loose = n - 1;

    let mut tree = Vec::new();
    for j in 2..n {
        let v_j = if is_loose[code[n - j - 1]] {
            code[n - j - 1]
        } else {
            largest_loose
        };
        if j != 2 {
            tree.push((code[n - j], v_j));
        } else {
            tree.push((n - 1, v_j));
        };
        is_loose[v_j] = false;
        while !is_loose[largest_loose] {
            largest_loose -= 1;
        }
    }
    tree.push((largest_loose, code[0]));

    assert_eq!(tree.len(), n - 1);
    tree
}

/// Decodes a Prufer code in a fast way, as described in [WWW09].
///
/// Complexity: O(n)
fn fast_prufer_decode_www09(n: usize, code: &Vec<usize>) -> Vec<(usize, usize)> {
    assert_eq!(code.len(), n - 2);

    let mut degrees = vec![1; n];
    for &v in code {
        degrees[v] += 1;
    }

    let mut current_min_position = (0..n).filter(|&v| degrees[v] == 1).next().unwrap();
    let mut minimal_leaf = current_min_position;

    let mut tree = Vec::new();
    for &current_neighbour in code {
        tree.push((current_neighbour, minimal_leaf));

        degrees[current_neighbour] -= 1;

        if current_neighbour < current_min_position && degrees[current_neighbour] == 1 {
            minimal_leaf = current_neighbour;
        } else {
            current_min_position = (current_min_position + 1..n)
                .filter(|&v| degrees[v] == 1)
                .next()
                .unwrap();
            minimal_leaf = current_min_position;
        }
    }
    tree.push((minimal_leaf, n - 1));

    tree
}

/// Encodes a tree with a Prufer code in a straightforward naive way.
///
/// Complexity: O(n log n)
fn prufer_encode(n: usize, tree: &Vec<(usize, usize)>) -> Vec<usize> {
    let mut neighbours = vec![BitSet::new(); n];
    for &(from, to) in tree {
        neighbours[from].insert(to);
        neighbours[to].insert(from);
    }

    let mut leafs: BinaryHeap<Reverse<usize>> = neighbours
        .iter()
        .enumerate()
        .filter_map(|(v, nbrs)| {
            if nbrs.len() == 1 {
                Some(Reverse(v))
            } else {
                None
            }
        })
        .collect();

    let mut code = Vec::new();
    while code.len() < n - 2 {
        let minimal_leaf = leafs.pop().unwrap().0;
        let neighbour = neighbours[minimal_leaf].iter().next().unwrap();
        code.push(neighbour);
        neighbours[neighbour].remove(minimal_leaf);
        if neighbours[neighbour].len() == 1 {
            leafs.push(Reverse(neighbour));
        }
    }
    assert_eq!(leafs.len(), 2);

    code
}

/// Generates a random tree using naive Prufer decoding.
///
/// Complexity: O(n log n)
pub fn sample_random_tree_naive_prufer(n: usize) -> Vec<(usize, usize)> {
    naive_prufer_decode(
        n,
        &(0..n - 2).map(|_| thread_rng().gen_range(0, n)).collect(),
    )
}

/// Generates a random tree using fast Prufer decoding [PS07].
///
/// Complexity: O(n)
pub fn sample_random_tree_fast_prufer_ps07(n: usize) -> Vec<(usize, usize)> {
    fast_prufer_decode_ps07(
        n,
        &(0..n - 2).map(|_| thread_rng().gen_range(0, n)).collect(),
    )
}

/// Generates a random tree using fast Prufer decoding [WW09].
///
/// Complexity: O(n)
pub fn sample_random_tree_fast_prufer_www09(n: usize) -> Vec<(usize, usize)> {
    fast_prufer_decode_www09(
        n,
        &(0..n - 2).map(|_| thread_rng().gen_range(0, n)).collect(),
    )
}

/// Generates a random tree using random walk a.k.a. Aldous-Broder algorithm [Bro89, Ald90].
///
/// Complexity: O(n)
pub fn sample_random_tree_walk(n: usize) -> Vec<(usize, usize)> {
    let mut is_visited = vec![false; n];
    is_visited[0] = true;
    let mut visited_vertices = vec![0 as usize];
    let mut not_visited_vertices: Vec<usize> = (1..n).collect();

    let mut tree = Vec::new();
    let mut current_vertex = 0 as usize;
    while not_visited_vertices.len() > 0 {
        // here we do a little trick to sometimes save a gen_range() call
        let mut next_vertex_index = thread_rng().gen_range(0, n);
        if next_vertex_index >= not_visited_vertices.len() {
            current_vertex = visited_vertices[next_vertex_index - not_visited_vertices.len()];
            next_vertex_index = thread_rng().gen_range(0, not_visited_vertices.len());
        }
        // next_vertex_index is in [0, |not visited vertices|) here
        let next_vertex = not_visited_vertices[next_vertex_index];
        tree.push((current_vertex, next_vertex));
        current_vertex = next_vertex;
        visited_vertices.push(next_vertex);
        let last_index = not_visited_vertices.len() - 1;
        not_visited_vertices.swap(next_vertex_index, last_index);
        not_visited_vertices.pop();
    }
    assert_eq!(tree.len(), n - 1);
    tree
}

/// Generates a random tree using a reformulation of Aldous-Broder algorithm [Bro89, Ald90].
/// from [Ald90, Algorithm 2].
///
/// Complexity: O(n)
pub fn sample_random_tree_walk_reformulated(n: usize) -> Vec<(usize, usize)> {
    let labels_mapping = {
        let mut labels_mapping: Vec<_> = (0..n).collect();
        labels_mapping.shuffle(&mut thread_rng());
        labels_mapping
    };

    let mut tree = Vec::new();
    for i in 0..n - 1 {
        let neighbour = thread_rng().gen_range(0, n).min(i);
        tree.push((labels_mapping[i + 1], labels_mapping[neighbour]));
    }

    assert_eq!(tree.len(), n - 1);
    tree
}

/// For an oriented graph with out-degree 1 finds its loops and for each loop returns
/// the minimal vertex and a vertex before it. This method is a part of Heilman algorithm [H20].
///
/// Complexity: O(n)
fn get_loops(n: usize, mapping: &Vec<usize>) -> Vec<(usize, usize)> {
    #[derive(Copy, Clone, PartialEq)]
    enum Status {
        NotVisited,
        InProgress,
        Visited,
    }
    let mut status = vec![Status::NotVisited; n];
    let mut loops = Vec::new();

    for current_start in 0..n {
        let mut current_vertex = current_start;
        while status[current_vertex] == Status::NotVisited {
            status[current_vertex] = Status::InProgress;
            current_vertex = mapping[current_vertex];
        }

        // found a loop
        if status[current_vertex] == Status::InProgress {
            let mut loop_vertices = Vec::new();
            let mut argmin = 0;
            while status[current_vertex] == Status::InProgress {
                loop_vertices.push(current_vertex);
                if current_vertex < loop_vertices[argmin] {
                    argmin = loop_vertices.len() - 1;
                }

                status[current_vertex] = Status::Visited;
                current_vertex = mapping[current_vertex];
            }

            loops.push((
                loop_vertices[argmin],
                loop_vertices[(argmin + loop_vertices.len() - 1) % loop_vertices.len()],
            ));
        }

        let mut current_vertex = current_start;
        while status[current_vertex] == Status::InProgress {
            status[current_vertex] = Status::Visited;
            current_vertex = mapping[current_vertex];
        }
    }

    loops
}

/// Generates a random tree using Heilman algorithm [H20].
///
/// Complexity: O(n)
pub fn sample_random_tree_mapping(n: usize) -> Vec<(usize, usize)> {
    let mapping: Vec<usize> = (0..n).map(|_| thread_rng().gen_range(0, n)).collect();

    let mut loops = get_loops(n, &mapping);
    loops.sort_by_key(|&x| Reverse(x));

    let mut removed_edges = vec![None; n];
    for &loop_ in &loops {
        removed_edges[loop_.1] = Some(loop_.0);
    }

    let mut tree = Vec::new();
    for (&loop_, &next_loop) in loops.iter().zip(loops.iter().skip(1)) {
        tree.push((loop_.1, next_loop.0));
    }
    for v in 0..n {
        if removed_edges[v] != Some(mapping[v]) {
            tree.push((v, mapping[v]));
        }
    }
    assert_eq!(tree.len(), n - 1);

    tree
}

#[cfg(test)]
mod tests {
    use crate::*;
    use std::collections::HashMap;

    #[test]
    fn test_naive_prufer() {
        let n = 6;
        let tree = vec![(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)];
        let code = prufer_encode(n, &tree);
        assert_eq!(code, vec![3, 3, 3, 4]);
        let decoded_tree = naive_prufer_decode(n, &code);

        let normalised_tree = normalise_tree(tree);
        assert_eq!(normalised_tree, normalise_tree(decoded_tree));
    }

    #[test]
    fn test_encode_decode_fast_prufer_ps07() {
        let n = 6;
        let tree = vec![(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)];
        let code = prufer_encode(n, &tree);
        assert_eq!(code, vec![3, 3, 3, 4]);
        let decoded_tree = fast_prufer_decode_ps07(n, &code);

        let normalised_tree = normalise_tree(tree);
        assert_eq!(normalised_tree, normalise_tree(decoded_tree));
    }

    #[test]
    fn test_encode_decode_fast_prufer_www09() {
        let n = 6;
        let tree = vec![(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)];
        let code = prufer_encode(n, &tree);
        assert_eq!(code, vec![3, 3, 3, 4]);
        let decoded_tree = fast_prufer_decode_www09(n, &code);

        let normalised_tree = normalise_tree(tree);
        assert_eq!(normalised_tree, normalise_tree(decoded_tree));
    }

    #[test]
    fn test_fast_decode_ps_07() {
        let n = 6;
        let code = vec![0, 2, 1, 3];
        let decoded_tree = fast_prufer_decode_ps07(n, &code);
        assert_eq!(
            vec![(0, 2), (0, 4), (1, 2), (1, 3), (3, 5)],
            normalise_tree(decoded_tree)
        );
    }

    #[test]
    fn test_fast_decode_ww09() {
        let n = 6;
        let code = vec![0, 2, 1, 3];
        let decoded_tree = fast_prufer_decode_www09(n, &code);
        assert_eq!(
            vec![(0, 2), (0, 4), (1, 2), (1, 3), (3, 5)],
            normalise_tree(decoded_tree)
        );
    }

    /// Takes a generator, generates `iters` trees of size `n` and checks that the number
    /// of trees of each type doesn't deviate much from its expectation.
    fn test_tree_generator(
        n: usize,
        iters: usize,
        tree_generator: impl Fn(usize) -> Vec<(usize, usize)>,
    ) {
        let mut trees_count: HashMap<Vec<(usize, usize)>, usize> = HashMap::new();
        for _ in 0..iters {
            let tree = normalise_tree(tree_generator(n));
            *trees_count.entry(tree).or_default() += 1;
        }

        assert_eq!(trees_count.len(), n.pow((n - 2) as u32));
        let expected_count = iters / trees_count.len();
        println!("expected: {}", expected_count);
        for (_tree, &count) in &trees_count {
            println!(" count: {}", count);
            assert!(
                ((count as i32 - expected_count as i32).abs() as f32)
                    < 5.0 * (expected_count as f32).powf(0.5)
            );
        }
    }

    #[test]
    fn test_naive_prufer_generator() {
        test_tree_generator(4, 10_000_000, sample_random_tree_naive_prufer);
        test_tree_generator(6, 10_000_000, sample_random_tree_naive_prufer);
    }

    #[test]
    fn test_fast_prufer_ps07_generator() {
        test_tree_generator(4, 10_000_000, sample_random_tree_fast_prufer_ps07);
        test_tree_generator(6, 10_000_000, sample_random_tree_fast_prufer_ps07);
    }

    #[test]
    fn test_fast_prufer_www09_generator() {
        test_tree_generator(4, 10_000_000, sample_random_tree_fast_prufer_www09);
        test_tree_generator(6, 10_000_000, sample_random_tree_fast_prufer_www09);
    }

    #[test]
    fn test_walk_generator() {
        test_tree_generator(4, 10_000_000, sample_random_tree_walk);
        test_tree_generator(6, 10_000_000, sample_random_tree_walk);
    }

    #[test]
    fn test_walk_reformulation_generator() {
        test_tree_generator(4, 10_000_000, sample_random_tree_walk_reformulated);
        test_tree_generator(6, 10_000_000, sample_random_tree_walk_reformulated);
    }

    #[test]
    fn test_mapping_generator() {
        test_tree_generator(4, 10_000_000, sample_random_tree_mapping);
        test_tree_generator(6, 10_000_000, sample_random_tree_mapping);
    }
}
