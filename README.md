# Random tree generators

This repository contains implementations of some algorithms for generating a random tree on `n` vertices and comparison of these algorithms. All algorithms sample labeled trees from uniform distribution, that is, each tree appears with probability `n^(-n+2)`.

I did that comparison when I was preparing for a reading group at [CombGeo Lab](https://combgeo.org/en/), and I wanted to approximately compare relative performance of these algorithms. That said, I didn't put ultimate efforts in optimizations, and some ~10% improvements are still possible. I didn't intend this code to be reusable as well (anyway, all these algorithms are quite short and simple).


## Algorithms

There are three types of algorithms: using Prüfer codes, using random mappings and using random walks.


### Prüfer codes

These algorithms take a random Prüfer code from `[n]^(n-2)` and decode it to a tree. Decoding can be done using a straightforward algorithm with `O(n log n)` complexity or using any fast algorithm with `O(n)` complexity, e.g. [PS07] or [WWW09].

### Random mappings

That algorithms takes a random mapping `f: [n] -> [n]` (or, equivalently, a directed graph with out-degree 1), and transforms it to a tree (such that each tree has exactly `n^2` pre-images). That algorithm is described in [H20].

### Random walks

These algorithms generate a random walk on a graph `K_n` and each time we visit a vertex for a first time, add corresponding edge to the tree [Bro89, Ald90]. It is easy to see that on average we need to do at least `Omega(n log n)` to visit all vertices. Still, with some tricks (like skipping steps inside of visited vertices) it is possible to implement that algorithm in a linear time.

In [ISZ19, Section 5] there is an elegant equivalent reformulation, which gives the simplest and one of the fastest algorithms for tree generation.

Note that walk-based algorithms use twice as much entropy as previous algorithms.

## Benchmarks

To run benchmarks, run `cargo bench`. Below are results for `n = 10^6` on my laptop:

- sampling `n` values from `{1, ..., n}`: 9 ms
- naive Prüfer decode: 93 ms
- fast Prüfer decode [PS07]: 28 ms
- fast Prüfer decode [WWW09]: 34 ms
- random mapping [H20]: 61 ms
- random walk [Bro89, Ald90]: 80 ms
- random walk reformulated [ISZ19]: 33 ms


## References

[Ald90] D. J. Aldous, The random walk construction of uniform spanning trees and uniform labelled trees, SIAM J. Discrete Math. 3, no. 4, 450-465, 1990.

[Bro89] A. Broder, Generating random spanning trees, Proceedings of the 30th Annual Symposium on Foundations of Computer Science (USA), SFCS'89, IEEE Computer Society, p. 442-447, 1989.

[H20] S. Heilman, Tree/Endofunction Bijections and Concentration Inequalities, preprint, arXiv:2006.06724, 2020.

[ISZ19] M. Isaev, A. Southwell, and M. Zhukovskii, Distribution of tree parameters by martingale approach, preprint, arXiv:1912.09838, 2019.

[PS07] T. Paulden, and D. K. Smith, Developing new locality results for the Prüfer Code using a remarkable linear-time decoding algorithm, The Electronic Journal of Combinatorics, R55-R55, 2007.

[WWW09] X. Wang, L. Wang, and Y. Wu, An Optimal Algorithm for Prufer Codes. JSEA, 2(2), 111-115, 2009.