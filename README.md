# STA 663 (Spring 2020) Syllabus

## Learning objectives

- Develop fluency in Python for scientific computing
- Explain how common statistical algorithms work
- Construct models using probabilistic programming
- Implement, test, optimize, and package a statistical algorithm

Note: The syllabus is aspirational and is likely to be adjusted over the semester depending on how fast we are able to cover the material.

## Administration

### Office hours 

- Cliburn: Thursday 4-5 PM at 11078 Hock Suite 1102
- Zixi Wang:
- Chudi Zhong: 

### Grading

- Homework 40%
- Midterm 1 15%
- Midterm 2 15%
- Project 30%

### Point range for letter grade

- A 94 - 100
- B 85 - 93
- C 70 - 84
- D Below 70

Grades will be based on rounded scores.

### Module 1: Develop fluency in Python for scientific computing

#### 1. Jupyter and Python

- Introduction to Jupyter
- Using Markdown
- Magic functions
- REPL
- Data types
- Operators
- Collections
- Functions and methods
- Control flow
- Packages and namespace
- Coding style
- Understanding error messages
- Getting help
- Saving and exporting Jupyter notebooks

#### 2. Text 

- The string package
- String methods
- Regular expressions
- Loading and saving text files
- Context managers
- Dealing with encoding errors

#### 3. Numerics

- Issues with floating point numbers
- The `math` package
- Constructing `numpy` arrays
- Indexing
- Splitting and merging arrays
- Universal functions - transforms and reductions
- Broadcasting rules
- Masking
- Sparse matrices with `scipy.sparse`

#### 4. Data manipulation

- Series and DataFrames in `pandas`
- Creating, loading and saving DataFrames
- Basic information
- Indexing
- Method chaining
- Selecting rows and columns
- Transformations
- Aggregate functions
- Split-apply-combine
- Window functions
- Hierarchical indexing

#### 5. Graphics

- Grammar of graphics
- Graphics from the group up with `matplotlib`
- Statistical visualizations with `seaborn`

#### 6. Functional programming in Python

- Writing a custom function
- Pure functions
- Anonymous functions
- Lazy evaluation
- Higher-order functions
- Decorators
- Partial application
- Using operator
- Using `functional`
- Using `itertools`
- Pipelines with `toolz`

### Midterm 1

### Module 2: Explain how common statistical algorithms work

#### 7. Data structures, algorithms and complexity

- Sequence and mapping containers
- Using collections
- Sorting
- Priority queues
- Working with recursive algorithms
- Tabling and dynamic programing
- Time and space complexity
- Measuring time
- Measuring space

#### 8. Solving linear equations

- Solving Ax = bAx=b
- Gaussian elimination and LR decomposition
- Symmetric matrices and Cholesky decomposition
- Geometry of the normal equations
- Gradient descent to solve linear equations
- Using `scipy.linalg`

#### 9. Singular Value Decomposition

- Change of basis
- Spectral decomposition
- Geometry of spectral decomposition
- The four fundamental subspaces of linear algebra
- The SVD
- Geometry of spectral decomposition
- SVD and low rank approximation
- Using `scipy.linalg`

#### 10. Optimization I

- Root finding
- Univariate optimization
- Geometry and calculus of optimization
- Gradient descent
- Batch, mini-batch and stochastic variants
- Improving gradient descent
- Root finding and univariate optimization with `scipy.optim`

#### 11. Optimization II

- Nelder-Mead (Zeroth order method)
- Line search methods
- Trust region methods
- IRLS
- Lagrange multipliers, KKT and constrained optimization
- Multivariate optimization with `scipy.optim`

#### 12. Dimension reduction

- Matrix factorization - PCA and SVD, MMF
- Optimization methods - MDS and t-SNE
- Using `sklearn.decomposition` and `sklearn.manifold`
  
#### 13. Interpolation

- Polynomial
- Spline
- Gaussian process
- Using `scipy.interpolate`

#### 14. Clustering

- Partitioning (k-means)
- Hierarchical (agglomerative Hierarchical Clustering)
- Density based (dbscan, mean-shift)
- Model based (GMM)
- Self-organizing maps
- Cluster initialization
- Cluster evaluation
- Cluster alignment (Munkres)
- Using `skearn.cluster`

### Midterm 2

### Module 3: Construct models using probabilistic programming

#### 15. Probability and random processes

- Working with probability distributions
- Using `random`
- Using `np.random`
- Using `scipy.statistics`
- Simulations

#### 16. Monte Carlo methods

- Sampling from data
- Bootstrap
- Permutation resampling
- Sampling from distributions
- Rejection sampling
- Importance sampling
- Monte Carlo integration
- Density estimation

#### 17. Gibbs and MH

- Bayes theorem and integration
- Numerical integration (quadrature)
- MCMC concepts
- Makrov chains
- Metropolis-Hastings random walk
- Gibbs sampler

#### 18. Hamiltonian Monte Carlo

- Hamiltonian systems
- Integration of Hamiltonian system dynamics
- Energy and probability distributions
- HMC
- NUTS

#### 19. Probabilistic programming

- Domain-specific languages
- Multi-level Bayesian models
- Using daft to draw plate diagrams
- Using `pymc3`
- Using `pystan`

##### 20. Using TesnorFlow probability (Tfp)

- TensorFlow basics
- Distributions and transformations
- Building probabilistic models with `Tfp`

#### 21. Probabilistic deep learning

- Loss functions and likelihood
- Optimization for deep learning
- Back-propagation
- Bayes by back-propagation
- Deep generative models

### Module 4: Implement, optimize, and package a statistical algorithm

#### 22. Python modules

- Organization of a module
- Writing the setup script
- The Python Package Index
- Package managers

#### 23. Containers

- Code optimization I

- Vectorization
- JIT compilation with `numba`
- AOT compilation with `cython`

#### 24. Code optimization II

- Interpreters and compilers
- Review of C++
- Wrapping C++ functions with `pybind11`

#### 25. Parallel programming

- Parallel, concurrent, asynchronous, distributed
- Threads and processes
- Shared memory programming pitfalls: deadlock and race conditions
- Embarrassingly parallel programs with `concurrent.futures` and `multiprocessing`
- Map-reduce
- Master-worker
- Using `ipyparallel` for interactive parallelization

### Final Project 
