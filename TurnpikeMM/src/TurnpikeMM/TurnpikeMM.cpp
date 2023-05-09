#include <random>
#include <Eigen/Core>
#include <boost/sort/sort.hpp>

#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using Vector = Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

namespace py = pybind11;

py::object eigen_to_scipy_sparse(const SparseMatrix& matrix) {
    py::module scipy_sparse = py::module::import("scipy.sparse");

    // Get the CSR representation of the Eigen sparse matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> csr_matrix = matrix;
    csr_matrix.makeCompressed();

    // Create the data, indices, and indptr arrays for the Scipy CSR matrix
    py::array_t<double> data(csr_matrix.nonZeros(), csr_matrix.valuePtr());
    py::array_t<int> indices(csr_matrix.nonZeros(), csr_matrix.innerIndexPtr());
    py::array_t<int> indptr(csr_matrix.outerSize() + 1, csr_matrix.outerIndexPtr());

    // Call the scipy.sparse.csr_matrix constructor
    py::object csr_class = scipy_sparse.attr("csr_matrix");
    return csr_class(py::make_tuple(data, indices, indptr), py::arg("shape") = py::make_tuple(csr_matrix.rows(), csr_matrix.cols()));
}

std::pair<int, int> get_row_col(int k) {
    /* Linearizes triangular matrix indices
     * Examples: k = 0 --> (0, 1), k = 1 --> (0, 2), k = 2 --> (1, 2),
     *           k = 3 --> (0, 3), k = 4 --> (1, 3), k = 5 --> (2, 3)
     */

    // Solve quadratic for (i, j) in k = j choose 2 + i
    int j = static_cast<int>(std::sqrt(2. * k + .25) - .5) + 1;
    int i = k - (j * j - j) / 2;

    return {i, j};
}

SparseMatrix build_incidence_matrix(const long n) {
    const long m{(n * n - n) / 2};

    // Signed incidence matrix for complete graph
    // Sparsity: 2m entries over n columns with n-1 non-zeros in each
    SparseMatrix Q(m, n);
    Q.reserve(Eigen::VectorXi::Constant(m, 2));

#pragma omp parallel for default(none) shared(Q, m)
    for (int k = 0; k < m; ++k) {
        const auto [i, j] = get_row_col(k);
        Q.insert(k, i) = -1;
        Q.insert(k, j) =  1;
    }

    Q.makeCompressed();

    return Q;
}

std::tuple<double, py::array_t<int>> online_matching(const py::array_t<double> &D_np,
                                                     const py::array_t<double> &Δ_np,
                                                     py::array_t<int> &I_np,
                                                     int idx) {
    // Get the underlying pointers to the NumPy array data
    int *I = I_np.mutable_data();
    const double *D = D_np.data();
    const double *Δ = Δ_np.data();

    // Compute matching…
    double cost = 0.;
    int m = static_cast<int>(D_np.size());
    int n = static_cast<int>(Δ_np.size());
    for (int k = 0; k < n; ++k) {
        int p = static_cast<int>(std::lower_bound(D, D + m, Δ[k]) - D);

        int i = std::max(0, p-1);
        int j = std::min(p, m-1);

        while (i > 0 && j < m && I[i] != -1 && I[j] != -1) {
            auto di = std::abs(Δ[k] - D[i]);
            auto dj = std::abs(Δ[k] - D[j]);
            int lt = di < dj;

            i -=     lt;
            j += 1 - lt;
        }

        if (I[j] != -1)
            while (i > 0 && I[i] != -1) i -= 1;

        if (I[i] != -1)
            while (j < m && I[j] != -1) j += 1;

        if (j == m) {
            I[i] = k + idx;
            cost += std::abs(Δ[k] - D[i]);
        } else {
            double di = std::abs(Δ[k] - D[i]);
            double dj = std::abs(D[j] - Δ[k]);

            if (I[i] == -1 && di < dj) {
                I[i] = k + idx;
                cost += di;
            } else {
                I[j] = k + idx;
                cost += dj;
            }
        }
    }

    return std::make_tuple(cost, I_np);
}

double match_distances(py::array_t<double> &dhat, const py::array_t<double> &d, int threads) {
    // Static permutation for likely case that this is called many times on the same size
    static std::vector<int> permutation;
    permutation.resize(d.size());

    // Fill with identity permutation
    std::iota(permutation.begin(), permutation.end(), 0);


    // Perform an argsort on the permutation array
    const double *dhat_arr = dhat.data();

    if (threads > 1) {
        boost::sort::sample_sort(permutation.begin(), permutation.end(), [&dhat_arr](int i, int j) {
            return dhat_arr[i] < dhat_arr[j];
        }, threads);
    } else {
        boost::sort::spinsort(permutation.begin(), permutation.end(), [&dhat_arr](int i, int j) {
            return dhat_arr[i] < dhat_arr[j];
        });
    }

    // Applies the inverse permutation Πᵀ
    double cost = 0.;
    const double *darr = d.data();
    double *dhat_write = dhat.mutable_data();
    for (std::size_t i = 0; i < permutation.size(); ++i) {
        cost += std::abs(dhat_write[permutation[i]] - darr[i]);
        dhat_write[permutation[i]] = darr[i];
    }

    return cost;
}

PYBIND11_MODULE(TurnpikeMM, m) {
    m.doc() = "MM TurnpikeSolver utilities implemented in C++ for vectorization and true parallelism.";

    m.def("Qz", [](const py::array_t<double> &z, py::array_t<double> &d) {
        if (z.ndim() != 1 || d.ndim() != 1)
            throw std::runtime_error("Both input arrays must be 1-dimensional.");

        const auto n = z.size();
        const auto m = d.size();

        if (m != (n * n - n) / 2)
            throw std::runtime_error("The size of array d must be choose2(z.size).");

        auto zdata = z.data();
        auto ddata = d.mutable_data();
        for (auto k = 0; k < m; ++k) {
            const auto [i, j] = get_row_col(k);
            ddata[k] = zdata[j] - zdata[i];
        }

        return py::none();
    }, py::arg("z"), py::arg("out"), "Compute Qz and store the result in d.");

    m.def("dQ", [](const py::array_t<double> &d, py::array_t<double> &z) {
        if (z.ndim() != 1 || d.ndim() != 1)
            throw std::runtime_error("Both input arrays must be 1-dimensional.");

        const auto n = z.size();
        const auto m = d.size();

        if (m != (n * n - n) / 2)
            throw std::runtime_error("The size of array d must be choose2(z.size).");

        auto ddata = d.data();
        auto zdata = z.mutable_data();

        std::fill(zdata, zdata + n, 0.);

        for (int k = 0; k < m; ++k) {
            const auto [i, j] = get_row_col(k);
            zdata[i] -= ddata[k];
            zdata[j] += ddata[k];
        }

        return py::none();
    }, py::arg("d"), py::arg("z"), "Compute dQ and store the result in z.");

    m.def("match_distances", [](py::array_t<double> &dhat_array, const py::array_t<double> &d, int threads) {
        if (threads == -1) threads = omp_get_max_threads();
        return match_distances(dhat_array, d, threads);
    }, py::arg("dhat"), py::arg("d"), py::arg("threads") = -1, "Replace dhat with the equivalently ordered entries in d.");

    m.def("build_sparse_matrix", [](int n, int threads) {
        if (threads == -1) threads = omp_get_max_threads();
        omp_set_num_threads(threads);
        Eigen::setNbThreads(threads);

        const auto Q = build_incidence_matrix(n);
        return eigen_to_scipy_sparse(Q);
    }, py::arg("n"), py::arg("threads") = -1);

    m.def("online_matching", &online_matching, "Computes the matching cost and assignment vector",
          py::arg("D"), py::arg("Δ"), py::arg("I"), py::arg("idx"));

    m.def("any_same", [](const py::array_t<int> &L, const py::array_t<int> &R) {
        auto n = L.size();

        if (R.size() != n) throw std::runtime_error("L and R arrays must be the same size.");

        const int *L_arr = L.data();
        const int *R_arr = R.data();

        for (decltype(n) k = 0; k < n; ++k)
            if (L_arr[k] == R_arr[k]) return true;

        return false;
    }, "Ensure at least one distance within tolerance", py::arg("L"), py::arg("R"));

    m.def("undo_matching", [](py::array_t<int> &I, int a) {
        auto m = I.size();

        int *arr = I.mutable_data();
        for (decltype(m) k = 0; k < m; ++k) {
            if (arr[k] >= a) arr[k] = -1;
        }
    }, "undo all matching between a and b", py::arg("I"), py::arg("a"));
}
