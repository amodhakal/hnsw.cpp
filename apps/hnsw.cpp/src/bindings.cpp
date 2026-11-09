#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnsw/hnsw.hpp"

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

std::size_t vector_length(const py::buffer_info& info) {
    if (info.ndim == 1) {
        return static_cast<std::size_t>(info.shape[0]);
    }
    if (info.ndim == 2 && info.shape[0] == 1) {
        return static_cast<std::size_t>(info.shape[1]);
    }
    throw std::invalid_argument(
        "Expected a 1-dimensional vector or a single-row 2-dimensional array");
}

py::array_t<std::uint64_t> labels_array(
    const std::vector<std::uint64_t>& values) {
    py::array_t<std::uint64_t> array(
        {static_cast<py::ssize_t>(values.size())});
    if (!values.empty()) {
        std::memcpy(array.mutable_data(),
                    values.data(),
                    values.size() * sizeof(std::uint64_t));
    }
    return array;
}

py::array_t<double> distances_array(const std::vector<double>& values) {
    py::array_t<double> array(
        {static_cast<py::ssize_t>(values.size())});
    if (!values.empty()) {
        std::memcpy(array.mutable_data(),
                    values.data(),
                    values.size() * sizeof(double));
    }
    return array;
}

py::array_t<std::uint64_t> labels_array_2d(
    const std::vector<std::uint64_t>& values,
    std::size_t rows,
    std::size_t columns) {
    py::array_t<std::uint64_t> array(
        {static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(columns)});
    if (!values.empty()) {
        std::memcpy(array.mutable_data(),
                    values.data(),
                    values.size() * sizeof(std::uint64_t));
    }
    return array;
}

py::array_t<double> distances_array_2d(const std::vector<double>& values,
                                       std::size_t rows,
                                       std::size_t columns) {
    py::array_t<double> array(
        {static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(columns)});
    if (!values.empty()) {
        std::memcpy(array.mutable_data(),
                    values.data(),
                    values.size() * sizeof(double));
    }
    return array;
}

}  // namespace

PYBIND11_MODULE(hnsw_cpp, module) {
    module.doc() =
        "C++ implementation of Hierarchical Navigable Small World graphs";

    py::class_<hnsw::Index>(module, "HNSWIndex")
        .def(py::init([](std::size_t dim,
                         const std::string& space,
                         std::size_t m,
                         std::size_t ef_construction,
                         std::size_t ef_search,
                         std::uint64_t seed,
                         bool heuristic,
                         const std::string& visited_mode,
                         double alpha,
                         double generosity) {
                 return std::make_unique<hnsw::Index>(hnsw::Config{
                     dim,
                     hnsw::parse_space(space),
                     m,
                     ef_construction,
                     ef_search,
                     seed,
                     heuristic,
                     hnsw::parse_visited_mode(visited_mode),
                     alpha,
                     generosity,
                 });
             }),
             py::arg("dim"),
             py::arg("space") = "cosine",
             py::arg("m") = 16,
             py::arg("ef_construction") = 200,
             py::arg("ef_search") = 50,
             py::arg("seed") = 100,
             py::arg("heuristic") = true,
             py::arg("visited_mode") = "bloom",
             py::arg("alpha") = 1.1,
             py::arg("generosity") = 1.1)
        .def(
            "add_items",
            [](hnsw::Index& index, py::object vectors, py::object labels) {
                FloatArray array(vectors);
                const py::buffer_info info = array.request();
                if (info.ndim != 2) {
                    throw std::invalid_argument(
                        "Expected a 2-dimensional array of vectors");
                }

                const std::size_t rows =
                    static_cast<std::size_t>(info.shape[0]);
                const std::size_t columns =
                    static_cast<std::size_t>(info.shape[1]);
                if (columns != index.dim()) {
                    throw std::invalid_argument(
                        "Expected " + std::to_string(index.dim()) +
                        "-dimensional vectors, got " +
                        std::to_string(columns));
                }

                std::vector<std::uint64_t> label_storage;
                const std::uint64_t* label_pointer = nullptr;
                if (!labels.is_none()) {
                    py::array_t<std::uint64_t,
                                py::array::c_style | py::array::forcecast>
                        label_array(labels);
                    const py::buffer_info label_info = label_array.request();
                    if (label_info.ndim != 1 ||
                        static_cast<std::size_t>(label_info.shape[0]) != rows) {
                        throw std::invalid_argument(
                            "labels and vectors must have the same length");
                    }
                    const auto* label_data =
                        static_cast<const std::uint64_t*>(label_info.ptr);
                    label_storage.assign(label_data, label_data + rows);
                    label_pointer = label_storage.data();
                }

                const float* data = static_cast<const float*>(info.ptr);
                py::gil_scoped_release release;
                index.add_items(data, rows, label_pointer);
            },
            py::arg("vectors"),
            py::arg("labels") = py::none())
        .def(
            "add_item",
            [](hnsw::Index& index, py::object vector, py::object label) {
                FloatArray array(vector);
                const py::buffer_info info = array.request();
                const std::size_t length = vector_length(info);
                if (length != index.dim()) {
                    throw std::invalid_argument(
                        "Expected " + std::to_string(index.dim()) +
                        "-dimensional vector, got " +
                        std::to_string(length));
                }

                std::optional<std::uint64_t> value;
                if (!label.is_none()) {
                    value = py::cast<std::uint64_t>(label);
                }

                const float* data = static_cast<const float*>(info.ptr);
                py::gil_scoped_release release;
                index.add_item(data, value);
            },
            py::arg("vector"),
            py::arg("label") = py::none())
        .def("set_ef", &hnsw::Index::set_ef, py::arg("ef"))
        .def(
            "search",
            [](const hnsw::Index& index, py::object query, std::size_t k) {
                FloatArray array(query);
                const py::buffer_info info = array.request();
                const std::size_t length = vector_length(info);
                if (length != index.dim()) {
                    throw std::invalid_argument(
                        "Expected " + std::to_string(index.dim()) +
                        "-dimensional query, got " +
                        std::to_string(length));
                }

                const float* data = static_cast<const float*>(info.ptr);
                hnsw::SearchResult result;
                {
                    py::gil_scoped_release release;
                    result = index.search(data, k);
                }
                return py::make_tuple(labels_array(result.labels),
                                      distances_array(result.distances));
            },
            py::arg("query"),
            py::arg("k") = 10)
        .def(
            "search_batch",
            [](const hnsw::Index& index, py::object queries, std::size_t k) {
                FloatArray array(queries);
                const py::buffer_info info = array.request();
                if (info.ndim != 2) {
                    throw std::invalid_argument(
                        "Expected a 2-dimensional array of queries");
                }

                const std::size_t rows =
                    static_cast<std::size_t>(info.shape[0]);
                const std::size_t columns =
                    static_cast<std::size_t>(info.shape[1]);
                if (columns != index.dim()) {
                    throw std::invalid_argument(
                        "Expected " + std::to_string(index.dim()) +
                        "-dimensional queries, got " +
                        std::to_string(columns));
                }

                const float* data = static_cast<const float*>(info.ptr);
                std::vector<hnsw::SearchResult> results;
                {
                    py::gil_scoped_release release;
                    results = index.search_batch(data, rows, k);
                }

                py::list output;
                for (const hnsw::SearchResult& result : results) {
                    output.append(py::make_tuple(labels_array(result.labels),
                                                 distances_array(result.distances)));
                }
                return output;
            },
            py::arg("queries"),
            py::arg("k") = 10)
        .def(
            "knn_query",
            [](const hnsw::Index& index, py::object queries, std::size_t k) {
                FloatArray array(queries);
                const py::buffer_info info = array.request();
                if (info.ndim != 2) {
                    throw std::invalid_argument(
                        "Expected a 2-dimensional array of queries");
                }
                if (k < 1) {
                    throw std::invalid_argument("k must be at least 1");
                }

                const std::size_t rows =
                    static_cast<std::size_t>(info.shape[0]);
                const std::size_t columns =
                    static_cast<std::size_t>(info.shape[1]);
                if (columns != index.dim()) {
                    throw std::invalid_argument(
                        "Expected " + std::to_string(index.dim()) +
                        "-dimensional queries, got " +
                        std::to_string(columns));
                }
                if (k > index.size()) {
                    throw std::invalid_argument(
                        "k cannot exceed the number of indexed vectors");
                }

                std::vector<std::uint64_t> labels(rows * k);
                std::vector<double> distances(rows * k);
                const float* data = static_cast<const float*>(info.ptr);
                const std::size_t dim = index.dim();
                {
                    py::gil_scoped_release release;
                    for (std::size_t row = 0; row < rows; ++row) {
                        const hnsw::SearchResult result =
                            index.search(data + row * dim, k);
                        for (std::size_t column = 0;
                             column < result.labels.size();
                             ++column) {
                            labels[row * k + column] = result.labels[column];
                            distances[row * k + column] =
                                result.distances[column];
                        }
                    }
                }

                return py::make_tuple(labels_array_2d(labels, rows, k),
                                      distances_array_2d(distances, rows, k));
            },
            py::arg("queries"),
            py::arg("k") = 10)
        .def("index_size_bytes", &hnsw::Index::index_size_bytes)
        .def_property_readonly("dim", &hnsw::Index::dim)
        .def_property_readonly("space", [](const hnsw::Index& index) {
            return std::string(hnsw::space_name(index.space()));
        })
        .def_property_readonly("ef_search", &hnsw::Index::ef_search)
        .def("__len__", &hnsw::Index::size)
        .def("__repr__", [](const hnsw::Index& index) {
            return "<hnsw_cpp.HNSWIndex(space='" +
                   std::string(hnsw::space_name(index.space())) +
                   "', dim=" + std::to_string(index.dim()) +
                   ", size=" + std::to_string(index.size()) + ")>";
        });
}
