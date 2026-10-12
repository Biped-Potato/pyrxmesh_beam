#include <pybind11/pybind11.h>

#include "pyrxmesh/diff_plugin_api.h"

using T = float;

namespace py = pybind11;
using namespace rxmesh;
using namespace pyrxmesh;

void compute_rest_shape(py::object mesh_obj,
                          py::object coords_obj,
                          py::object rest_coords_obj)
{
    auto coordinates = pyrxmesh::vertex_attribute<T>(coords_obj);
    // 2x2 column major matrix as a vector
    auto rest_shape  = pyrxmesh::face_attribute<T>(rest_coords_obj);

    pyrxmesh::for_each<Op::FV, 256>(
        mesh_obj,
        [coordinates, rest_shape] __device__(const FaceHandle& fh,
                                 const VertexIterator& iter) mutable {
            const VertexHandle v0 = iter[0];
            const VertexHandle v1 = iter[1];
            const VertexHandle v2 = iter[2];
            
            assert(v0.is_valid() && v1.is_valid() && v2.is_valid());

            // 3d position
            Eigen::Vector3<T> ar_3d = coordinates.to_eigen<3>(v0);
            Eigen::Vector3<T> br_3d = coordinates.to_eigen<3>(v1);
            Eigen::Vector3<T> cr_3d = coordinates.to_eigen<3>(v2);

            // Local 2D coordinate system
            Eigen::Vector3<T> n  = (br_3d - ar_3d).cross(cr_3d - ar_3d);
            Eigen::Vector3<T> b1 = (br_3d - ar_3d).normalized();
            Eigen::Vector3<T> b2 = n.cross(b1).normalized();

            // Express a, b, c in local 2D coordinates system
            Eigen::Vector2<T> ar_2d(T(0.0), T(0.0));
            Eigen::Vector2<T> br_2d((br_3d - ar_3d).dot(b1), T(0.0));
            Eigen::Vector2<T> cr_2d((cr_3d - ar_3d).dot(b1),
                                    (cr_3d - ar_3d).dot(b2));

            // Save 2-by-2 matrix with edge vectors as columns
            Eigen::Matrix<T, 2, 2> fout = col_mat(br_2d - ar_2d, cr_2d - ar_2d);

            rest_shape(fh, 0) = fout(0, 0);
            rest_shape(fh, 1) = fout(1, 0);
            rest_shape(fh, 2) = fout(0, 1);
            rest_shape(fh, 3) = fout(1, 1);
        });
}

using Problem =
    diff::ScalarGradientProblem<T, 3, VertexHandle>;

void add_terms(Problem& problem, py::object rest_coords_obj){
    auto rest_shape  = pyrxmesh::face_attribute<T>(rest_coords_obj);

    problem.template add_term<Op::FV>(
        [rest_shape] __device__(const auto& fh, const auto& iter, auto& opt_var) {
            // fh is a face handle
            // iter is an iterator over fh's vertices
            // opt_var is the uv coordinates

            assert(iter[0].is_valid() && iter[1].is_valid() &&
                   iter[2].is_valid());

            assert(iter.size() == 3);

            // autodiff scalar; carries derivatives w.r.t. the uv variables
            using ActiveT = ACTIVE_TYPE(fh);

            // uv
            Eigen::Vector3<ActiveT> a = opt_var.template active<3>(fh, iter, 0);
            Eigen::Vector3<ActiveT> b = opt_var.template active<3>(fh, iter, 1);
            Eigen::Vector3<ActiveT> c = opt_var.template active<3>(fh, iter, 2);
            
            Eigen::Vector2<ActiveT> a_2d(a(0),a(1));
            Eigen::Vector2<ActiveT> b_2d(b(0),b(1));
            Eigen::Vector2<ActiveT> c_2d(c(0),c(1));


            // Triangle flipped?
            Eigen::Matrix<ActiveT, 2, 2> M = col_mat(b_2d - a_2d, c_2d - a_2d);


            if (M.determinant() <= 0.0) {
                using PassiveT = PassiveType<ActiveT>;
                return ActiveT(std::numeric_limits<PassiveT>::max());
            }

            // Get constant 2D rest shape and area of triangle t
            Eigen::Matrix<T, 2, 2> Mr;
            
            Mr(0,0) = rest_shape(fh, 0);
            Mr(1,0) = rest_shape(fh, 1);
            Mr(0,1) = rest_shape(fh, 2);
            Mr(1,1) = rest_shape(fh, 3);

            const T A = T(0.5) * Mr.determinant();

            // Compute symmetric Dirichlet energy
            Eigen::Matrix<ActiveT, 2, 2> J = M * Mr.inverse();

            ActiveT res = A * (J.squaredNorm() + J.inverse().squaredNorm());

            return res;
        });
}

std::shared_ptr<diff::ScalarEnergyBase> make_energy(
    py::object mesh_obj,
    py::object rest_coords_obj
){
    auto energy = std::make_shared<Problem>(mesh_obj);
    add_terms(*energy, rest_coords_obj);
    return energy;
}


PYBIND11_MODULE(_pyrxmesh_beam, m)
{
    pyrxmesh::require_compatible_runtime(m);
    m.def(
        "compute_rest_shape",
        &compute_rest_shape,
        py::arg("mesh"),
        py::arg("coords"),
        py::arg("rest_coords")
    );

    m.def(
        "make_energy",
        &make_energy,
        py::arg("mesh"),
        py::arg("rest_coords")
    );
}
