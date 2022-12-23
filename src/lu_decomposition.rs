use ndarray::{arr2, Array2, ArrayView2};

use crate::assert::is_convergent;

/// LU decomposition method for equations
///
/// returns solution vector for Ax = b.
///
/// * `a` - Matrix A.
/// * `b` - Vector b.
pub fn solve_by_lu(a: ArrayView2<f64>, b: &Vec<f64>) -> Vec<f64> {
    let lu = lu_decompose(a);
    let n = lu.shape()[0];
    let mut y = vec![0.0; n];

    // solve Ly = b, where y = Ux.
    for i in 0..n {
        let mut yi = b[i];
        for j in 0..i {
            yi -= lu[[i, j]] * y[j];
        }
        y[i] = yi;
    }

    // solve Ux = y.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in i + 1..n {
            x[i] -= x[j] * lu[[i, j]];
        }
        x[i] /= lu[[i, i]];
    }

    x
}

/// LU decomposition
///
/// Decompose matrix A into LU.
///
/// * `a` - Regular Square Matrix A
pub fn lu_decompose(a: ArrayView2<f64>) -> Array2<f64> {
    let mut a = a.to_owned();
    let n = a.shape()[0];
    // loop for pivot row
    for i in 0..n {
        // forward elimination (loop for columns under the pivot row)
        for j in i + 1..n {
            // elementary row transformation
            a[[j, i]] /= a[[i, i]];
            for k in i + 1..n {
                a[[j, k]] -= a[[j, i]] * a[[i, k]];
            }
        }
    }
    a
}

/// Calculate PA, P is permutation matrix
///
/// permute the rows of A.
///
/// * `p` - Matrix P
/// * `a` - Matrix A
pub fn permute(p: ArrayView2<f64>, a: ArrayView2<f64>) -> Array2<f64> {
    p.to_owned().dot(&a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn lu_solve_2x2() {
        let eps = 1e-10;
        let a = arr2(&[[5.0, 4.0], [2.0, 3.0]]);
        let b = vec![13.0, 8.0];
        let x = solve_by_lu(a.view(), &b);
        assert!(
            is_convergent(&vec![1.0, 2.0], &x, eps),
            "eps={}, x={:?}",
            eps,
            x
        );
    }

    #[test]
    fn lu_solve_other_2x2() {
        let eps = 1e-10;
        let a = arr2(&[[2.0, 3.0], [5.0, 4.0]]);
        let b = vec![8.0, 13.0];
        let x = solve_by_lu(a.view(), &b);
        assert!(
            is_convergent(&vec![1.0, 2.0], &x, eps),
            "eps={}, x={:?}",
            eps,
            x
        );
    }

    #[test]
    fn lu_decompose_2x2_unchanged() {
        let eps = 1e-10;
        let a = arr2(&[[5.0, 4.0], [2.0, 3.0]]);
        //let b = vec![13.0, 8.0];
        assert_eq!(a, compose_a(lu_decompose(a.view())));
    }

    #[test]
    fn lu_decompose_2x2_row_changed() {
        let eps = 1e-10;
        let a = arr2(&[[2.0, 3.0], [5.0, 4.0]]);
        //let b = vec![8.0, 13.0];
        assert_eq!(a, compose_a(lu_decompose(a.view())));
    }

    #[test]
    fn lu_decompose_3x3_row_changed() {
        let eps = 1e-10;
        let a = arr2(&[[3.0, 1.0, 0.0], [6.0, 1.0, -2.0], [-3.0, 0.0, 3.0]]);
        assert_eq!(a, compose_a(lu_decompose(a.view())));
    }

    fn compose_a(lu: Array2<f64>) -> Array2<f64> {
        let mut l = Array2::eye(lu.shape()[0]);
        for i in 1..l.shape()[0] {
            for j in 0..i {
                l[[i, j]] = lu[[i, j]];
            }
        }

        let mut u = Array2::zeros((lu.shape()[0], lu.shape()[1]));
        for i in 0..u.shape()[0] {
            for j in i..u.shape()[1] {
                u[[i, j]] = lu[[i, j]];
            }
        }

        l.dot(&u)
    }

    #[test]
    fn permute_2x2_unchanged() {
        let p = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let a = arr2(&[[0.0, 1.0], [2.0, 3.0]]);
        assert_eq!(a, permute(p.view(), a.view()));
    }

    #[test]
    fn permute_2x2_permuted() {
        let p = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
        let a = arr2(&[[0.0, 1.0], [2.0, 3.0]]);
        assert_eq!(arr2(&[[2.0, 3.0], [0.0, 1.0]]), permute(p.view(), a.view()));
    }
}
