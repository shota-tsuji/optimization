use ndarray::{arr2, Array2, ArrayView2};

/// Jacobi method
///
/// returns solution vector for Ax = b.
///
/// * `a` - Matrix A.
/// * `x` - Vector x.
/// * `b` - Vector b.
pub fn jacobi(a: ArrayView2<f64>, x_0: &Vec<f64>, b: &Vec<f64>, eps: f64) -> Vec<f64> {
    let mut x_k1 = x_0.clone();
    let mut y = x_k1.clone();
    loop {
        //let y = x.clone();
        for i in 0..x_k1.len() {
            let mut xi = b[i];
            for j in 0..y.len() {
                if j != i {
                    xi -= a[[i, j]] * y[j];
                }
            }
            x_k1[i] = xi / a[[i, i]];
        }

        if is_convergent(&x_k1, &y, eps) {
            return x_k1;
        }

        for i in 0..y.len() {
            y[i] = x_k1[i];
        }
    }
}

/// Gauss-Seidel method
///
/// returns solution vector for Ax = b.
///
/// * `a` - Matrix A.
/// * `x` - Vector x.
/// * `b` - Vector b.
pub fn gauss_seidel(a: ArrayView2<f64>, x_0: &Vec<f64>, b: &Vec<f64>, eps: f64) -> Vec<f64> {
    let mut x_k1 = x_0.clone();
    let mut x_k_ = x_k1.clone();
    loop {
        for i in 0..x_k1.len() {
            let mut xi = b[i];
            for j in 0..x_k1.len() {
                if j != i {
                    xi -= a[[i, j]] * x_k1[j];
                }
            }
            x_k1[i] = xi / a[[i, i]];
        }

        if is_convergent(&x_k1, &x_k_, eps) {
            return x_k1;
        }

        println!("{:?}", x_k1);
        for i in 0..x_k1.len() {
            x_k_[i] = x_k1[i];
        }
    }
}

/// LU decomposition
///
/// Decompose matrix A into LU.
///
/// * `a` - Matrix A
pub fn lu_decompose(a: ArrayView2<f64>) -> Array2<f64> {
    let mut a = a.to_owned();
    for i in 0..a.shape()[0] {
        let uii = a[[i, i]];
        for j in i + 1..a.shape()[0] {
            let l_ij = a[[j, i]] / uii;
            a[[j, i]] = l_ij;
            for k in i + 1..a.shape()[1] {
                a[[j, k]] -= l_ij * a[[i, k]];
            }
        }
    }
    a
}

pub fn is_diagonally_dominant(a: ArrayView2<f64>) -> bool {
    for i in 0..a.shape()[0] {
        let a_ii = a[[i, i]];
        let mut non_diagonals = 0.0;
        for j in 0..a.shape()[1] {
            if j != i {
                non_diagonals += a[[i, j]].abs();
            }
        }

        if a_ii.abs() <= non_diagonals {
            return false;
        }
    }

    true
}

/// check whether convergent
///
/// returns true if satisfy convergence, otherwise returns false.
///
/// * `x_k1` - Vector x^(k+1).
/// * `x_k_` - Vector x^(k).
/// * `eps` - epsilon (error).
fn is_convergent(x_k1: &Vec<f64>, x_k_: &Vec<f64>, eps: f64) -> bool {
    let mut delta_sum = 0.0;
    for i in 0..x_k1.len() {
        delta_sum += (x_k1[i] - x_k_[i]).abs();
    }

    let mut ans_sum = 0.0;
    for i in 0..x_k1.len() {
        ans_sum += x_k1[i].abs();
    }

    if delta_sum < eps * ans_sum {
        true
    } else {
        false
    }
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
    fn jacobi_3x3_eps6() {
        let eps = 1e-6;
        let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
        let b = vec![0.0, 4.0, 6.0];
        let x_0 = vec![0.0, 0.0, 0.0];
        assert!(is_convergent(
            &vec![-1.0, 1.0, 2.0],
            &jacobi(a.view(), &x_0, &b, eps),
            eps
        ));
    }

    #[test]
    fn jacobi_3x3_eps10() {
        let eps = 1e-10;
        let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
        let b = vec![0.0, 4.0, 6.0];
        let x_0 = vec![0.0, 0.0, 0.0];
        assert!(is_convergent(
            &vec![-1.0, 1.0, 2.0],
            &jacobi(a.view(), &x_0, &b, eps),
            eps
        ));
    }

    #[test]
    fn jacobi_2x2() {
        let eps = 1e-10;
        let a = arr2(&[[5.0, 4.0], [2.0, 3.0]]);
        let b = vec![13.0, 8.0];
        let x_0 = vec![0.0, 0.0];
        assert!(is_convergent(
            &vec![1.0, 2.0],
            &jacobi(a.view(), &x_0, &b, eps),
            eps
        ));
    }

    #[test]
    fn gauss_seidel_2x2() {
        let eps = 1e-10;
        let a = arr2(&[[5.0, 4.0], [2.0, 3.0]]);
        let b = vec![13.0, 8.0];
        let x_0 = vec![0.0, 0.0];
        let x_k = &gauss_seidel(a.view(), &x_0, &b, eps);
        assert!(
            is_convergent(&vec![1.0, 2.0], x_k, eps),
            "eps={}, x={:?}",
            eps,
            x_k
        );
    }

    #[test]
    fn gauss_seidel_3x3() {
        let eps = 1e-10;
        let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
        let b = vec![0.0, 4.0, 6.0];
        let x_0 = vec![0.0, 0.0, 0.0];
        let x_k = &gauss_seidel(a.view(), &x_0, &b, eps);
        assert!(
            is_convergent(&vec![-1.0, 1.0, 2.0], x_k, eps),
            "eps={}, x={:?}",
            eps,
            x_k
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
    fn diagonally_dominant_2x2() {
        let a = arr2(&[[3.0, -2.0], [1.0, -3.0]]);
        assert!(is_diagonally_dominant(a.view()));
    }

    #[test]
    fn not_diagonally_dominant_2x2() {
        let a = arr2(&[[0.0, 1.0], [0.0, 1.0]]);
        assert!(!is_diagonally_dominant(a.view()));
    }

    #[test]
    fn diagonally_dominant_3x3() {
        let a = arr2(&[[3.0, -1.0, 1.0], [1.0, -3.0, 1.0], [-1.0, 2.0, 4.0]]);
        assert!(is_diagonally_dominant(a.view()));
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
