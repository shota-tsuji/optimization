use ndarray::ArrayView2;

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

pub fn next_set(f_1: fn(f64) -> f64, x: f64, h: f64) -> (f64, f64, f64) {
    let h = f_1(x).signum() * h.abs();
    (h, x, x + h)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    const F_1: fn(f64) -> f64 = |x: f64| (-2.0 * x);

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
    fn return_pulus_set() {
        let h = 0.1;
        let x = -3.0;
        assert_eq!((h, x, x + h), next_set(F_1, x, h));
    }

    #[test]
    fn return_pulus_set2() {
        let h = 0.2;
        let x = -3.0;
        assert_eq!((h, x, x + h), next_set(F_1, x, h));
    }

    #[test]
    fn return_x_plus() {
        let h = 0.2;
        let x = -1.1;
        assert_eq!((h, x, x + h), next_set(F_1, x, h));
    }

    #[test]
    fn return_minus_set() {
        let h = 0.1;
        let x = 3.0;
        assert_eq!((-h, x, x - h), next_set(F_1, x, h));
    }

    #[test]
    fn return_x_minus() {
        let h = 0.1;
        let x = 1.1;
        assert_eq!((-h, x, x - h), next_set(F_1, x, h));
    }

    #[test]
    fn return_x_set2() {
        let h = 0.2;
        let x = 1.1;
        assert_eq!((-h, x, x - h), next_set(F_1, x, h));
    }

    #[test]
    fn signum() {
        let a: f64 = -0.0;
        assert_eq!(0.0, a.signum() * a.abs());
    }
}
