use ndarray::ArrayView2;

fn main() {
    println!("Hello, world!");
}

/// Jacobi method
///
/// returns solution vector for Ax = b.
///
/// * `a` - Matrix A.
/// * `x` - Vector x.
/// * `b` - Vector b.
fn jacobi(a: ArrayView2<f64>, x: Vec<f64>, b: Vec<f64>, eps: f64) -> Vec<f64> {
    let mut x = x;
    let mut n = 0;
    loop {
        let y = x.clone();
        for i in 0..x.len() {
            let mut xi = b[i];
            for j in 0..y.len() {
                if j != i {
                    xi -= a[[i, j]] * y[j];
                }
            }
            x[i] = xi / a[[i, i]];
        }
        println!("{}: {:?}", n, x);

        let mut flag = true;
        for i in 0..x.len() {
            if (x[i] - y[i]).abs() > eps {
                flag = false;
            }
        }
        if flag {
            return x;
        }
        n += 1;
    }
}

fn is_diagonally_dominant(a: ArrayView2<f64>) -> bool {
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

fn f(x: f64) -> f64 {
    return -x * x;
}

fn next_set(f_1: fn(f64) -> f64, x: f64, h: f64) -> (f64, f64, f64) {
    let h = f_1(x).signum() * h.abs();
    (h, x, x + h)
}

fn increase_x(f: fn(f64) -> f64, x_: f64, step: f64) -> (f64, f64) {
    let mut h = step;
    let mut x;
    let mut x_ = x_;
    loop {
        h = 2.0 * h;
        x = x_;
        x_ = x + h;
        println!("x = {}", x);
        if f(x) >= f(x_) {
            break;
        }
    }
    return (x_ - h, h / 2.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use std::iter::zip;

    const F_1: fn(f64) -> f64 = |x: f64| (-2.0 * x);

    fn is_convergent(ans: Vec<f64>, x_k: Vec<f64>, eps: f64) -> bool {
        let mut delta_sum = 0.0;
        for i in 0..ans.len() {
            delta_sum += (ans[i] - x_k[i]).abs();
        }

        let mut ans_sum = 0.0;
        for i in 0..ans.len() {
            ans_sum += ans[i].abs();
        }

        if delta_sum < eps * ans_sum {
            true
        } else {
            false
        }
    }

    #[test]
    fn calc1() {
        let eps = 1e-6;
        let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
        let b = vec![0.0, 4.0, 6.0];
        let x = vec![0.0, 0.0, 0.0];
        let xk = jacobi(a.view(), x, b, eps);
        let ans = [-1.0, 1.0, 2.0];
        assert!(
            zip(&xk, ans).all(|(&x_i, ans_i)| (ans_i - x_i).abs() < eps),
            "x = [{:?}]",
            xk
        );
    }

    #[test]
    fn calc2() {
        let eps = 1e-10;
        let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
        let b = vec![0.0, 4.0, 6.0];
        let x = vec![0.0, 0.0, 0.0];
        let xk = jacobi(a.view(), x, b, eps);
        let ans = [-1.0, 1.0, 2.0];
        assert!(
            zip(&xk, ans).all(|(&x_i, ans_i)| (ans_i - x_i).abs() < eps),
            "x = [{:?}]",
            xk
        );
    }

    #[test]
    fn jacobi_2x2() {
        let eps = 1e-10;
        let a = arr2(&[[5.0, 4.0], [2.0, 3.0]]);
        let b = vec![13.0, 8.0];
        let x_0 = vec![0.0, 0.0];
        let x_k = jacobi(a.view(), x_0, b, eps);
        let ans = vec![1.0, 2.0];
        assert!(is_convergent(ans, x_k, eps));
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
