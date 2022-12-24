pub fn newton(x_0: f64, f_x: fn(f64)-> f64, f_xx: fn(f64)-> f64, eps: f64) -> f64 {
    let mut x_k_ = x_0;
    let mut x_k1 = 0.0;
    loop {
        x_k1 = x_k_ - f_x(x_k_) / f_xx(x_k_);
        if (x_k1 - x_k_).abs() < eps {
            break;
        }
        x_k_ = x_k1;
        println!("{}", x_k1)
    }

    x_k1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_1x1() {
        // f(x) = x^2 - 2x + 1
        let eps = 1e-10;
        let x_0 = 0.0;
        let f_x = |x: f64| 2.0*x - 2.0;
        let f_xx = |x: f64| 2.0;
        assert!((1.0 - newton(x_0, f_x, f_xx, eps)).abs() < eps);
    }

    #[test]
    fn newton_1x1_other() {
        // f(x) = x^2 -4x +4
        let eps = 1e-10;
        let x_0 = 0.0;
        let f_x = |x: f64| 2.0*x - 4.0;
        let f_xx = |x: f64| 2.0;
        assert!((2.0 - newton(x_0, f_x, f_xx, eps)).abs() < eps);
    }
}