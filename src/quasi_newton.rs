/// Secant method
///
/// returns the root of f(x).
///
/// * `x_0` - initial value for recurrence relation.
/// * `x_1` - initial value for recurrence relation.
/// * `f` - f(x).
/// * `eps` - error threshold.
pub fn secant(f: &fn(f64) -> f64, x_0: f64, x_1: f64, eps: f64) -> f64 {
    // define x0, x1, x2
    let mut x0 = x_0;
    let mut x1 = x_1;

    //update x0, x1
    loop {
        let x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0));
        println!("{}", x2);
        x0 = x1;
        x1 = x2;
        // if convergence, return x2
        if f64::abs(x0 - x1) < eps {
            return x2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use core::f64;
    

    #[test]
    fn secant_sin_cos() {
        let eps = 1e-10;
        // arrange f(x), x0, x1, n
        let f: fn(f64) -> f64 = |x: f64| f64::sin(x / 4.0) - f64::cos(x / 4.0);
        let x_0 = 0.0;
        let x_1 = 8.0;
        // calculate x using secant-method
        let x = secant(&f, x_0, x_1, eps);
        // x approximately equals to PI.
        assert!(std::f64::consts::PI - x < 1e-10);
    }
}
