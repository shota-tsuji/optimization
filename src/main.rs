use ndarray::{arr1, ArrayView1};
use optimization::assert::norm_l2;
use optimization::newton::FuncX;
use optimization::quasi_newton::bfgs;

fn main() {
    println!("hello");

    let eps = 1e-6;
    let _x_0 = arr1(&[0.0]);
    let f: FuncX = |x: ArrayView1<f64>| x[0].powf(2.0) - 2.0 * x[0] + 1.0;
    let f_x: FuncX = |x: ArrayView1<f64>| 2.0 * x[0] - 2.0;
    let del_f = vec![&f_x];
    let x_0 = arr1(&[0.0]);
    let ans = arr1(&[1.0]);
    let x_k1 = bfgs(f, del_f, x_0.view());
    println!("x={:?}, truth={:?}", x_k1, ans);
    assert!(norm_l2((ans - x_k1).view()) < eps);
}