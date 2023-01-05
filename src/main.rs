mod assert;
mod gauss_seidel;
mod jacobi;
mod newton;
mod quasi_newton;

use ndarray::{arr1, Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use newton::FuncX;

fn main() {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("dummy.csv")
        .unwrap();
    let n = rdr.records().count();
    println!("{}", n);

    // ラベルと特徴量に分ける
    let mut y = Array::zeros(n);
    let mut features = Array::zeros((n, 123));

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("dummy.csv")
        .unwrap();
    for (i, record) in rdr.records().enumerate() {
        let record = record.unwrap();
        y[i] = record[0].parse::<i8>().unwrap();
        for (j, data) in record.iter().enumerate() {
            if j > 0 {
                let index = data.parse::<usize>().unwrap() - 1;
                features[[i, index]] = 1.0;
            }
        }
    }

    println!("{:?}, {:?},{:?}", y, features, features.shape()[1]);
    println!("{:?}", &features.shape()[1]);
    let regression = Regression::new(y, features.clone());
    println!("loss={}", regression.loss(features.view()));

    let _eps = 1e-6;
    let _x_0 = arr1(&[0.0]);
    let _f: FuncX = |x: ArrayView1<f64>| x[0].powf(2.0) - 2.0 * x[0] + 1.0;
    let f_x: FuncX = |x: ArrayView1<f64>| 2.0 * x[0] - 2.0;
    let _del_f = vec![&f_x];
    let _x_0 = arr1(&[0.0]);
    let _ans = arr1(&[1.0]);
    //let x_k1 = bfgs(f, del_f, x_0.view());
    //println!("x={:?}, truth={:?}", x_k1, ans);
    //assert!(norm_l2((ans - x_k1).view()) < eps);
}

struct Regression {
    y: Array1<i8>,
    features: Array2<f64>,
    w: Array1<f64>,
    n: usize,
    l: usize,
    del_f: Array1<f64>,
}

impl Regression {
    fn new(y: Array1<i8>, features: Array2<f64>) -> Regression {
        let l = y.len();
        let n = features.shape()[1];
        let w = Array1::zeros(n);
        let del_f = Array1::zeros(n);
        Regression {
            y,
            features,
            w,
            n,
            l,
            del_f,
        }
    }

    fn loss(&self, features: ArrayView2<f64>) -> f64 {
        let mut sum = 0.0;

        for i in 0..self.l {
            let mut wx = 0.0;
            for j in 0..self.n {
                wx += self.w[j] * features[[i, j]];
            }
            let yi = f64::from(self.y[i]);
            sum += f64::ln(1.0 + f64::exp(-yi * wx));
        }

        sum
    }

    fn derivative(&mut self, mut del_f: ArrayViewMut1<f64>) {
        for i in 0..self.l {
            let mut wx = 0.0;
            for j in 0..self.n {
                wx += self.w[j] * self.features[[i, j]];
            }

            let yi = f64::from(self.y[i]);

            for j in 0..self.n {
                del_f[j] += -yi * self.features[[i, j]] / (1.0 + f64::exp(yi * wx));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn new() {
        let n = 10;
        let y = Array::zeros(n);
        let features = Array::zeros((n, 1));
        let regression = Regression::new(y, features.clone());
        //let x_0 = Array1::zeros(());
        assert_eq!(6.931471805599453, regression.loss(features.view()));
    }

    #[test]
    fn derivative() {
        let _n = 2;
        let y = arr1(&[-1, -1]);
        let features = arr2(&[[1.0, 0.0], [1.0, 0.0]]);
        let mut regression = Regression::new(y, features);
        let mut del_f = Array1::zeros(regression.n);

        regression.derivative(del_f.view_mut());

        assert_eq!(arr1(&[1.0, 0.0]), &del_f);
    }
}
