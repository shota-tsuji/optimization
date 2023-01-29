use ndarray::Array1;

use optimization::regression::logistic as lg;
use optimization::regression::Regression;
use optimization::utils;

fn main() {
    // divide to labels and samples
    let path = String::from("./dummy.csv");
    let (y, mat_x) = utils::load_csv(path);
    // unique
    // index array for value == -1
    let y_nega_idx: Vec<usize> = y
        .iter()
        .enumerate()
        .filter(|(_, &val)| val == -1)
        .map(|(i, _)| i)
        .collect();
    // ones-array
    // zero with index-array
    let mut y_bin = Array1::ones(y.len());
    for i in y_nega_idx {
        y_bin[i] = 0;
    }

    let logistic = lg::Logistic::new(y_bin, mat_x.clone());
    let mut regression = Regression {};
    if let Ok(state) = regression.train(logistic) {
        println!("{:#?}", state.termination_reason);
    }
}
