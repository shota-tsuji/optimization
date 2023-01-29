use ndarray::{Array, Array1, Array2};

pub fn load_csv(path: String) -> (Array1<i8>, Array2<f64>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&path)
        .unwrap();
    let n = rdr.records().count();
    let mut y = Array::zeros(n);
    let mut features = Array::zeros((n, 123));

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&path)
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

    (y, features)
}
