
#[derive(Debug)]
pub enum Aggregator {
    Sum,
    Average,
    // WeightedAverage(&'a Vec<f64>),
    Max,
    Min,
    Count,
    Std
}

impl Aggregator {
    pub(crate) fn aggregate(&self, data: &Vec<f64>, distances: &Vec<f64>, step: f64, max_distance: f64) -> Vec<f64> {
        let bin_count = ((max_distance + 0.1) / step).ceil() as usize;
        let mut bins = vec![0.0; bin_count];
        match self {
            Aggregator::Sum => {
                for (distance, value) in distances.iter().zip(data) {
                    let bin = (distance / step).floor() as usize;
                    bins[bin] += value;
                }
                bins
            }
            Aggregator::Average => {
                let mut counts = vec![0.0; bin_count];
                for (distance, value) in distances.iter().zip(data) {
                    let bin = (distance / step).floor() as usize;
                    bins[bin] += value;
                    counts[bin] += 1.0;
                }
                bins.iter().zip(counts.iter()).map(|(bin, count)| if *count != 0.0 { bin / count } else { 0.0 }).collect()
            }
            // Aggregator::WeightedAverage(weights) => {
            //     let mut counts = vec![0.0; bin_count];
            //
            //     for ((distance, value), weight) in distances.iter().zip(data).zip(*weights) {
            //         let bin = (distance / step).floor() as usize;
            //         bins[bin] += value * weight;
            //         counts[bin] += weight;
            //     }
            //     bins.iter().zip(counts.iter()).map(|(bin, count)| if *count != 0.0 { bin / count } else { 0.0 }).collect()
            // }
            Aggregator::Max => {
                let mut max_value = 0.0;
                for (distance, value) in distances.iter().zip(data) {
                    let bin = (distance / step).floor() as usize;
                    if *value > max_value {
                        bins[bin] = *value;
                        max_value = *value;
                    }
                }
                bins
            }
            Aggregator::Min => {
                panic!()
            }
            Aggregator::Count => {
                for distance in distances {
                    let bin = (distance / step).floor() as usize;
                    bins[bin] += 1.0;
                }
                bins
            }
            Aggregator::Std => {
                let mut counts = vec![0.0; bin_count];
                let mut sums = vec![0.0; bin_count];
                for (distance, value) in distances.iter().zip(data) {
                    let bin = (distance / step).floor() as usize;
                    bins[bin] += value;
                    counts[bin] += 1.0;
                }
                let mean = bins.iter().zip(counts.iter()).map(|(bin, count)| if *count != 0.0 { bin / count } else { 0.0 }).collect::<Vec<f64>>();
                for (distance, value) in distances.iter().zip(data) {
                    let bin = (distance / step).floor() as usize;
                    sums[bin] += (value - mean[bin]).powi(2);
                }
                sums.iter().zip(counts.iter()).map(|(sum, count)| if *count != 0.0 { (sum / count).sqrt() } else { 0.0 }).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use super::*;

    #[test]
    fn test_aggregate_sum() {
        let aggregator = Aggregator::Sum;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let distances = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let result = aggregator.aggregate(&data, &distances, 1.0, 4.0);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let distances = vec![0.0, 0.1, 2.0, 3.0, 4.0];
        let result = aggregator.aggregate(&data, &distances, 1.0, 4.0);
        assert_eq!(result, vec![3.0, 0.0, 3.0, 4.0, 5.0]);
        let result = aggregator.aggregate(&data, &distances, 1.0, 6.0);
        assert_eq!(result, vec![3.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0]);
    }

    #[test]
    fn test_aggregate_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let distances = vec![0.0, 0.1, 2.0, 2.3, 0.4];
        let step = 1.0;
        let aggregator = Aggregator::Average;
        let result = aggregator.aggregate(&data, &distances, step, 4.0);
        assert_relative_eq!(result[0], 2.6666666, epsilon = 0.0001);
        assert_eq!(result[1], 0.0);
        assert_relative_eq!(result[2], 3.5);
        assert_eq!(result[3], 0.0);
        assert_eq!(result[4], 0.0);
    }

    #[test]
    fn test_aggregate_count() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let distances = vec![0.0, 0.1, 2.0, 2.3, 0.4];
        let step = 1.0;
        let aggregator = Aggregator::Count;
        let result = aggregator.aggregate(&data, &distances, step, 4.0);
        assert_eq!(result, vec![3.0, 0.0, 2.0, 0.0, 0.0]);
    }

}