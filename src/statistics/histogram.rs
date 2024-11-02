use std::collections::HashMap;
use std::rc::Rc;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::ThreadRng;
use rand_chacha::ChaCha8Rng;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use struct_iterable::Iterable;
use crate::stream::{ChunkedRerunStream, Stream};

type RecordingStream = ChunkedRerunStream;

#[derive(Deserialize, Clone, Debug)]
struct Bin<T: InfluenceMap> {
    bounds: (f64, f64),
    median: f64,
    probability: f64,
    influences: T
}

impl<T: InfluenceMap> Bin<T> {
    fn get_influence(&self, values: &T::InfluenceValuesType) -> f64 {
        self.influences.get_weight(values)
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct ProbabilityHistogram<T: InfluenceMap> {
    bins: Vec<Bin<T>>
}

pub trait InfluenceMap {
    type InfluenceValuesType: InfluenceValues;

    fn get_influence_distributions(&self, name: &str) -> &crate::statistics::distributions::Distribution;

    fn get_weight(&self, values: &Self::InfluenceValuesType) -> f64 {
        let mut total_probability = 0.0;
        for (influence_name, influence_value) in values.items() {
            let distribution = self.get_influence_distributions(influence_name);
            total_probability += distribution.probability(influence_value);
        }
        total_probability / values.count() as f64
    }
}

pub trait InfluenceValues {
    // TODO make actual iterator
    fn items(&self) -> Vec<(&str, f64)>;
    fn count(&self) -> usize;
}

#[derive(Deserialize, Clone, Debug)]
pub struct StreetInfluenceMap {
    previous_segment_forward_angle: crate::statistics::distributions::Distribution,
    previous_segment_length: crate::statistics::distributions::Distribution,
    distance_to_last_intersection: crate::statistics::distributions::Distribution
}

impl InfluenceMap for StreetInfluenceMap {
    type InfluenceValuesType = StreetInfluenceValues;

    fn get_influence_distributions(&self, name: &str) -> &crate::statistics::distributions::Distribution {
        match name {
            "previous_segment_forward_angle" => &self.previous_segment_forward_angle,
            "previous_segment_length" => &self.previous_segment_length,
            "distance_to_last_intersection" => &self.distance_to_last_intersection,
            _ => {panic!("Could not find distribution {:?}", name)}
        }
    }
}

pub struct StreetInfluenceValues {
    previous_segment_forward_angle: f64,
    previous_segment_length: f64,
    distance_to_last_intersection: f64
}

impl StreetInfluenceValues {
    pub fn new(previous_segment_forward_angle: f64, previous_segment_length: f64, distance_to_last_intersection: f64) -> Self {
        StreetInfluenceValues { previous_segment_forward_angle, previous_segment_length, distance_to_last_intersection}
    }
}


impl InfluenceValues for StreetInfluenceValues {
    fn items(&self) -> Vec<(&str, f64)> {
        vec![
            ("previous_segment_forward_angle", self.previous_segment_forward_angle)
        ]
    }

    fn count(&self) -> usize {
        1
    }
}



pub enum BinSampleMethod {
    Median
}

impl BinSampleMethod{
    pub fn sample<T: InfluenceMap>(&self, bin: &Bin<T>) -> f64 {
        match self {
            BinSampleMethod::Median => {bin.median}
        }
    }
}

impl<T: InfluenceMap> ProbabilityHistogram<T> {
    pub fn sample(&self, influences: &T::InfluenceValuesType, sample_method: &BinSampleMethod, mut rng: &mut ChaCha8Rng, rec: &Option<Rc<RecordingStream>>) -> f64
    {
        let mut bin_weights = vec![];
        let mut inf = vec![];
        for bin in &self.bins {
            let influence = bin.get_influence(&influences);
            let bin_weight = bin.probability * influence;
            inf.push(influence);
            bin_weights.push(bin_weight);
        }
        rec.stream_message("bins", &self.bins.iter().map(|x| x.bounds.0.to_string()).collect::<Vec<String>>().join(","));
        rec.stream_message("sampled bin weights", &bin_weights.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(","));
        rec.stream_message("sampled influences", &inf.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(","));
        let index = WeightedIndex::new(&bin_weights).unwrap();
        let chosen_bin = &self.bins[index.sample(&mut rng)];
        sample_method.sample(chosen_bin)
    }
}