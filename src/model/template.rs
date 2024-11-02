use std::collections::HashMap;
use std::fs::File;
use ndarray::Array2;
use ndarray_npy::NpzReader;
use anyhow::Result;
use serde::{Deserialize, Deserializer, Serialize};
use std::io::Read;
use glob::glob;
use crate::statistics::distributions::Distribution;
use crate::statistics::histogram::{ProbabilityHistogram, StreetInfluenceMap};


#[derive(Clone, Deserialize, Debug)]
pub struct ClusterStreetData{
    pub forward_angle: ProbabilityHistogram<StreetInfluenceMap>,
    pub segment_length: ProbabilityHistogram<StreetInfluenceMap>,
    pub next_node_degree: ProbabilityHistogram<StreetInfluenceMap>
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Metric {
    metric: String,
    pub distribution: Distribution
}

#[derive(Clone)]
pub struct Template{
    pub clusters_street: Array2<i32>,
    pub clusters_street_data: HashMap<i32, ClusterStreetData>,
    pub classes_buildings: Array2<i32>,
    pub city_center: (usize, usize)
}

impl Template {
    pub fn from_file(filename: &str, cluster_directory: &str) -> Result<Self> {
        let mut npz = NpzReader::new(File::open(filename)?)?;
        let clusters_street: Array2<i32> = npz.by_name("cluster_street.npy")?;
        let classes_buildings: Array2<i32> = npz.by_name("building_class.npy")?;
        let city_center_grid: Array2<i32> = npz.by_name("city_center.npy")?;
        // City center is the only cell with value 1
        let city_center = city_center_grid.indexed_iter().find(|(_, &value)| value == 1).unwrap().0;


        let mut clusters_street_data = HashMap::new();

        let pattern = format!("{}{}", cluster_directory, "/cluster_*.json");

        for entry in glob(&pattern).expect("Failed to read glob pattern") {
            match entry {
                Ok(path) => {
                    let file_name = path.file_stem().unwrap().to_str().unwrap();
                    let number: i32 = file_name.split('_').nth(1).unwrap().parse().unwrap();

                    let mut file = File::open(&path)?;
                    let mut contents = String::new();
                    file.read_to_string(&mut contents)?;

                    let data: ClusterStreetData = serde_json::from_str(&contents).unwrap();

                    println!("File: {:?}, Number: {}, Data: {:?}", path, number, data);
                    clusters_street_data.insert(number, data);
                },
                Err(e) => println!("{:?}", e),
            }
        }
        Ok(Template{clusters_street, clusters_street_data, classes_buildings, city_center})
    }
}