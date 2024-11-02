use geo::Geometry;
use crate::model::grid::{TypologyGrid, TypologyGridX};
use crate::model::layer::CityLayer;

pub struct StreetNetworkLayer {
    network: crate::streets::streets_experiment::StreetNetwork,
    typology_grid: Box<dyn TypologyGrid>
}

impl StreetNetworkLayer {
    pub(crate) fn new(network: crate::streets::streets_experiment::StreetNetwork, typology_grid: Box<dyn TypologyGrid>) -> Self {
        Self {
            network,
            typology_grid
        }
    }
}

impl CityLayer for StreetNetworkLayer {
    fn to_geojson(&self) -> String {
        todo!();
        // let mut features = vec![];
        // for line in &self.network.segments.iter().map(|s| s.line).collect() {
        //     let poly = geojson::Value::from(line);
        //
        //     let feature = geojson::Feature {
        //         bbox: None,
        //         geometry: Some(geojson::Geometry::new(poly)),
        //         id: None,
        //         properties: None,
        //         foreign_members: None
        //     };
        //     features.push(feature);
        // }
        // let collection = geojson::FeatureCollection {
        //     bbox: None,
        //     features,
        //     foreign_members: None
        // };
        // collection.to_string()
    }

    fn get_features(&self) -> Vec<Geometry> {
        self.network.segments.iter().map(|segment| segment.line.clone().into()).collect()
    }
}