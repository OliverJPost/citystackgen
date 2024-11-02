use geo::{Geometry, Polygon};
use crate::model::grid::TypologyGrid;
use crate::model::layer::CityLayer;

pub struct BuildingSystemLayer {
    buildings: Vec<Polygon>,
    typology_grid: Box<dyn TypologyGrid>
}

impl BuildingSystemLayer {
    pub fn new(buildings: Vec<Polygon>, typology_grid: Box<dyn TypologyGrid>) -> Self {
        Self {
            buildings,
            typology_grid
        }
    }
}

impl CityLayer for BuildingSystemLayer {
    fn to_geojson(&self) -> String {
        todo!();
        // let mut features = vec![];
        // for building in self.buildings {
        //     let poly = geojson::Value::from(building);
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
        self.buildings.iter().map(|poly| poly.clone().into()).collect()
    }
}