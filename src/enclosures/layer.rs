use geo::{Geometry, Polygon};
use crate::model::grid::TypologyGrid;
use crate::model::layer::CityLayer;

pub struct EnclosureLayer {
    plots: Vec<Polygon>,
    typology_grid: Box<dyn TypologyGrid>
}

impl EnclosureLayer {
    pub fn new(plots: Vec<Polygon>, typology_grid: Box<dyn TypologyGrid>) -> Self {
        Self {
            plots,
            typology_grid
        }
    }
}

impl CityLayer for EnclosureLayer {
    fn to_geojson(&self) -> String {
        todo!();
        // let mut features = vec![];
        // for plot in self.plots {
        //     let poly = geojson::Value::from(plot.exterior());
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
        self.plots.iter().map(|poly| poly.clone().into()).collect()
    }
}