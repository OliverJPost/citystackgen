use std::rc::Rc;
use geo::Polygon;
use ndarray::Array2;
use crate::model::grid::{TypologyGrid, TypologyGridX};
use crate::model::layer::{CityLayer, CityLayerBuilder};
use crate::plots::layer::PlotSystemLayer;
use crate::plots::voronoi_method::VoronoiPlotSystemBuilder;

pub trait BuildingSystemBuilder: CityLayerBuilder {
    fn generate_typology_grid(&self, existing_layers: &Vec<Box<dyn CityLayer>>) -> Box<dyn TypologyGrid>{
        let mut array = Array2::<i32>::zeros((200, 200));


        Box::new(TypologyGridX{class_grid: array})
    }

    fn generate_features(&self, typology_grid: &Box<dyn TypologyGrid>, existing_layers: &Vec<Box<dyn CityLayer>>) -> Vec<Polygon>;
}

