use std::collections::VecDeque;
use std::panic::catch_unwind;
use std::rc::Rc;
use geo::{Area, BooleanOps, BoundingRect, Centroid, Coordinate, EuclideanDistance, EuclideanLength, GeometryCollection, Intersects, Line, LineInterpolatePoint, LineString, MinimumRotatedRect, MultiPolygon, Point, Polygon, Relate, Translate};
use geo::line_intersection::line_intersection;
use geo::LineIntersection::SinglePoint;
use geos::Geom;
use geos::wkt::{ToWkt, Wkt};
use ndarray::Array2;
use ordered_float::Pow;
use petgraph::visit::Walker;
//use rerun::{RecordingStream, TextLog};
use voronoice::{BoundingBox, VoronoiBuilder};
use crate::buildings::builder::BuildingSystemBuilder;
use crate::buildings::layer::BuildingSystemLayer;
use crate::model::grid::{TypologyGrid, TypologyGridX};
use crate::model::layer::{CityLayer, CityLayerBuilder};
use crate::model::template::Template;
use crate::plots::layer::PlotSystemLayer;
use crate::stream::{ChunkedRerunStream, Stream};
use crate::streets::builder::StreetNetworkBuilder;
use crate::streets::layer::StreetNetworkLayer;
use geo_buffer::{buffer_multi_polygon, buffer_polygon};

type RecordingStream = ChunkedRerunStream;

const AREA_THRESHOLD: f64 = 100.;

pub struct BuildingSystemBuilderInset {
    template: Template,
    rerun_stream: Option<Rc<RecordingStream>>
}

impl BuildingSystemBuilderInset {
    pub fn new(template: Template) -> Self {
        Self {
            template,
            rerun_stream: None
        }
    }
}

impl BuildingSystemBuilder for BuildingSystemBuilderInset {
    fn generate_features(&self, typology_grid: &Box<dyn TypologyGrid>, existing_layers: &Vec<Box<dyn CityLayer>>) -> Vec<Polygon> {
        let enclosures_layer = &existing_layers[2];
        let mut all_buildings = vec![];
        println!("Generating buildings...");
        for enclosure in enclosures_layer.get_features() {
            let enclosure = &Polygon::try_from(enclosure).unwrap();
            let buildings = generate_buildings_inset(enclosure, &self.rerun_stream);
            all_buildings.extend(buildings);

        }
        println!("Generated {} buildings", all_buildings.len());
        all_buildings
    }
}

pub fn generate_buildings_inset(enclosure: &Polygon, stream: &Option<Rc<ChunkedRerunStream>>) -> Vec<Polygon> {
    let building = buffer_polygon(enclosure, -4.);
    let mut buildings = vec!();
    for polygon in building {
        let holes = buffer_polygon(&polygon, -16.);
        for hole in &holes {
            stream.stream_geometry("building_hole", &hole.clone().into());
        }
        let interiors = holes.iter().map(|p| p.exterior().clone()).collect();
        let re = Polygon::new(polygon.exterior().clone(), interiors);
        stream.stream_geometry("building", &re.clone().into());
        buildings.push(re);
    }
    buildings
}

impl CityLayerBuilder for BuildingSystemBuilderInset {


    fn with_rerun_stream(mut self: Box<Self>, stream: Rc<RecordingStream>) -> Box<dyn CityLayerBuilder> {
        self.rerun_stream = Some(stream);
        self
    }

    fn with_progress_bar(&self) {
        todo!()
    }

    fn build(&self, existing_layers: &Vec<Box<dyn CityLayer>>) -> Box<dyn CityLayer> {
        let typology_grid = self.generate_typology_grid(&existing_layers);
        let buildings = self.generate_features(&typology_grid, &existing_layers);
        Box::new(BuildingSystemLayer::new(buildings, typology_grid))
    }
}