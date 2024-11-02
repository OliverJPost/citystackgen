use std::collections::VecDeque;
use std::panic::catch_unwind;
use std::rc::Rc;
use geo::{Area, BooleanOps, BoundingRect, Centroid, Coordinate, EuclideanDistance, EuclideanLength, GeometryCollection, Intersects, Line, LineInterpolatePoint, LineString, MinimumRotatedRect, Orient, Point, Polygon, Relate, Translate};
use geo::line_intersection::line_intersection;
use geo::LineIntersection::SinglePoint;
use geo::orient::Direction;
use geos::Geom;
use geos::wkt::{ToWkt, Wkt};
use ndarray::Array2;
use ordered_float::Pow;
use petgraph::visit::Walker;
//use rerun::{RecordingStream, TextLog};
use voronoice::{BoundingBox, VoronoiBuilder};
use crate::enclosures::builder::EnclosureBuilder;
use crate::model::grid::{TypologyGrid, TypologyGridX};
use crate::model::layer::{CityLayer, CityLayerBuilder};
use crate::model::template::Template;
use crate::plots::builder::PlotSystemBuilder;
use crate::plots::layer::PlotSystemLayer;
use crate::stream::{ChunkedRerunStream, Stream};
use crate::streets::builder::StreetNetworkBuilder;
use crate::streets::layer::StreetNetworkLayer;
type RecordingStream = ChunkedRerunStream;

const AREA_THRESHOLD: f64 = 100.;

pub struct EnclosureBuilderGEOS {
    template: Template,
    rerun_stream: Option<Rc<RecordingStream>>
}

impl EnclosureBuilderGEOS {
    pub fn new(template: Template) -> Self {
        Self {
            template,
            rerun_stream: None
        }
    }
}

impl EnclosureBuilder for EnclosureBuilderGEOS {
    fn generate_features(&self, typology_grid: &Box<dyn TypologyGrid>, existing_layers: &Vec<Box<dyn CityLayer>>) -> Vec<Polygon> {
        println!("Generating enclosures...");
        let street_segments = &existing_layers[1].get_features();
        let mut geometries: Vec<geos::Geometry> = vec![];
        for line in street_segments {
            let line_geo = Line::try_from(line.clone()).unwrap();
            let linestring = LineString::from(line_geo);
            let geom = geos::Geometry::try_from(&linestring).unwrap();
            //let geom = geos::Geometry::new_from_wkt(&line.wkt_string()).unwrap();
            geometries.push(geom);
        }
        let re = geos::Geometry::polygonize(&geometries).unwrap();

        // TODO why is this necessary?
        let wkt: Wkt<f64> = re.to_wkt().expect("Fail").parse().expect("Failed to parse WKT");
        let g: GeometryCollection = wkt.try_into().unwrap();

        let mut enclosures = vec![];
        for geom in g {
            let poly: Polygon = geom.try_into().unwrap();
            let poly = poly.orient(Direction::Default);
            self.rerun_stream.stream_geometry("enclosure", &poly.clone().into());
            enclosures.push(poly)
        }
        println!("Generated {} enclosures", enclosures.len());
        enclosures
    }

}


impl CityLayerBuilder for EnclosureBuilderGEOS {


    fn with_rerun_stream(mut self: Box<Self>, stream: Rc<RecordingStream>) -> Box<dyn CityLayerBuilder> {
        self.rerun_stream = Some(stream);
        self
    }

    fn with_progress_bar(&self) {
        todo!()
    }

    fn build(&self, existing_layers: &Vec<Box<dyn CityLayer>>) -> Box<dyn CityLayer> {
        let typology_grid = self.generate_typology_grid(&existing_layers);
        let plots = self.generate_features(&typology_grid, &existing_layers);
        Box::new(PlotSystemLayer::new(plots, typology_grid))
    }
}