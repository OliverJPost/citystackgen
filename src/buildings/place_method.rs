use std::collections::VecDeque;
use std::panic::catch_unwind;
use std::rc::Rc;
use geo::{Area, BooleanOps, BoundingRect, Centroid, Contains, Coordinate, EuclideanDistance, EuclideanLength, GeometryCollection, Intersects, Line, LineInterpolatePoint, LineString, MinimumRotatedRect, MultiPolygon, Point, Polygon, Rect, Relate, Rotate, Scale, Translate};
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
use rand::Rng;

type RecordingStream = ChunkedRerunStream;

const AREA_THRESHOLD: f64 = 100.;

pub struct BuildingSystemBuilderPlace {
    template: Template,
    rerun_stream: Option<Rc<RecordingStream>>
}

impl BuildingSystemBuilderPlace {
    pub fn new(template: Template) -> Self {
        Self {
            template,
            rerun_stream: None
        }
    }
}

pub fn generate_buildings_place(enclosure: &Polygon, stream: &Option<Rc<ChunkedRerunStream>>) -> Vec<Polygon> {
    let square_polygon_footprint = Polygon::new(
        LineString::from(vec![
            (0., 0.),
            (0., 10.),
            (10., 10.),
            (10., 0.),
            (0., 0.)
        ]),
        vec![]
    );
    let bbox = square_polygon_footprint.bounding_rect().unwrap();
    //let buffered_bbox = Rect::new(bbox.min().translate(-5., -5.), bbox.max().translate(5., 5.));
    let footprint_anchor = Point::new(5., -3.);

    let mut enclosure_buildings = vec![];
    let perimeter_length = enclosure.exterior().euclidean_length();
    let mut current_offset = 0.0;
    let mut rng = rand::thread_rng();

    'outer: while current_offset < perimeter_length {
        let fraction = current_offset / perimeter_length;
        let current_point = enclosure.exterior().line_interpolate_point(fraction).unwrap();
        let current_angle = angle_at_fraction(enclosure.exterior(), fraction);
        let offset_x = current_point.x() - footprint_anchor.x();
        let offset_y = current_point.y() - footprint_anchor.y();
        // Randomly scale footprint
        let scale = rng.gen_range(0.5..4.0);
        let chosen_building = square_polygon_footprint.scale_around_point(scale, scale, footprint_anchor);
        let chosen_bbox = chosen_building.bounding_rect().unwrap();
        let building = chosen_building.translate(offset_x, offset_y).rotate_around_point(current_angle.to_degrees(), current_point);
        if !enclosure.contains(&building) {
            current_offset += 3.;
            stream.stream_geometry("building/rejected_enclosure", &building.clone().into());
            continue;
        }
        for other_building in &enclosure_buildings {
            if building.intersects(other_building) {
                current_offset += 1.;
                stream.stream_geometry("building/rejected_intersects", &building.clone().into());
                continue 'outer;
            }
        }

        stream.stream_geometry("building/accepted", &building.clone().into());
        enclosure_buildings.push(building);
        current_offset += chosen_bbox.width() + 2.;
    }
    enclosure_buildings
}

impl BuildingSystemBuilder for BuildingSystemBuilderPlace {
    fn generate_features(&self, typology_grid: &Box<dyn TypologyGrid>, existing_layers: &Vec<Box<dyn CityLayer>>) -> Vec<Polygon> {
        let enclosures_layer = &existing_layers[2];
        let rng = &mut rand::thread_rng();
        let mut all_buildings = vec![];
        println!("Generating buildings...");
        for enclosure in enclosures_layer.get_features() {
            let enclosure = Polygon::try_from(enclosure).unwrap();
            let enclosure_buildings = generate_buildings_place(&enclosure, &self.rerun_stream);
            all_buildings.extend(enclosure_buildings);
        }
        println!("Generated {} buildings", all_buildings.len());
        all_buildings
    }
}

fn angle_at_fraction(line_string: &LineString, fraction: f64) -> f64 {
    let scale = line_string.euclidean_length();
    let one_meter_fraction = (1.0 / scale).min(1.0);
    let current_point = line_string.line_interpolate_point(fraction).unwrap();
    let next_point = line_string.line_interpolate_point(fraction + one_meter_fraction).unwrap();
    let angle = (next_point.y() - current_point.y()).atan2(next_point.x() - current_point.x());
    angle
}

impl CityLayerBuilder for BuildingSystemBuilderPlace {


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