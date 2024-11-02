use std::collections::VecDeque;
use std::panic::catch_unwind;
use std::rc::Rc;
use geo::{Area, BooleanOps, BoundingRect, Centroid, Contains, Coordinate, EuclideanDistance, EuclideanLength, GeometryCollection, Intersects, Line, LineInterpolatePoint, LineString, MinimumRotatedRect, MultiPolygon, Orient, Point, Polygon, Rect, Relate, Rotate, Scale, Simplify, Translate};
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
use geo_clipper::{Clipper, EndType, JoinType};
use offset_polygon::offset_polygon;
use rand::Rng;

type RecordingStream = ChunkedRerunStream;

const AREA_THRESHOLD: f64 = 100.;

pub struct BuildingSystemBuilderSweep {
    template: Template,
    rerun_stream: Option<Rc<RecordingStream>>
}

impl BuildingSystemBuilderSweep {
    pub fn new(template: Template) -> Self {
        Self {
            template,
            rerun_stream: None
        }
    }
}

pub fn generate_buildings_sweep(enclosure: &Polygon, stream: &Option<Rc<ChunkedRerunStream>>) -> Vec<Polygon> {
    let square_polygon_footprint = Polygon::new(
        LineString::from(vec![
            (0., 0.),
            (0., 10.),
            (6., 10.),
            (6., 0.),
            (0., 0.)
        ]),
        vec![]
    );
    let bbox = square_polygon_footprint.bounding_rect().unwrap();
    //let buffered_bbox = Rect::new(bbox.min().translate(-5., -5.), bbox.max().translate(5., 5.));
    let footprint_anchor = Point::new(3., -3.);

    let mut enclosure_buildings = vec![];
    let simplified_enclosure = enclosure.exterior().simplify(&1.0);
    for segment in simplified_enclosure.lines() {
        let segment_length = segment.euclidean_length();
        if segment_length < 10. {
            continue;
        }
        let segment_angle = (segment.end.y - segment.start.y).atan2(segment.end.x - segment.start.x);
        let mut current_offset = 0.0;
        'outer: while current_offset < segment_length {
            let fraction = current_offset / segment_length;
            let current_point = segment.line_interpolate_point(fraction).unwrap();
            let building = square_polygon_footprint.translate(current_point.x() - footprint_anchor.x(), current_point.y() - footprint_anchor.y()).rotate_around_point(segment_angle.to_degrees(), current_point);

            if !enclosure.contains(&building) {
                current_offset += 3.;
                stream.stream_geometry("building/rejected_enclosure", &building.clone().into());
                continue;
            }
            // let inset_building = buffer_polygon(&building.orient(Direction::Default), -1.0);//.offset(-1.0, JoinType::Square, EndType::ClosedPolygon, 1.0);
            // for inset in &inset_building {
            //     self.rerun_stream.stream_geometry("building/inset", &inset.clone().into());
            // }

            for other_building in &enclosure_buildings {
                if building.relate(other_building).is_overlaps() && !touches(&building, other_building) {
                    current_offset += 3.;
                    stream.stream_geometry("building/rejected_intersects", &building.clone().into());
                    continue 'outer;
                }
            }

            stream.stream_geometry("building/accepted", &building.clone().into());
            enclosure_buildings.push(building);
            current_offset += 6.;
        }
    }
    enclosure_buildings
}

impl BuildingSystemBuilder for BuildingSystemBuilderSweep {
    fn generate_features(&self, typology_grid: &Box<dyn TypologyGrid>, existing_layers: &Vec<Box<dyn CityLayer>>) -> Vec<Polygon> {
        let enclosures_layer = &existing_layers[2];
        let rng = &mut rand::thread_rng();
        let mut all_buildings = vec![];
        println!("Generating buildings...");
        for enclosure in enclosures_layer.get_features() {
            let enclosure = Polygon::try_from(enclosure).unwrap();
            let enclosure_buildings = generate_buildings_sweep(&enclosure, &self.rerun_stream);
            all_buildings.extend(enclosure_buildings);
        }
        println!("Generated {} buildings", all_buildings.len());
        all_buildings
    }
}

fn touches(building1: &Polygon, building2: &Polygon) -> bool {
    let building1_points = building1.exterior().points_iter().collect::<Vec<_>>();
    let building2_points = building2.exterior().points_iter().collect::<Vec<_>>();
    let mut touching_points = 0;
    for i in 0..building1_points.len() {
        for j in 0..building2_points.len() {
            if building1_points[i].euclidean_distance(&building2_points[j]) < 0.1 {
                touching_points += 1;
            }
        }
    }
    touching_points > 1
}

fn angle_at_fraction(line_string: &LineString, fraction: f64) -> f64 {
    let scale = line_string.euclidean_length();
    let one_meter_fraction = (1.0 / scale).min(1.0);
    let current_point = line_string.line_interpolate_point(fraction).unwrap();
    let next_point = line_string.line_interpolate_point(fraction + one_meter_fraction).unwrap();
    let angle = (next_point.y() - current_point.y()).atan2(next_point.x() - current_point.x());
    angle
}

impl CityLayerBuilder for BuildingSystemBuilderSweep {


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