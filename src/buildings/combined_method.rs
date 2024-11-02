use std::collections::VecDeque;
use std::panic::catch_unwind;
use std::rc::Rc;
use geo::{Area, BooleanOps, BoundingRect, Centroid, Coordinate, EuclideanDistance, EuclideanLength, GeometryCollection, Intersects, Line, LineInterpolatePoint, LineString, MinimumRotatedRect, MultiPolygon, Point, Polygon, Relate, Translate};
use geo::line_intersection::line_intersection;
use geo::LineIntersection::SinglePoint;
use geos::Geom;
use geos::wkt::{ToWkt, Wkt};
use ndarray::{Array2, s};
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
use crate::stream::{ChunkedRerunStream, RASTER_CELL_SIZE, Stream};
use crate::streets::builder::StreetNetworkBuilder;
use crate::streets::layer::StreetNetworkLayer;
use geo_buffer::{buffer_multi_polygon, buffer_polygon};
use crate::buildings::inset_method::generate_buildings_inset;
use crate::buildings::oobb_method::generate_buildings_oobb;
use crate::buildings::place_method::generate_buildings_place;
use crate::buildings::sweep_method::generate_buildings_sweep;
use crate::constants::DOWNSCALING;
use crate::patchgridannealing;
use crate::patchgridannealing::{Aggregator, PatchGridMetricID};

type RecordingStream = ChunkedRerunStream;

#[derive(Debug)]
pub enum BuildingClass {
    Detached,
    Terraced,
    BigCommercial,
    Industrial,
    Apartments,
    FilledBlock,
    PerimeterBlock,
    IrregularBlock,
    Agriculture,
    Complex,
    Tower,
    None
}


impl BuildingClass {
    pub fn from_int(i: i32) -> Self {
        match i {
            0 => BuildingClass::Detached,
            1 => BuildingClass::Terraced,
            2 => BuildingClass::BigCommercial,
            3 => BuildingClass::Industrial,
            4 => BuildingClass::Apartments,
            5 => BuildingClass::FilledBlock,
            6 => BuildingClass::PerimeterBlock,
            7 => BuildingClass::IrregularBlock,
            8 => BuildingClass::Agriculture,
            9 => BuildingClass::Complex,
            10 => BuildingClass::Tower,
            99 => BuildingClass::None,
            14 => BuildingClass::Apartments,
            15 => BuildingClass::BigCommercial,
            16 => BuildingClass::Complex,
            17 => BuildingClass::Detached,
            18 => BuildingClass::FilledBlock,
            19 => BuildingClass::Industrial,
            20 => BuildingClass::IrregularBlock,
            21 => BuildingClass::PerimeterBlock,
            22 => BuildingClass::Terraced,
            -2 => BuildingClass::None,
            _ => panic!("Unknown building class {}", i)
        }
    }

    pub fn generate_buildings(&self, enclosure: &Polygon, stream: &Option<Rc<ChunkedRerunStream>>) -> Vec<Polygon> {
        match self {
            BuildingClass::FilledBlock => generate_buildings_oobb(enclosure, stream),
            BuildingClass::Apartments | BuildingClass::Agriculture | BuildingClass::Complex | BuildingClass::BigCommercial | BuildingClass::Detached | BuildingClass::Industrial | BuildingClass::Tower => generate_buildings_place(enclosure, stream),
            BuildingClass::Terraced => generate_buildings_sweep(enclosure, stream),
            BuildingClass::IrregularBlock => vec!(),
            BuildingClass::PerimeterBlock => generate_buildings_inset(enclosure, stream),
            BuildingClass::None => vec!()
        }
    }
}


pub struct BuildingSystemBuilderCombined {
    template: Template,
    rerun_stream: Option<Rc<RecordingStream>>,
    annealing: bool
}

impl BuildingSystemBuilderCombined {
    pub fn new(template: Template, no_annealing: bool) -> Self {
        Self {
            template,
            rerun_stream: None,
            annealing: !no_annealing
        }
    }
}

impl BuildingSystemBuilder for BuildingSystemBuilderCombined {
    fn generate_typology_grid(&self, existing_layers: &Vec<Box<dyn CityLayer>>) -> Box<dyn TypologyGrid> {
        let array;
        if DOWNSCALING > 1 {
            array = self.template.classes_buildings.slice(s![..;DOWNSCALING, ..;DOWNSCALING]).to_owned();
        } else {
            array = self.template.classes_buildings.clone();
        }
        let city_center_pt: Point = Point::try_from(existing_layers[0].get_features()[0].clone()).unwrap();
        let city_center = (city_center_pt.x(), city_center_pt.y());

        if !self.annealing {
            let base = array.clone() + 1;
            self.rerun_stream.stream_raster("buildings/typology_grid", base);
            return Box::new(TypologyGridX{class_grid: array})
        }

        let statistics = patchgridannealing::PatchGridStatistics::new()
            .with_metric(Box::new(patchgridannealing::metrics::PatchArea::new()))
            .with_metric(Box::new(patchgridannealing::metrics::PatchDistanceToCenter::new(city_center, 100.0 * DOWNSCALING as f64)))
            .with_metric(Box::new(patchgridannealing::metrics::PatchCoreAreaIndex::new()))
            .with_metric(Box::new(patchgridannealing::metrics::PatchShapeIndex::new()))
            ;

        let grid = patchgridannealing::PatchGrid::init(array.clone(), statistics);
        println!("Unique values {:?}", grid.class_ids());

        let statistics = patchgridannealing::PatchGridStatistics::new()
            .with_metric(Box::new(patchgridannealing::metrics::PatchArea::new()))
            .with_metric(Box::new(patchgridannealing::metrics::PatchDistanceToCenter::new(city_center, 100.0 * DOWNSCALING as f64)))
            .with_metric(Box::new(patchgridannealing::metrics::PatchCoreAreaIndex::new()))
            .with_metric(Box::new(patchgridannealing::metrics::PatchShapeIndex::new()))
            ;

        let copy_grid = patchgridannealing::PatchGrid::init(array, statistics);
        let target = copy_grid.statistics;
        let acceptor = patchgridannealing::Acceptor::MetrolopolisCriteria;
        let cooler = patchgridannealing::CoolingSchedule::Exponential{cooling_rate: 0.99999};

        let evaluator = patchgridannealing::GridStatisticsComparer::from_target(target, patchgridannealing::Comparer::Intersection)
            .with_binned_comparison(PatchGridMetricID::PatchArea, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Average)
            .with_binned_comparison(PatchGridMetricID::PatchArea, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Sum)
            .with_binned_comparison(PatchGridMetricID::PatchArea, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Count)
            //.with_binned_comparison(PatchGridMetricID::PatchArea, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Std)
            .with_binned_comparison(PatchGridMetricID::PatchCoreAreaIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Average)
            //.with_binned_comparison(PatchGridMetricID::PatchCoreAreaIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Std)
            .with_binned_comparison(PatchGridMetricID::PatchShapeIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Average)
            //.with_binned_comparison(PatchGridMetricID::PatchShapeIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Std)
            .with_binned_comparison(PatchGridMetricID::PatchCoreAreaIndex, PatchGridMetricID::PatchArea, Aggregator::Average)
            ;

        let rc = Rc::clone(&self.rerun_stream.as_ref().unwrap());
        let annealer = patchgridannealing::PatchGridAnnealer::new(grid, Box::new(evaluator), acceptor, cooler, 1000., 0.000000001, rc);
        let mut rng = rand::thread_rng();
        let new_grid = annealer.anneal(&mut rng);

        let array = new_grid.class_grid;
        let base1 = array.clone();
        self.rerun_stream.stream_raster("buildings/typology_grid", base1 + 1);
        Box::new(TypologyGridX{class_grid: array.clone()})
    }

    fn generate_features(&self, typology_grid: &Box<dyn TypologyGrid>, existing_layers: &Vec<Box<dyn CityLayer>>) -> Vec<Polygon> {
        let enclosures_layer = &existing_layers[2];
        let mut all_buildings = vec![];
        let class_grid = typology_grid.get_class_grid();
        println!("Generating buildings...");
        for enclosure in enclosures_layer.get_features() {
            let enclosure = Polygon::try_from(enclosure).unwrap();
            let class = class_grid.get(((enclosure.centroid().unwrap().y() / RASTER_CELL_SIZE as f64) as usize, (enclosure.centroid().unwrap().x() / RASTER_CELL_SIZE as f64) as usize)).unwrap();
            let bclass = BuildingClass::from_int(*class);
            println!("Building class {:?}", &bclass);
            if enclosure.unsigned_area() < 10000. {
                let enclosure_buildings = bclass.generate_buildings(&enclosure, &self.rerun_stream);
                all_buildings.extend(enclosure_buildings)
            }

        }
        println!("Generated {} buildings", all_buildings.len());
        all_buildings
    }
}


impl CityLayerBuilder for BuildingSystemBuilderCombined {


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