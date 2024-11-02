use std::rc::Rc;
use geo::Point;
use ndarray::{Array2, s};
use crate::model::grid::{TypologyGrid, TypologyGridX};
use crate::model::layer::{CityLayer, CityLayerBuilder};
use crate::model::template::Template;
use image::io::Reader as ImageReader;
use crate::constants::DOWNSCALING;
use crate::model::domain::Domain;
use crate::patchgridannealing;
use crate::patchgridannealing::{Aggregator, PatchGrid, PatchGridMetricID};
use crate::stream::{ChunkedRerunStream, Stream};
use crate::streets::layer::StreetNetworkLayer;
use crate::streets::streets_experiment::{StreetNetwork, StreetNetworkBuilderX};

type RecordingStream = ChunkedRerunStream;

pub struct StreetNetworkBuilder {
    domain: Domain,
    template: Template,
    rerun_stream: Option<Rc<RecordingStream>>,
    annealing: bool,
}

impl StreetNetworkBuilder {
    pub fn new(template: Template, domain: Domain, no_annealing: bool) -> Self {
        Self {
            domain,
            template,
            rerun_stream: None,
            annealing: !no_annealing
        }
    }
    fn generate_typology_grid(&self, existing_layers: &Vec<Box<dyn CityLayer>>) -> Box<dyn TypologyGrid>{
        // let img = ImageReader::open("/Users/ole/Downloads/rijssen.tiff").unwrap().decode().unwrap().into_luma8();
        // let (width, height) = img.dimensions();
        // let mut array = Array2::<i32>::zeros((height as usize, width as usize));
        // let mut i = 0;
        // for pixel in img.pixels() {
        //     let x = i % width;
        //     let y = i / width;
        //     array[[y as usize, x as usize]] = pixel.0[0] as i32;
        //     i += 1;
        // }
        let city_center_pt: Point = Point::try_from(existing_layers[0].get_features()[0].clone()).unwrap();
        let city_center = (city_center_pt.x(), city_center_pt.y());
        let array;
        if DOWNSCALING > 1 {
            array = self.template.clusters_street.slice(s![..;DOWNSCALING, ..;DOWNSCALING]).to_owned();
        } else {
           array = self.template.clusters_street.clone();
        }
        if !self.annealing {
            let base = array.clone();
            self.rerun_stream.stream_raster("streets/typology_grid", base + 1);
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
            //.with_binned_comparison(PatchGridMetricID::PatchArea, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Max)
            //.with_binned_comparison(PatchGridMetricID::PatchArea, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Std)
            .with_binned_comparison(PatchGridMetricID::PatchCoreAreaIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Average)
            .with_binned_comparison(PatchGridMetricID::PatchCoreAreaIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Max)
            //.with_binned_comparison(PatchGridMetricID::PatchCoreAreaIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Std)
            .with_binned_comparison(PatchGridMetricID::PatchShapeIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Average)
            //.with_binned_comparison(PatchGridMetricID::PatchShapeIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Std)
            .with_binned_comparison(PatchGridMetricID::PatchCoreAreaIndex, PatchGridMetricID::PatchArea, Aggregator::Average)
            ;

        let rc = Rc::clone(&self.rerun_stream.as_ref().unwrap());
        let annealer = patchgridannealing::PatchGridAnnealer::new(grid, Box::new(evaluator), acceptor, cooler, 100000., 0.000000001, rc); // FIXME final temperature
        let mut rng = rand::thread_rng();
        let new_grid = annealer.anneal(&mut rng);

        // let x = SegmentationImage::try_from(array.clone());
        // dbg!(x);
        let base1 = new_grid.class_grid.clone();
        self.rerun_stream.stream_raster("streets/typology_grid", base1 + 1);
        Box::new(TypologyGridX{class_grid: new_grid.class_grid})
    }

    fn generate_features(&self, typology_grid: &Box<dyn TypologyGrid>, existing_layers: &Vec<Box<dyn CityLayer>>) -> StreetNetwork {
        let city_center_pt: Point = Point::try_from(existing_layers[0].get_features()[0].clone()).unwrap();
        let mut builder = StreetNetworkBuilderX::new(&typology_grid, city_center_pt, self.domain.clone(), self.template.clone());

        if let Some(rerun_stream) = &self.rerun_stream {
            builder = builder.with_rerun_stream(Rc::clone(rerun_stream));
        }

        builder.build(30000)
    }

}

impl CityLayerBuilder for StreetNetworkBuilder {

    fn with_rerun_stream(mut self: Box<Self>, stream: Rc<RecordingStream>) -> Box<dyn CityLayerBuilder> {
        self.rerun_stream = Some(stream);
        self
    }

    fn with_progress_bar(&self) {
        todo!()
    }

    fn build(&self, existing_layers: &Vec<Box<dyn CityLayer>>) -> Box<dyn CityLayer> {
        let typology_grid = self.generate_typology_grid(&existing_layers);
        let network = self.generate_features(&typology_grid, &existing_layers);
        Box::new(StreetNetworkLayer::new(network, typology_grid))
    }
}