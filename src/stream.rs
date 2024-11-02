use std::cell::RefCell;
use std::error::Error;
use std::fmt::Debug;
use std::ops::Add;
use std::rc::Rc;
use geo::Geometry;
use ndarray::Array2;
use rand_distr::num_traits::real::Real;
use rerun::{LineStrips2D, Points2D, RecordingStream, TensorData, TextLog};
use serde::de::StdError;
use crate::constants::DOWNSCALING;

// #[derive(PartialEq, Eq, Ord, PartialOrd, Debug)]
// enum StreamLevel {
//     Debug,
//     Info,
//     Warning,
//     Error,
//     Critical
// }
//
// pub struct LeveledRecordingStream {
//     stream: RecordingStream,
//     level: StreamLevel
// }
//
// impl LeveledRecordingStream {
//     pub fn new(stream: RecordingStream, level: StreamLevel) -> Self {
//         Self {
//             stream,
//             level
//         }
//     }
// }
//


pub struct ChunkedRerunStream{
    pub stream: RecordingStream,
    chunk_size: usize,
    skip_text_messages: bool,
    point_queue: RefCell<Vec<[f32;2]>>,
    linestrip_queue: RefCell<Vec<Vec<[f32;2]>>>,
    path: RefCell<String>
}

impl ChunkedRerunStream {
    pub fn new(stream: RecordingStream, chunk_size: usize, skip_text_messages: bool) -> Self {
        Self {
            stream,
            chunk_size,
            skip_text_messages,
            point_queue: RefCell::new(vec![]),
            linestrip_queue: RefCell::new(vec![]),
            path: RefCell::new("".to_string())
        }
    }

    pub fn flush(&self) {
        if !self.point_queue.borrow().is_empty() {
            self.stream.log(&*self.path.take(), &Points2D::new(self.point_queue.take())).unwrap();
            self.point_queue.replace(vec![]);
        }
        if !self.linestrip_queue.borrow().is_empty() {
            self.stream.log(&*self.path.take(), &LineStrips2D::new(self.linestrip_queue.take())).unwrap();
            self.point_queue.replace(vec![]);
        }
    }
}

impl Stream for Option<Rc<ChunkedRerunStream>> {
    fn stream_message(&self, path: &str, message: &str) {
        ()
    }

    fn stream_geometry(&self, path: &str, geom: &Geometry<f64>) {
        match self {
            Some(stream) => {
                if stream.point_queue.borrow().len() + stream.linestrip_queue.borrow().len() >= stream.chunk_size || path != *stream.path.borrow()  {
                    stream.flush();
                    stream.path.replace(path.to_string());
                }
                match geom {
                    Geometry::Point(p) => {
                        let point: [f32; 2] = [p.x() as f32 / RASTER_CELL_SIZE, p.y() as f32 / RASTER_CELL_SIZE];
                        stream.point_queue.borrow_mut().push(point)
                    }
                    Geometry::Line(l) => {
                        let strip = [[l.start.x as f32 / RASTER_CELL_SIZE, l.start.y as f32 / RASTER_CELL_SIZE], [l.end.x as f32 / RASTER_CELL_SIZE, l.end.y as f32 / RASTER_CELL_SIZE]];
                        stream.linestrip_queue.borrow_mut().push(strip.to_vec())
                    }
                    Geometry::Polygon(p) => {
                        let mut linestrips: Vec<Vec<[f32;2]>> = vec![];
                        let mut points = vec![];
                        for point in p.exterior().points_iter() {
                            points.push([point.x() as f32 / RASTER_CELL_SIZE, point.y() as f32 / RASTER_CELL_SIZE]);
                        }
                        linestrips.push(points);

                        for interior in p.interiors() {
                            let mut points = vec![];
                            for point in interior.points_iter() {
                                points.push([point.x() as f32 / RASTER_CELL_SIZE, point.y() as f32 / RASTER_CELL_SIZE]);
                            }
                            linestrips.push(points);
                        }

                        stream.linestrip_queue.borrow_mut().extend(linestrips)
                    }
                    _ => unimplemented!("Geometry type not implemented")
                }
            },
            None => {}
        }
    }

    fn stream_raster<T>(&self, path: &str, raster: T)
    where
        T: TryInto<TensorData> + Clone + Debug,
        <T as TryInto<TensorData>>::Error: Error
    {
        match self {
            Some(stream) => stream.stream.stream_raster(path, raster),
            None => {}
        }
    }

    fn stream_scalar(&self, path: &str, scalar: f64) {
        match self {
            Some(stream) => stream.stream.stream_scalar(path, scalar),
            None => {}
        }
    }
}

pub const RASTER_CELL_SIZE: f32 = 100. * DOWNSCALING as f32;

impl Stream for Option<Rc<RecordingStream>> {
    fn stream_message(&self, path: &str, message: &str) {
        match self {
            Some(stream) => stream.stream_message(path, message),
            None => {}
        }
    }

    fn stream_geometry(&self, path: &str, geom: &Geometry<f64>) {
        match self {
            Some(stream) => stream.stream_geometry(path, geom),
            None => {}
        }
    }

    fn stream_raster<T>(&self, path: &str, raster: T)
    where
        T: TryInto<rerun::TensorData> + Clone + std::fmt::Debug, <T as TryInto<rerun::TensorData>>::Error: StdError
    {
        match self {
            Some(stream) => stream.stream_raster(path, raster),
            None => {}
        }
    }

    fn stream_scalar(&self, path: &str, scalar: f64) {
        match self {
            Some(stream) => stream.log(path, &rerun::Scalar::new(scalar)).unwrap(),
            None => {}
        }
    }
}

pub trait Stream {
    fn stream_message(&self, path: &str, message: &str);

    fn stream_geometry(&self, path: &str, geom: &Geometry<f64>);

    fn stream_raster<T>(&self, path: &str, raster: T)
    where
        T: TryInto<rerun::TensorData> + Clone  + std::fmt::Debug, <T as TryInto<rerun::TensorData>>::Error: StdError
    ;
    fn stream_scalar(&self, path: &str, scalar: f64);
}

impl Stream for RecordingStream {
    fn stream_message(&self, path: &str, message: &str) {
        self.log(path, &TextLog::new(message)).unwrap();
    }

    fn stream_scalar(&self, path: &str, scalar: f64) {
        self.log(path, &rerun::Scalar::new(scalar)).unwrap();
    }

    fn stream_geometry(&self, path: &str, geom: &Geometry<f64>) {
        match geom {
            Geometry::Point(p) => {
                let point: [f32; 2] = [p.x() as f32 / RASTER_CELL_SIZE, p.y() as f32 / RASTER_CELL_SIZE];
                self.log(path, &Points2D::new([point])).unwrap();
            }
            Geometry::Line(l) => {
                let strip = [
                    [l.start.x as f32 / RASTER_CELL_SIZE, l.start.y as f32 / RASTER_CELL_SIZE],
                    [l.end.x as f32 / RASTER_CELL_SIZE, l.end.y as f32 / RASTER_CELL_SIZE]
                ];
                self.log(path, &LineStrips2D::new([strip])).unwrap();
            }
            Geometry::Polygon(p) => {
                let mut linestrips = vec![];
                let mut points = vec![];
                for point in p.exterior().points_iter() {
                    points.push([point.x() as f32 / RASTER_CELL_SIZE, point.y() as f32 / RASTER_CELL_SIZE]);
                }
                linestrips.push(points);
                for interior in p.interiors() {
                    let mut points = vec![];
                    for point in interior.points_iter() {
                        points.push([point.x() as f32 / RASTER_CELL_SIZE, point.y() as f32 / RASTER_CELL_SIZE]);
                    }
                    linestrips.push(points);
                }


                self.log(path, &LineStrips2D::new(linestrips)).unwrap();
            }
            _ => unimplemented!("Geometry type not implemented")
        }

    }

    fn stream_raster<T>(&self, path: &str, raster: T)
    where
        T: TryInto<rerun::TensorData> + Clone + std::fmt::Debug, <T as TryInto<rerun::TensorData>>::Error: StdError
    {
        let annotation = rerun::AnnotationContext::new([
            // Adding the colours from the first set
            (0, "0", rerun::Rgba32::from_rgb(240, 200, 8)),   // "#f0c808"
            (1, "1", rerun::Rgba32::from_rgb(93, 75, 32)),    // "#5d4b20"
            (10, "10", rerun::Rgba32::from_rgb(70, 147, 116)),// "#469374"
            (11, "11", rerun::Rgba32::from_rgb(147, 65, 179)),// "#9341b3"
            (12, "12", rerun::Rgba32::from_rgb(227, 66, 125)),// "#e3427d"
            (2, "2", rerun::Rgba32::from_rgb(230, 134, 83)),  // "#e68653"
            (3, "3", rerun::Rgba32::from_rgb(154, 21, 0)),    // "#9a1500"
            (4, "4", rerun::Rgba32::from_rgb(38, 255, 0)),    // "#26ff00"
            (5, "5", rerun::Rgba32::from_rgb(58, 86, 230)),   // "#3a56e6"
            (6, "6", rerun::Rgba32::from_rgb(0, 157, 255)),   // "#009dff"
            (7, "7", rerun::Rgba32::from_rgb(3, 92, 0)),      // "#035c00"
            (8, "8", rerun::Rgba32::from_rgb(214, 225, 59)),  // "#d6e13b"
            (9, "9", rerun::Rgba32::from_rgb(40, 227, 211)),  // "#28e3d3"

            // Adding the colours from the second set
            (14, "apartments", rerun::Rgba32::from_rgb(147, 65, 179)),      // "#9341b3"
            (15, "big_commercial", rerun::Rgba32::from_rgb(49, 72, 220)),   // "#3148dc"
            (16, "complex", rerun::Rgba32::from_rgb(7, 240, 53)),           // "#07f035"
            (17, "detached", rerun::Rgba32::from_rgb(155, 30, 20)),         // "#9b1e14"
            (18, "filled_block", rerun::Rgba32::from_rgb(230, 134, 83)),    // "#e68653"
            (19, "industrial", rerun::Rgba32::from_rgb(93, 75, 32)),        // "#5d4b20"
            (20, "irregular_block", rerun::Rgba32::from_rgb(16, 223, 255)), // "#10dfff"
            (21, "perimeter_block", rerun::Rgba32::from_rgb(214, 224, 72)), // "#d6e048"
            (22, "terraced", rerun::Rgba32::from_rgb(70, 147, 116)),        // "#469374"

            (99, "None", rerun::Rgba32::from_rgb(0, 0, 0)),
        ]);

        self.log_static("/", &annotation).unwrap();

        let image = rerun::SegmentationImage::try_from(raster.clone()).unwrap();
        self.log(path, &image.with_opacity(0.5)).unwrap();
    }
}