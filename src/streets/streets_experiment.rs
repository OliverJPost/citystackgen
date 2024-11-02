use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::f64::consts::PI;
use std::rc::Rc;
use geo::{BoundingRect, Centroid, Coord, EuclideanDistance, Geometry, GeometryCollection, Line, LineIntersection, MultiPolygon, Point, Polygon};
use geo::line_intersection::line_intersection;
use geojson::JsonObject;
use geos::{Geom, wkt};
use geos::wkt::{ToWkt, Wkt};
use ordered_float::OrderedFloat;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand::{Rng, thread_rng};
use rerun::{LineStrips2D, TextLog};
use rstar::{AABB, Envelope, PointDistance, RStarInsertionStrategy, RTree, RTreeObject, RTreeParams};
use serde::{Deserialize, Serialize};
use image::io::Reader as ImageReader;
use lox::core::HalfEdgeMesh;
use ndarray::Array2;
use nshare::ToNdarray2;
use plexus::FromRawBuffers;
use plexus::geometry::convert::IntoGeometry;
use plexus::graph::MeshGraph;
use plexus::primitive::Edge;
use rand::SeedableRng;
use rand_distr::num_traits::Pow;
use rerun::external::log::debug;
use uuid::Uuid;
use crate::model::domain::Domain;
use crate::model::grid::TypologyGrid;
use crate::model::template::Template;
use crate::statistics::histogram::{BinSampleMethod, StreetInfluenceValues};
use crate::stream::{ChunkedRerunStream, RASTER_CELL_SIZE, Stream};

type RecordingStream = ChunkedRerunStream;

const SNAP_DISTANCE: f64 = 1.;
const DEFAULT_LENGTH: f64 = 5.;
const ANGLE_STD: f64 = 1.;
const BRANCH_PROBABILITY: f64 = 0.5;

#[derive(Copy, Clone)]
struct Node{
    pt: Point
}
//
// struct LargeNodeParameters;
//
// impl RTreeParams for LargeNodeParameters
// {
//     const MIN_SIZE: usize = 25;
//     const MAX_SIZE: usize = 50;
//     const REINSERTION_COUNT: usize = 2;
//     type DefaultInsertionStrategy = RStarInsertionStrategy;
// }

impl Node {
    fn from_offset(start: &Node, angle_degrees: f64, distance: f64) -> Self {
        let angle_radians = angle_degrees * PI / 180.0;
        let x = start.pt.x() + distance * angle_radians.cos();
        let y = start.pt.y() + distance * angle_radians.sin();
        Node{pt: Point::new(x, y)}
    }

    fn get_angle_degrees(&self, other: &Self) -> f64 {
        (other.pt.y() - self.pt.y()).atan2(other.pt.x() - self.pt.x()) * 180.0 / PI
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
enum RoadType {
    Major,
    Street
}

pub(crate) struct StreetSegment {
    start_node: usize,
    end_node: usize,
    roadtype: RoadType,
    order: i64,
    pub(crate) line: Line,
    id: Uuid
}

impl Clone for StreetSegment {
    fn clone(&self) -> Self {
        StreetSegment {
            start_node: self.start_node,
            end_node: self.end_node,
            roadtype: self.roadtype,
            order: self.order,
            line: self.line,
            id: self.id
        }
    }
}

impl PartialEq for StreetSegment {
    fn eq(&self, other: &Self) -> bool {
        self.start_node == other.start_node && self.end_node == other.end_node
    }
}

impl StreetSegment {
    fn new(start_node: usize, end_node: usize, nodes: &Vec<Node>, road_type: RoadType, order: i64, id: Uuid) -> Self {
        StreetSegment {
            start_node: start_node,
            end_node: end_node,
            roadtype: road_type,
            line: Line::new(nodes[start_node].pt, nodes[end_node].pt),
            order,
            id
        }
    }

    fn update_line(&mut self, nodes: &Vec<Node>) {
        self.line = Line::new(nodes[self.start_node].pt, nodes[self.end_node].pt);
    }

    fn get_angle(&self, nodes: &Vec<Node>) -> f64 {
        nodes[self.start_node].get_angle_degrees(&nodes[self.end_node])
    }

    // Calculate length of the line
    fn get_length(&self, nodes: &Vec<Node>) -> f64 {
        nodes[self.start_node].pt.euclidean_distance(&nodes[self.end_node].pt)
    }

    // Get the middle point of the line
    fn get_middle(&self) -> Point {
        self.line.centroid()
    }

    fn intersection(&self, other: &Self, nodes: &Vec<Node>) -> Option<Coord> {
        match line_intersection(self.line, other.line) {
            Some(LineIntersection::SinglePoint {intersection, is_proper}) => {
                if is_proper {
                    Some(intersection)
                } else {
                    None
                }
            },
            Some(LineIntersection::Collinear{intersection}) => panic!("Collinear lines"),
            None => None
        }
    }

    fn as_rerun(&self, nodes: &Vec<Node>) -> LineStrips2D {
        let start = nodes[self.start_node].pt;
        let end = nodes[self.end_node].pt;
        let strip = [[start.x() as f32, start.y() as f32], [end.x() as f32, end.y() as f32]];
        LineStrips2D::new([strip.to_vec()])//.with_labels(vec![format!("id: {}", self.id)])
    }
}

impl RTreeObject for StreetSegment {
    type Envelope = AABB<Coord>;

    fn envelope(&self) -> Self::Envelope {
        let rect = self.line.bounding_rect();
        AABB::from_corners(rect.min(), rect.max())
    }
}

impl PointDistance for StreetSegment {
    fn distance_2(&self, point: &Coord) -> f64 {
        self.line.euclidean_distance(point)
    }
}

struct SegmentQuery {
    cost: i64,
    start_node: usize,
    end_node: Option<usize>,
    angle_degrees: f64,
    length: f64,
    was_truncated: bool,
    roadtype: RoadType,
    id: Uuid,
    previous_segment_forward_angle: f64,
    distance_to_last_intersection: f64
}

impl SegmentQuery {
    fn new(cost: i64, start_node: usize, angle_degrees: f64, length: f64, roadtype: RoadType, previous_segment_forward_angle: f64, distance_to_last_intersection: f64) -> Self {
        Self{
            cost,
            start_node,
            end_node: None,
            angle_degrees,
            length,
            was_truncated: false,
            roadtype,
            id: Uuid::new_v4(),
            previous_segment_forward_angle,
            distance_to_last_intersection
        }
    }

    fn to_segment(mut self, mut nodes: &mut Vec<Node>) -> StreetSegment {
        self.generate_end(nodes);
        StreetSegment::new(self.start_node, self.end_node.unwrap(), nodes, self.roadtype, self.cost, self.id)
    }

    fn generate_end(&mut self, mut nodes: &mut Vec<Node> ) -> usize {
        match self.end_node {
            Some(node) => node,
            None => {
                let node =  Node::from_offset(&nodes[self.start_node], self.angle_degrees, self.length);
                nodes.push(node);
                let idx = nodes.len() - 1;
                self.end_node = Some(idx);
                idx
            }
        }
    }
    fn get_end_pt(&self, nodes: &Vec<Node> ) -> Point {
        match self.end_node {
            Some(node) => nodes[node].pt,
            None => {
                let node =  Node::from_offset(&nodes[self.start_node], self.angle_degrees, self.length);
                node.pt
            }
        }
    }

    fn as_line(&self, nodes: & Vec<Node>) -> Line {
        let end = self.get_end_pt(nodes);
        Line::new(nodes[self.start_node].pt, end)
    }

    fn update_from_local_constraints(&mut self, network: &mut StreetNetwork, stream: &Option<Rc<RecordingStream>>) {
        let closest_intersection_pt = network.closest_segment_intersection(&self, &stream);
        match closest_intersection_pt {
            (Some(point), Some(intersecting_segment)) => {
                self.length = network.nodes[self.start_node].pt.euclidean_distance(&point);
                let new_node = Node{pt: point};
                network.nodes.push(new_node);
                let new_node_idx = network.nodes.len() - 1;
                let new_left_segment = StreetSegment::new(intersecting_segment.start_node, new_node_idx, &network.nodes, intersecting_segment.roadtype, intersecting_segment.order, Uuid::new_v4());
                let new_right_segment = StreetSegment::new(new_node_idx, intersecting_segment.end_node, &network.nodes, intersecting_segment.roadtype, intersecting_segment.order, Uuid::new_v4());
                network.segments.remove(&intersecting_segment);
                network.add_segment(new_left_segment);
                network.add_segment(new_right_segment);

                println!("Truncated segment at intersection");
                stream.stream_message("local_constraints", &format!("Truncated segment {} at intersection to {:?}", self.id, self.length));
                self.was_truncated = true;
                return; // todo look into this
            },
            _ => ()
        }

        let closest_node = network.closest_node(&self);
        match closest_node {
            Some(node_idx) => {
                let node = &network.nodes[node_idx];
                let distance = node.pt.euclidean_distance(&self.get_end_pt(&network.nodes));
                if distance < SNAP_DISTANCE {
                    self.length = network.nodes[self.start_node].pt.euclidean_distance(&node.pt);
                    let old_angle = self.angle_degrees;
                    self.angle_degrees = network.nodes[self.start_node].get_angle_degrees(&node);
                    println!("Snapped segment to node");
                    stream.stream_message("local_constraints", &format!("Snapped segment {} to node. Old angle {} new {} ", self.id, old_angle, self.angle_degrees));
                    self.end_node = Some(node_idx);
                    self.was_truncated = true;
                }
            },
            _ => ()
        }
    }

    fn as_rerun(&self, nodes: &Vec<Node>) -> LineStrips2D {
        let end = self.get_end_pt(nodes);
        let start = nodes[self.start_node].pt;
        let strip = [[start.x() as f32, start.y() as f32], [end.x() as f32, end.y() as f32]];
        LineStrips2D::new([strip.to_vec()])//.with_labels(vec![format!("id: {}", self.id)])
    }
}

impl Eq for SegmentQuery {}

impl PartialEq<Self> for SegmentQuery {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl PartialOrd<Self> for SegmentQuery {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SegmentQuery {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}


pub(crate) struct StreetNetwork {
    pub segments: RTree<StreetSegment>,
    nodes: Vec<Node>
}

impl StreetNetwork {
    fn new() -> Self {
        Self{
            segments: RTree::new(),
            nodes: vec![]
        }
    }

    fn add_segment(&mut self, segment: StreetSegment) {
        self.segments.insert(segment);
    }
    fn closest_segment_intersection(&self, query: &SegmentQuery, stream: &Option<Rc<RecordingStream>>) -> (Option<Point>, Option<StreetSegment>) {
        // TODO rtree
        let segments_in_range = self.segments.locate_within_distance(query.get_end_pt(&self.nodes).into(), query.length);
        let rect = query.as_line(&self.nodes).bounding_rect();
        let line_bbox = AABB::from_corners(rect.min(), rect.max());
        let candidate_segments = self.segments.locate_in_envelope_intersecting(&line_bbox);
        let mut closest_intersection: Option<Point> = None;
        let mut closest_intersection_distance = f64::INFINITY;
        let mut closest_segment = None;
        let mut i = 0;
        for segment in candidate_segments {
            i += 1;
            let intersect = line_intersection(segment.line, query.as_line(&self.nodes));
            match intersect {
                Some(LineIntersection::SinglePoint {intersection, is_proper}) => {
                    if is_proper {
                        let intersection_pt: Point = intersection.into();
                        let distance = intersection_pt.euclidean_distance(&self.nodes[query.start_node].pt);
                        if distance < closest_intersection_distance {
                            closest_intersection = Some(intersection_pt);
                            closest_intersection_distance = distance;
                            closest_segment = Some(segment.clone());
                        }
                    }
                },
                _ => ()
            }
        }
        stream.stream_message("intersections", &format!("Intersection count: {:?}", i));
        (closest_intersection, closest_segment)
    }

    fn closest_node(&self, query: &SegmentQuery) -> Option<usize> {
        let segments_in_range = self.segments.locate_within_distance(query.get_end_pt(&self.nodes).into(), query.length);
        let mut closest_node = None;
        let mut closest_node_distance = f64::INFINITY;
        let end = query.get_end_pt(&self.nodes);
        for segment in segments_in_range {
            for node_idx in [segment.start_node, segment.end_node] {
                let node = &self.nodes[node_idx];
                let distance = end.euclidean_distance(&node.pt);
                if distance < closest_node_distance {
                    closest_node = Some(node_idx);
                    closest_node_distance = distance;
                }
            }
        }
        closest_node
    }

    fn to_plots(&self) -> Vec<Polygon> {
        let mut geometries: Vec<geos::Geometry> = vec![];
        for segment in &self.segments {
            let line = segment.line;
            // FIXME performance
            let geom = geos::Geometry::new_from_wkt(&line.wkt_string()).unwrap();
            geometries.push(geom);
        }
        let re = geos::Geometry::polygonize(&geometries).unwrap();
        let wkt: Wkt<f64> = re.to_wkt().expect("Fail").parse().expect("Failed to parse WKT");

        let g: GeometryCollection = wkt.try_into().unwrap();
        let mut polygons = vec![];
        for geom in g {
            let poly: Polygon = geom.try_into().unwrap();
            polygons.push(poly)
        }
        polygons
    }

    fn to_geojson_file(&self, fp: &str) {
        let mut features = vec![];
        for segment in &self.segments {
            let line = geojson::Value::from(&segment.line);
            let properties = serde_json::json!({
                "roadtype": segment.roadtype,
                "cost": segment.order
            });

            let feature = geojson::Feature {
                bbox: None,
                geometry: Some(geojson::Geometry::new(line)),
                id: None,
                properties: Some(properties.as_object().unwrap().clone()),
                foreign_members: None
            };
            features.push(feature);
        }
        let collection = geojson::FeatureCollection {
            bbox: None,
            features,
            foreign_members: None
        };
        let geojson_str = collection.to_string();
        std::fs::write(fp, geojson_str).unwrap();
    }
}

pub struct StreetNetworkBuilderX {
    network: StreetNetwork,
    queries: BinaryHeap<SegmentQuery>, // Min-heap due to Ord implementation on SegmentQuery
    tick_count: usize,
    stream: Option<Rc<RecordingStream>>,
    typology_grid: Array2<i32>,
    angle_std_map: HashMap<i32, f64>,
    domain: Domain,
    template: Template,
    rng: ChaCha8Rng
}

impl StreetNetworkBuilderX {
    pub fn new(typology_grid: &Box<dyn TypologyGrid>, center: Point, domain: Domain, template: Template) -> Self {
        let mut network = StreetNetwork::new();
        let start_point = Node{pt: center};
        let mut queries = BinaryHeap::new();
        network.nodes.push(start_point);
        let start_point_idx = network.nodes.len() -1;
        queries.push(SegmentQuery::new(0, start_point_idx, 90., DEFAULT_LENGTH, RoadType::Major, 0., DEFAULT_LENGTH));

        let array = typology_grid.get_class_grid();

        let mut angle_std_map: HashMap<i32, f64> = HashMap::new();
        angle_std_map.insert(-2, 0.);
        angle_std_map.insert(0, 0.);
        angle_std_map.insert(1, 40.);
        angle_std_map.insert(2, 20.);
        angle_std_map.insert(3, 10.);
        angle_std_map.insert(4, 0.);
        angle_std_map.insert(5, 2.5);
        angle_std_map.insert(99, 0.0);

        let seed: [u8; 32] = [42; 32];
        let random_seed = thread_rng().gen::<u8>();
        let seed = [random_seed; 32];
        let rng = ChaCha8Rng::from_seed(seed);
        Self{
            stream: None,
            network,
            queries,
            tick_count: 0,
            typology_grid: array.clone(),
            angle_std_map,
            domain,
            template,
            rng
        }
    }

    pub fn with_rerun_stream<'a>(mut self, stream: Rc<RecordingStream>) -> Self
    {
    self.stream = Some(stream);
    self
    }

    fn tick(&mut self) {
        if self.queries.is_empty() {
            println!("No more queries");
            return;
        }
        let mut rng = &mut self.rng;

        let mut query = self.queries.pop().unwrap();
        if !self.domain.is_inside(query.get_end_pt(&self.network.nodes)) {
            return;
        }
        let cost = query.cost;
        self.stream.stream_message("length", &format!("Length before {:?}.", query.length));
        query.update_from_local_constraints(&mut self.network, &self.stream);
        self.stream.stream_message("length", &format!("Length after {:?}.", query.length));
        let mut nodes = &mut self.network.nodes;
        let was_truncated = query.was_truncated;
        let previous_segment_forward_angle = query.previous_segment_forward_angle;
        let previous_segment_length = query.length;
        let distance_to_last_intersection = query.distance_to_last_intersection;
        let segment = query.to_segment(&mut nodes);
        let segment_angle = segment.get_angle(&nodes);
        let segment_type = segment.roadtype;
        let end_node = segment.end_node.clone();
        let id = segment.id;
        self.stream.stream_geometry("segments", &segment.line.into());
        self.network.add_segment(segment);
        let mut nodes = &mut self.network.nodes;

        if was_truncated {
            self.stream.stream_message("local_constraints", &format!("No new segments for {} because truncated.", id));
            return;
        }

        // Randomly choose from normal distribution
        let mut closest_class = *self.typology_grid.get(((nodes[end_node].pt.y() / RASTER_CELL_SIZE as f64) as usize, (nodes[end_node].pt.x() / RASTER_CELL_SIZE as f64) as usize)).unwrap_or(&99);
        //let closest_class = 2_i32;
        let mut none_zone = false;
        if closest_class.is_negative() || closest_class == 99 {
            closest_class = 10_i32;
            none_zone = true;
            // // TODO deal with NULL cells
            // return;
        }
        let mut re = self.template.clusters_street_data.get_mut(&(closest_class as i32));
        let cluster_data = match re {
            Some(data) => data,
            None => panic!("No cluster data for class {:?}", closest_class)
        };
        let influences = StreetInfluenceValues::new(previous_segment_forward_angle, previous_segment_length, distance_to_last_intersection);
        let median = &BinSampleMethod::Median;
        let mut intersection_degree = cluster_data.next_node_degree.sample(&influences, median, rng, &self.stream) as u32;
        if intersection_degree == 1 {
            return;
        }
        if previous_segment_length < 12. && intersection_degree > 2 || none_zone {
            intersection_degree = 2; // Hardcode to prevent unrealistic patterns caused by parking lot statistics
        }
        let new_forward_angle = cluster_data.forward_angle.sample(&influences, median, rng, &self.stream);
        let new_angle = segment_angle + new_forward_angle;
        let new_length: f64 = cluster_data.segment_length.sample(&influences, median, rng, &self.stream);

        let new_distance_to_last_intersection;
        if intersection_degree > 2 {
            new_distance_to_last_intersection = new_length;
        } else {
            new_distance_to_last_intersection = distance_to_last_intersection + new_length;
        }

        let forward_query = SegmentQuery::new(cost + 1, end_node, new_angle, new_length.abs().min(150.0), segment_type, new_forward_angle, new_distance_to_last_intersection);
        //self.stream.stream_geometry("queries", &forward_query.as_line(&nodes).into());
        self.queries.push(forward_query);

        let build_left; let build_right;
        if intersection_degree == 3 {
            build_left = rng.gen::<f64>() < 0.5;
            build_right = !build_left;
        } else if intersection_degree >=4 {
            build_left = true; build_right = true;
        } else {
            build_left = false; build_right = false;
        }

        let left_segment_type = RoadType::Major; // todo
        if build_left {
            let delay = cost + 5;
            let left_angle = segment_angle + 90.;
            let left_length = cluster_data.segment_length.sample(&influences, median, &mut rng, &self.stream);
            let left_query = SegmentQuery::new(delay, end_node.clone(), left_angle, left_length.abs().min(150.), left_segment_type, new_forward_angle, left_length);
            //self.stream.stream_geometry("queries", &left_query.as_line(&nodes).into());
            self.queries.push(left_query);
        }

        let right_segment_type = RoadType::Major; // todo
        if build_right {
            let delay = cost + 5;
            let right_angle = segment_angle - 90.;
            let right_length = cluster_data.segment_length.sample(&influences, median, &mut rng, &self.stream);
            let right_query = SegmentQuery::new(delay, end_node.clone(), right_angle, right_length.abs().min(150.), right_segment_type, new_forward_angle, right_length);
            //self.stream.stream_geometry("queries", &right_query.as_line(&nodes).into());
            self.queries.push(right_query);
        }
    }

    fn simulate(&mut self, n_ticks: usize) {
        for _ in 0..n_ticks {
            self.tick();
            self.tick_count += 1;
            println!("Tick: {}", self.tick_count);
        }
    }

    pub fn build(mut self, n_ticks: usize) -> StreetNetwork {
        self.simulate(n_ticks);
        self.network
    }
}


pub(crate) fn pt_to_vpt(pt: Point) -> voronoice::Point {
    voronoice::Point{x: pt.x(), y: pt.y()}
}

pub(crate) fn co_to_vpt(co: Coord) -> voronoice::Point {
    voronoice::Point{x: co.x, y: co.y}
}

pub(crate) fn vpt_to_co(pt: &voronoice::Point) -> Coord {
    Coord{x: pt.x, y: pt.y}
}

fn vpt_to_rerun(pt: &voronoice::Point) -> rerun::Points2D {
    rerun::Points2D::new(vec![[pt.x as f32, pt.y as f32]])
}

fn pt_to_rerun(pt: Point) -> rerun::Points2D {
    rerun::Points2D::new(vec![[pt.x() as f32, pt.y() as f32]])
}

pub(crate) fn vpts_to_rerun(pts: &Vec<voronoice::Point>) -> rerun::Points2D {
    let mut points = vec![];
    for pt in pts {
        points.push([pt.x as f32, pt.y as f32]);
    }
    rerun::Points2D::new(points)
}

fn pts_to_rerun(pts: &Vec<Point>) -> rerun::Points2D {
    let mut points = vec![];
    for pt in pts {
        points.push([pt.x() as f32, pt.y() as f32]);
    }
    rerun::Points2D::new(points)
}

fn polys_to_rerun(polys: &[Polygon]) -> rerun::LineStrips2D {
    let mut linestrips = vec![];
    for poly in polys {
        let mut points = vec![];
        for point in poly.exterior().points_iter() {
            points.push([point.x() as f32, point.y() as f32]);
        }
        linestrips.push(points);
    }
    rerun::LineStrips2D::new(linestrips)
}
//
// #[cfg(test)]
// mod tests {
//     use geo::Geometry::Polygon;
//     use geo::{BooleanOps, EuclideanLength, LineInterpolatePoint};
//     use voronoice::{BoundingBox, VoronoiBuilder};
//     use super::*;
//
//     #[test]
//     fn test_builder() {
//
//         let mut builder = StreetNetworkBuilderX::new("/Users/ole/Downloads/rijssen.tiff", (38, 26));
//         builder.simulate(50000);
//         builder.network.to_geojson_file("streets.geojson");
//         let enclosures = builder.network.to_plots();
//         let rec = &builder.rec;
//         polygons_to_geojson(&enclosures, "enclosures.geojson");
//
//         let mut plots = vec![];
//         for enclosure in &enclosures {
//             let boundary = enclosure.exterior();
//             rec.log("vor/domain", &polys_to_rerun(&[enclosure.clone()])).unwrap();
//             let bbox = boundary.bounding_rect().unwrap();
//             let center =  co_to_vpt(bbox.center());
//             let width = bbox.width();
//             let height = bbox.height();
//             let bbox2 = BoundingBox::new(center, width, height);
//             let length = boundary.euclidean_length();
//             let plot_width = 20.1;
//             let steps = (length / plot_width).ceil() as usize;
//             let mut points = vec![];
//             for i in 0..steps {
//                 let point = boundary.line_interpolate_point(((i as f64) * plot_width) / length).unwrap();
//                 points.push(pt_to_vpt(point));
//             }
//             rec.log("vor/pts", &vpts_to_rerun(&points)).unwrap();
//
//             if points.len() < 2 {
//                 plots.push(enclosure.clone());
//                 rec.log("vor/log", &TextLog::new("Too few points")).unwrap();
//                 continue;
//             }
//
//             let re = std::panic::catch_unwind(||VoronoiBuilder::default()
//                 .set_sites(points)
//                 .set_bounding_box(bbox2)
//                 .build());
//
//             if let Ok(Some(voronoi)) = re {
//                 let mut polys = vec![];
//                 let v = voronoi.vertices();
//                 for cell in voronoi.cells() {
//                     let poly = geo::Polygon::new(cell.iter().map(|idx| vpt_to_co(&v[*idx])).collect(), vec![]);
//                     // intersection with enclosure
//                     let r = std::panic::catch_unwind(||poly.intersection(enclosure).0[0].clone());
//                     match r {
//                         Ok(poly) => {
//                             polys.push(poly);
//                         },
//                         Err(_) => {
//                             println!("Failed to create voronoi due to panic");
//                             rec.log("vor/log", &TextLog::new("Failed to create voronoi. Panics!")).unwrap();
//                         }
//                     }
//                 }
//                 plots.extend(polys.clone());
//                 rec.log("vor/cells", &polys_to_rerun(&polys)).unwrap();
//             } else {
//                 plots.push(enclosure.clone());
//                 rec.log("vor/log", &TextLog::new("Failed to create voronoi")).unwrap();
//                 println!("Failed to create voronoi");
//             }
//         }
//         polygons_to_geojson(&plots, "plots.geojson");
//     }
//
//     fn polygons_to_geojson(polys: &Vec<geo::Polygon>, fp: &str) {
//         let mut features = vec![];
//         for plot in polys {
//             let poly = geojson::Value::from(plot);
//
//             let feature = geojson::Feature {
//                 bbox: None,
//                 geometry: Some(geojson::Geometry::new(poly)),
//                 id: None,
//                 properties: None,
//                 foreign_members: None
//             };
//             features.push(feature);
//         }
//         let collection = geojson::FeatureCollection {
//             bbox: None,
//             features,
//             foreign_members: None
//         };
//         let geojson_str = collection.to_string();
//         std::fs::write(fp, geojson_str).unwrap();
//     }
// }