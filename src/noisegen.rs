#![allow(unused)]
use core::f32;

use noise::{NoiseFn, OpenSimplex};
use rand::{RngCore, SeedableRng};
use sha2::{Sha256, digest::Update, Digest};
use egui::*;
use splines::{Key, Spline};

// let mut hasher = Sha256::default();
// Digest::update(&mut hasher, b"Hello, world");
// let result = hasher.finalize();
// let mut buffer = [0u8; 32];
// buffer.copy_from_slice(&result);
// let mut rng = rand::rngs::StdRng::from_seed(buffer);
// println!("Seed1: {}", rng.next_u64());
// println!("Seed2: {}", rng.next_u64());
// println!("Seed3: {}", rng.next_u64());
// println!("Seed4: {}", rng.next_u64());
fn make_seed<T: AsRef<[u8]>>(bytes: T) -> u32 {
    let mut hasher = Sha256::default();
    Digest::update(&mut hasher, bytes.as_ref());
    let result = hasher.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&result);
    let mut rng = rand::rngs::StdRng::from_seed(seed);
    rng.next_u32()
}

fn octave_noise(noise_fn: &OpenSimplex, point: Point, octaves: u32, persistence: f64, scale: f64) -> f64 {
    let mut total = 0.0;
    let mut frequency = scale;
    let mut amplitude = 1.0;
    let mut max_value = 1.0;
    for _ in 0..octaves {
        let noise_value = noise_fn.get([point.x * frequency, point.y * frequency]);
        total += noise_value * amplitude;
        max_value += amplitude;
        frequency *= 2.0;
        amplitude *= persistence;
    }
    total / max_value
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Point {
        Self {
            x,
            y
        }
    }

    pub fn dot(self, other: Point) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn normalized(self) -> Self {
        let magnitude = self.magnitude();
        Self::new(self.x / magnitude, self.y / magnitude)
    }

    pub fn magnitude(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn check_between(self, min: Self, max: Self) -> Option<f64> {
        let ab = max - min;
        let ap = self - min;
        let ab_dot_ab = ab.dot(ab);
        let ap_dot_ab = ap.dot(ab);
        let t = ap_dot_ab / ab_dot_ab;
        if 0.0 <= t && t <= 1.0 {
            Some(t)
        } else {
            None
        }
    }

    pub fn distance_between(self, a: Point, b: Point) -> f64 {
        let ray = (b - a).normalized();
        let point = self - a;
        point.dot(ray)
    }

    pub fn check_between_closest(self, min: Self, max: Self) -> Option<Self> {
        let ab = max - min;
        let ap = self - min;
        let ab_dot_ab = ab.dot(ab);
        let ap_dot_ab = ap.dot(ab);
        let t = ap_dot_ab / ab_dot_ab;
        if 0.0 <= t && t <= 1.0 {
            Some(min + t * ab)
        } else {
            None
        }
    }
}


impl std::ops::Add<Point> for Point {
    type Output = Self;
    fn add(self, rhs: Point) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::Mul<Point> for f64 {
    type Output = Point;
    fn mul(self, rhs: Point) -> Self::Output {
        Point::new(self * rhs.x, self * rhs.y)
    }
}

impl std::ops::Mul<f64> for Point {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl std::ops::Sub<Point> for Point {
    type Output = Self;
    fn sub(self, rhs: Point) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y
        }
    }
}

#[cfg(test)]
mod testing_sandbox {
    use itertools::Itertools;

    // TODO: Remove this sandbox when it is no longer in use.
    use super::*;
    #[test]
    fn sandbox() {
        // let seed = make_seed("");
        // println!("{seed}");
        
        let keys = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 5.0),
            Point::new(3.0, 0.0),
            Point::new(4.0, 8.0)
        ];
        let point = Point::new(2.5, 5.0);
        let mut found: Option<usize> = None;
        for i in 1..keys.len() {
            let first = keys[i-1];
            let second = keys[i];
            if point.x >= first.x && point.x < second.x {
                found = Some(i);
                break;
            }
        }
        println!("{found:?}");
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimplexGui {
    enabled: bool,
    seed: String,
    intervals: Vec<NoiseGenIntervalGui>,
}

impl Default for SimplexGui {
    fn default() -> Self {
        Self {
            enabled: true,
            seed: String::from(""),
            intervals: vec![NoiseGenIntervalGui::default()]
        }
    }
}

pub struct Simplex {
    simplex: OpenSimplex,
    intervals: Vec<NoiseGenInterval>,
}

impl From<SimplexGui> for Simplex {
    fn from(value: SimplexGui) -> Self {
        let seed = make_seed(value.seed);
        let simplex = OpenSimplex::new(seed);
        Self {
            simplex,
            intervals: value.intervals.into_iter().filter(|interval| interval.enabled).map(NoiseGenInterval::from).collect()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
enum Interpolation {
    CatmullRom,
    Cosine,
    Linear,
}

impl Into<splines::Interpolation<f64, f64>> for Interpolation {
    fn into(self) -> splines::Interpolation<f64, f64> {
        match self {
            Interpolation::CatmullRom => splines::Interpolation::CatmullRom,
            Interpolation::Cosine => splines::Interpolation::Cosine,
            Interpolation::Linear => splines::Interpolation::Linear,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub enum NoiseBound {
    /// Clamp to value
    Clamp(f64),
    /// When value is beyond threshold, cutoff to zero.
    Cutoff(f64),
    /// Clamp to value but also normalize range.
    Range(f64),
}

impl NoiseBound {
    pub fn value(self) -> f64 {
        match self {
            NoiseBound::Clamp(value) => value,
            NoiseBound::Cutoff(value) => value,
            NoiseBound::Range(value) => value,
        }
    }

    pub fn value_mut(&mut self) -> &mut f64 {
        match self {
            NoiseBound::Clamp(value) => value,
            NoiseBound::Cutoff(value) => value,
            NoiseBound::Range(value) => value,
        }
    }

    pub fn make_clamp(&mut self) {
        *self = match *self {
            NoiseBound::Clamp(value) => NoiseBound::Clamp(value),
            NoiseBound::Cutoff(value) => NoiseBound::Clamp(value),
            NoiseBound::Range(value) => NoiseBound::Clamp(value),
        }
    }

    pub fn make_cutoff(&mut self) {
        *self = match *self {
            NoiseBound::Clamp(value) => NoiseBound::Cutoff(value),
            NoiseBound::Cutoff(value) => NoiseBound::Cutoff(value),
            NoiseBound::Range(value) => NoiseBound::Cutoff(value),
        }
    }

    pub fn make_range(&mut self) {
        *self = match *self {
            NoiseBound::Clamp(value) => NoiseBound::Range(value),
            NoiseBound::Cutoff(value) => NoiseBound::Range(value),
            NoiseBound::Range(value) => NoiseBound::Range(value),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SplineGui {
    enabled: bool,
    keys: Vec<(f64, f64, Interpolation)>,
}

impl Default for SplineGui {
    fn default() -> Self {
        Self {
            enabled: false,
            keys: vec![
                (0., 0., Interpolation::CatmullRom),
                (0., 0., Interpolation::CatmullRom),
                (1., 1., Interpolation::CatmullRom),
                (1., 1., Interpolation::CatmullRom)
            ]
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NoiseGenIntervalGui {
    enabled: bool,
    spline: SplineGui,
    octaves: u32,
    persistence: f64,
    scale: f64,
    x_mult: f64,
    y_mult: f64,
    low: NoiseBound,
    high: NoiseBound,
}

impl Default for NoiseGenIntervalGui {
    fn default() -> Self {
        Self {
            enabled: true,
            spline: SplineGui::default(),
            octaves: 4,
            persistence: 0.5,
            scale: 0.5,
            x_mult: 0.1,
            y_mult: 0.1,
            low: NoiseBound::Clamp(0.),
            high: NoiseBound::Clamp(1.0),
        }
    }
}

pub struct NoiseGenInterval {
    keys: Option<Spline<f64, f64>>,
    octaves: u32,
    persistence: f64,
    scale: f64,
    x_mult: f64,
    y_mult: f64,
    low: NoiseBound,
    high: NoiseBound,
}

impl From<NoiseGenIntervalGui> for NoiseGenInterval {
    fn from(value: NoiseGenIntervalGui) -> Self {
        Self {
            keys: if value.spline.enabled {
                Some(Spline::from_iter(value.spline.keys.into_iter()
                    .map(|key| Key::new(key.0 as f64, key.0, key.2.into()))))
            } else {
                None
            },
            octaves: value.octaves,
            persistence: value.persistence,
            scale: value.scale,
            x_mult: value.x_mult,
            y_mult: value.y_mult,
            low: value.low,
            high: value.high
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NoiseGenGui {
    simplexes: Vec<SimplexGui>,
}

impl Default for NoiseGenGui {
    fn default() -> Self {
        Self {
            simplexes: vec![SimplexGui::default()],
        }
    }
}

pub struct NoiseGen {
    simplexes: Vec<Simplex>,
}

impl From<NoiseGenGui> for NoiseGen {
    fn from(value: NoiseGenGui) -> Self {
        Self {
            simplexes: value.simplexes.into_iter().filter(|simplex| simplex.enabled).map(|simplex| simplex.into()).collect()
        }
    }
}

impl NoiseGenInterval {
    pub fn sample(&self, simplex: &OpenSimplex, point: Point) -> f64 {
        let point = Point::new(point.x * self.x_mult, point.y * self.y_mult);
        let noise = octave_noise(simplex, point, self.octaves, self.persistence, self.scale);
        let gradient = (noise + 1.) * 0.5;
        let gradient = if let Some(keys) = &self.keys {
            keys.sample(gradient).expect("Failed to sample spline.")
        } else {
            gradient
        };
        use NoiseBound::*;
        match (self.low, self.high) {
            (Range(low), Range(high)) => {
                let clamped = gradient.max(low).min(high);
                let diff = high - low;
                let rel = clamped - low;
                rel * (1. / diff)
            }
            (Range(low), Clamp(high)) => {
                let clamped = gradient.max(low).min(high);
                let diff = 1.0 - low;
                let rel = clamped - low;
                rel * (1. / diff)
            }
            (Range(low), Cutoff(high)) => {
                if gradient > high {
                    0.0
                } else {
                    let clamped = gradient.max(low);
                    let diff = 1.0 - low;
                    let rel = clamped - low;
                    rel * (1. / diff)
                }
            }
            (Clamp(low), Range(high)) => {
                let clamped = gradient.max(low).min(high);
                clamped * (1. / high)
            }
            (Clamp(low), Clamp(high)) => {
                gradient.max(low).min(high)
            }
            (Clamp(low), Cutoff(high)) => {
                if gradient > high {
                    0.
                } else {
                    gradient.max(low)
                }
            }
            (Cutoff(low), Range(high)) => {
                if gradient < low {
                    0.
                } else {
                    let clamped = gradient.min(high);
                    clamped * (1. / high)
                }
            }
            (Cutoff(low), Clamp(high)) => {
                if gradient < low {
                    0.
                } else {
                    gradient.min(high)
                }
            }
            (Cutoff(low), Cutoff(high)) => {
                if gradient < low || gradient > high {
                    0.
                } else {
                    gradient
                }
            }
        }
    }
}

impl Simplex {
    pub fn sample(&self, point: Point) -> f64 {
        if self.intervals.is_empty() {
            return f64::NEG_INFINITY;
        }
        let (noise_accum, div) = self.intervals.iter().fold((0., 0.), |(accum, div), interval| {
            (accum + interval.sample(&self.simplex, point), div + 1.)
        });
        noise_accum / div
    }
}

impl NoiseGen {
    pub fn sample(&self, point: Point) -> f64 {
        if self.simplexes.is_empty() {
            return 0.0;
        }
        let (noise_accum, div) = self.simplexes.iter().fold((0., 0.), |(accum, div), simplex| {
            let sample = simplex.sample(point);
            if sample == f64::NEG_INFINITY {
                (accum, div)
            } else {
                (accum + simplex.sample(point), div + 1.)
            }
        });
        noise_accum / div
    }
}

const LABEL_WIDTH: f32 = 100.;

trait UiExt {
    fn labeled<R, Text: Into<widget_text::WidgetText>, F: FnMut(&mut Ui) -> R>(&mut self, label_width: f32, text: Text, add_contents: F) -> R;
}

impl UiExt for Ui {
    fn labeled<R, Text: Into<widget_text::WidgetText>, F: FnMut(&mut Ui) -> R>(&mut self, label_width: f32, text: Text, add_contents: F) -> R {
        let mut ui = add_contents;
        self.horizontal(|hor| {
            let (rect, _) = hor.allocate_exact_size(Vec2::new(label_width, hor.spacing().interact_size.y), Sense::hover());
            let label = egui::widgets::Label::new(text);
            hor.put(rect, label);
            ui(hor)
        }).inner
    }
}

trait BoolUiExt: Sized + Copy {
    fn ui_checkbox(&mut self, ui: &mut Ui, text: impl Into<WidgetText>) -> Response;

    fn ui_toggle(&mut self, ui: &mut Ui, text: impl Into<WidgetText>) -> Response;

    fn toggle(&mut self) -> bool;

    fn opt<T>(self, value: T) -> Option<T>;
    fn not_opt<T>(self, value: T) -> Option<T>;

    fn select<T>(self, _true: T, _false: T) -> T;
}

impl BoolUiExt for bool {
    fn toggle(&mut self) -> bool {
        let old = *self;
        *self = !old;
        old
    }
    
    fn select<T>(self, _true: T, _false: T) -> T {
        if self {
            _true
        } else {
            _false
        }
    }

    fn not_opt<T>(self, value: T) -> Option<T> {
        if self {
            None
        } else {
            Some(value)
        }
    }

    fn opt<T>(self, value: T) -> Option<T> {
        if self {
            Some(value)
        } else {
            None
        }
    }

    fn ui_checkbox(&mut self, ui: &mut Ui, text: impl Into<WidgetText>) -> Response {
        ui.checkbox(self, text)
    }

    fn ui_toggle(&mut self, ui: &mut Ui, text: impl Into<WidgetText>) -> Response {
        ui.toggle_value(self, text)
    }
}

impl Widget for &mut SplineGui {
    fn ui(self, ui: &mut Ui) -> Response {
        let mut resp = self.enabled.ui_checkbox(ui, "Spline Enabled");
        // Let me see what it looks like at a certain size first.
        
        ui.group(|ui| {
            let (rect, _) = ui.allocate_exact_size(vec2(200.0, 100.0), Sense::hover());
            ui.allocate_ui_at_rect(rect, |ui| {
                
            });
        });
        resp
    }
}

impl Widget for &mut NoiseGenIntervalGui {
    fn ui(self, ui: &mut Ui) -> Response {
        ui.vertical(|ui| {
            let mut resp = ui.checkbox(&mut self.enabled, "Interval Enabled");
            if !self.enabled {
                ui.disable();
            }
            // ui.label(RichText::from("TODO: Add Spline Editor").color(Color32::RED).strong());
            ui.add(&mut self.spline);
            ui.labeled(LABEL_WIDTH, "Octaves", |ui| {
                let drag = egui::DragValue::new(&mut self.octaves)
                    .speed(0.1)
                    .range(1..=4);
                resp = resp.union(ui.add(drag));
            });
            ui.labeled(LABEL_WIDTH, "Persistence", |ui| {
                let drag = egui::DragValue::new(&mut self.persistence)
                    .speed(0.01)
                    .range(0.0..=16.0);
                resp = resp.union(ui.add(drag));
            });
            ui.labeled(LABEL_WIDTH, "Scale", |ui| {
                let drag = egui::DragValue::new(&mut self.scale)
                    .speed(0.01)
                    .range(0.0..=16.0);
                resp = resp.union(ui.add(drag));
            });
            ui.labeled(LABEL_WIDTH, "X Multiplier", |ui| {
                let drag = egui::DragValue::new(&mut self.x_mult)
                    .speed(0.01)
                    .range(0.0..=100.0f32);
                resp = resp.union(ui.add(drag));
            });
            ui.labeled(LABEL_WIDTH, "Y Multiplier", |ui| {
                let drag = egui::DragValue::new(&mut self.y_mult)
                    .speed(0.01)
                    .range(0.0..=100.0f32);
                resp = resp.union(ui.add(drag));
            });
            resp
        }).inner

    }
}

impl SimplexGui {
    pub fn ui(&mut self, index: usize, ui: &mut Ui) -> Response {
        let mut resp = ui.checkbox(&mut self.enabled, "Simplex Enabled");
        if !self.enabled {
            ui.disable();
        }
        ui.labeled(LABEL_WIDTH, "Seed", |ui| {
            resp = resp.union(ui.text_edit_singleline(&mut self.seed));
        });
        if ui.button("Add Interval").clicked() {
            self.intervals.push(NoiseGenIntervalGui::default());
        }
        ui.group(|ui| {
            let mut remove_index = None;
            self.intervals.iter_mut().enumerate().for_each(|(index, interval)| {
                ui.group(|ui| {
                    if ui.button("Remove Interval").clicked() {
                        remove_index.replace(index);
                    }
                    resp = resp.union(interval.ui(ui));
                });
            });
            if let Some(index) = remove_index {
                resp.mark_changed();
                self.intervals.remove(index);
            }
        });
        resp
    }
}

impl Widget for &mut NoiseGenGui {
    fn ui(self, ui: &mut Ui) -> Response {
        let mut resp = ui.allocate_response(Vec2::ZERO, Sense::hover());
        if ui.button("Add Simplex").clicked() {
            self.simplexes.push(SimplexGui::default());
        }
        ui.group(|ui| {
            ScrollArea::vertical().max_height(f32::INFINITY).show(ui, |ui| {
                ui.vertical(|ui| {
                    let mut remove_index = None;
                    self.simplexes.iter_mut().enumerate().for_each(|(index, simplex)| {
                        ui.group(|ui| {
                            if ui.button("Remove Simplex").clicked() {
                                remove_index.replace(index);
                            }
                            resp = resp.union(simplex.ui(index, ui));
                        });
                    });
                    if let Some(index) = remove_index {
                        resp.mark_changed();
                        self.simplexes.remove(index);
                    }
                });
            });
        });
        resp
    }
}