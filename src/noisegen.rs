#![allow(unused)]
use core::f32;
use std::{default, path::Path, sync::atomic::AtomicU64};

use itertools::Itertools;
use noise::{NoiseFn, OpenSimplex};
use rand::{RngCore, SeedableRng};
use serde::{de::Visitor, ser::{SerializeSeq, SerializeStruct}};
use sha2::{Sha256, digest::Update, Digest};
use egui::*;
use splines::Key;

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

fn next_counter() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

fn next_id() -> Id {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    const BUFFER: &'static [u8] = b"next_id---id:>        ";
    let mut buffer: [u8; 22] = [0u8; 22];
    buffer.copy_from_slice(&BUFFER);
    let next_index = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let bytes = next_index.to_ne_bytes();
    buffer[14..22].copy_from_slice(&bytes);
    Id::new(bytes)
}

fn next_key_id() -> Id {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    const BUFFER: &'static [u8] = b"next_key_id---id:>        ";
    let mut buffer: [u8; 26] = [0u8; 26];
    buffer.copy_from_slice(&BUFFER);
    let next_index = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let bytes = next_index.to_ne_bytes();
    buffer[18..26].copy_from_slice(&bytes);
    Id::new(bytes)
}

fn octave_noise(noise_fn: &OpenSimplex, point: Point, octaves: u32, persistence: f64, lacunarity: f64, scale: f64, initial_amplitude: f64) -> f64 {
    let mut total = 0.0;
    let mut frequency = scale;
    let mut amplitude = initial_amplitude;
    let mut max_value = 1.0;
    let mut scale = 1.0;
    for _ in 0..octaves {
        let noise_value = noise_fn.get([point.x * frequency, point.y * frequency]);
        total += scale * noise_value * amplitude;
        scale *= 0.5;
        max_value += amplitude;
        frequency *= lacunarity;
        amplitude *= persistence;
    }
    // total / max_value
    total
}


struct NoiseLayer {
    noise: f64,
    amplitude: f64,
    frequency: f64,
    total_amplitude: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OctaveBlend {
    Scale = 0,
    Multiply = 1,
    Average = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OctaveGen {
    pub persistence: f64,
    pub lacunarity: f64,
    pub initial_amplitude: f64,
    pub scale: f64,
    pub x_mult: f64,
    pub y_mult: f64,
    pub rotation: f64,
    pub blend_mode: OctaveBlend,
    pub offset: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct NoiseGenSampler {
    simplexes: Vec<SimplexSampler>,
    weights: Vec<f64>,
    octave_gen: OctaveGen,
}

#[derive(Debug, Clone)]
pub struct SimplexSampler {
    enabled: bool,
    simplex: OpenSimplex,
    intervals: Vec<NoiseGenIntervalSampler>,
    weights: Vec<f64>,
    octave_gen: OctaveGen,
}

#[derive(Debug, Clone)]
pub struct NoiseGenIntervalSampler {
    enabled: bool,
    spline: Option<splines::Spline<f64, f64>>,
    octaves: u32,
    octave_gen: OctaveGen,
    invert: bool,
    bounds: NoiseBounds,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NoiseGenConfig {
    simplexes: Vec<SimplexConfig>,
    octave_gen: OctaveGen,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimplexConfig {
    enabled: bool,
    intervals: Vec<NoiseGenIntervalConfig>,
    octave_gen: OctaveGen,
    seed: String,
    weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
struct IntervalId {
    id: u64
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NoiseGenIntervalConfig {
    enabled: bool,
    spline: SplineConfig,
    octaves: u32,
    octave_gen: OctaveGen,
    invert: bool,
    bounds: NoiseBounds,
    weight: f64,
    #[serde(skip, default)]
    id: IntervalId,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SplineConfig {
    enabled: bool,
    spline: Vec<InterpKey>,
    #[serde(skip, default)]
    id: SplineId,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
struct InterpKey {
    x: f64,
    y: f64,
    interpolation: Interpolation,
    #[serde(skip, default)]
    id: DefId,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
struct SplineId(Id);

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct DefId(Id);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub enum NoiseBoundMode {
    Clamp,
    Cutoff,
    Range,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NoiseBound {
    pub t: f64,
    pub mode: NoiseBoundMode,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NoiseBounds {
    pub low: NoiseBound,
    pub high: NoiseBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
enum Interpolation {
    CatmullRom = 0,
    Cosine = 1,
    Linear = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

pub struct SimplexInterval<'a> {
    simplex: &'a OpenSimplex,
    interval: &'a NoiseGenIntervalSampler,
}

impl NoiseGenConfig {
    pub fn export<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<bincode::ErrorKind>> {
        let data = bincode::serialize(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn import<P: AsRef<Path>>(path: P) -> Result<Self, Box<bincode::ErrorKind>> {
        let data = std::fs::read(path)?;
        bincode::deserialize(&data)
    }
}

impl NoiseBound {
    pub fn clamp(t: f64) -> Self {
        Self {
            t,
            mode: NoiseBoundMode::Clamp
        }
    }

    pub fn cutoff(t: f64) -> Self {
        Self {
            t,
            mode: NoiseBoundMode::Cutoff
        }
    }

    pub fn range(t: f64) -> Self {
        Self {
            t,
            mode: NoiseBoundMode::Range
        }
    }
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

impl InterpKey {
    fn new(x: f64, y: f64, interpolation: Interpolation) -> Self {
        Self {
            x,
            y,
            interpolation,
            id: DefId::default(),
        }
    }
}

impl NoiseBounds {
    pub const fn new(low: NoiseBound, high: NoiseBound) -> Self {
        Self {
            low,
            high
        }
    }

    pub fn bound(self, t: f64) -> f64 {
        use NoiseBoundMode::*;
        match (self.low.mode, self.high.mode) {
            (Range, Range) => {
                let low = self.low.t;
                let high = self.high.t;
                let clamped = t.max(low).min(high);
                let diff = high - low;
                let rel = clamped - low;
                rel * (1. / diff)
            }
            (Range, Clamp) => {
                let low = self.low.t;
                let clamped = t.max(low).min(self.high.t);
                let diff = 1.0 - low;
                let rel = clamped - low;
                rel * (1. / diff)
            }
            (Range, Cutoff) => {
                if t > self.high.t {
                    0.0
                } else {
                    let low = self.low.t;
                    let clamped = t.max(low);
                    let diff = 1.0 - low;
                    let rel = clamped - low;
                    rel * (1. / diff)
                }
            }
            (Clamp, Range) => {
                let high = self.high.t;
                let clamped = t.max(self.low.t).min(high);
                clamped * (1. / high)
            }
            (Clamp, Clamp) => {
                t.max(self.low.t).min(self.high.t)
            }
            (Clamp, Cutoff) => {
                if t > self.high.t {
                    0.
                } else {
                    t.max(self.low.t)
                }
            }
            (Cutoff, Range) => {
                if t < self.low.t {
                    0.
                } else {
                    let high = self.high.t;
                    let clamped = t.min(high);
                    clamped * (1. / high)
                }
            }
            (Cutoff, Clamp) => {
                if t < self.low.t {
                    0.
                } else {
                    t.min(self.high.t)
                }
            }
            (Cutoff, Cutoff) => {
                if t < self.low.t || t > self.high.t {
                    0.
                } else {
                    t
                }
            }
        }
    }
}

impl NoiseGenIntervalSampler {
    pub fn sample(&self, simplex: &OpenSimplex, point: Point) -> f64 {
        let point = Point::new(point.x * 0.125, point.y * 0.125);
        // let point = Point::new(point.x * self.x_mult * 0.01, point.y * self.y_mult * 0.01);
        let noise = self.octave_gen.sample(point, (0..self.octaves).map(|_| simplex));
        // let noise = octave_noise(simplex, point, self.octaves, self.persistence, self.lacunarity, self.scale, self.initial_amplitude);
        let gradient = (noise + 1.) * 0.5;
        let gradient = if let Some(spline) = &self.spline {
            spline.sample(gradient).unwrap_or(gradient)
        } else {
            gradient
        };
        let gradient = self.bounds.bound(gradient);
        if self.invert {
            -gradient + 1.0
        } else {
            gradient
        }
    }
}

impl SimplexSampler {
    pub fn enabled(&self) -> bool {
        self.enabled && !self.intervals.is_empty() && self.intervals.iter().any(|interval| interval.enabled)
    }

    pub fn sample(&self, point: Point) -> f64 {
        self.octave_gen.weighted_sample(point, &self.weights,self.intervals.iter().filter_map(|interval| interval.enabled.opt(SimplexInterval {
            interval,
            simplex: &self.simplex
        })))
        // let (noise_accum, div) = self.intervals.iter().fold((0., 0.), |(accum, div), interval| {
        //     (accum + interval.sample(&self.simplex, point), div + 1.)
        // });
        // noise_accum / div
    }
}

impl NoiseGenSampler {
    pub fn sample(&self, point: Point) -> f64 {
        if self.simplexes.is_empty() {
            return 0.0;
        }
        self.octave_gen.weighted_sample(point, &self.weights, self.simplexes.iter().filter(|simplex| simplex.enabled))
    }
}

const LABEL_WIDTH: f32 = 100.;

impl Widget for &mut SplineConfig {
    fn ui(self, ui: &mut Ui) -> Response {
        // let mut resp = self.enabled.ui_checkbox(ui, "Spline Enabled");
        // Let me see what it looks like at a certain size first.
        let resp = CollapsingHeader::new("Spline")
            .default_open(self.enabled)
            .id_source(self.id.0)
            .show(ui, |ui| {
                egui::Frame::dark_canvas(ui.style()).rounding(Rounding::ZERO)
                .show(ui, |ui| {
                        let (rect, _) = ui.allocate_exact_size(vec2(410.0, 210.0), Sense::click());
                        let inner_rect = rect.shrink(5.0);
                        let point_transformer = |x: f64, y: f64| {
                            let y = -y + 1.0;
                            Pos2::new(
                                inner_rect.left() + x as f32 * inner_rect.width(),
                                inner_rect.top() + y as f32 * inner_rect.height(),
                            )
                        };
                        
                        /// Transform a Pos2 in gui space into a normalized key coordinate.
                        let key_transformer = |pos: Pos2| {
                            let x = pos.x - inner_rect.left();
                            let y = pos.y - inner_rect.top();
                            let x = x / inner_rect.width();
                            let y = y / inner_rect.height();
                            (x as f64, -(y as f64) + 1.0)
                        };
                        
                        ui.allocate_ui_at_rect(rect, |ui| {
                            let mut resp = ui.allocate_rect(inner_rect, Sense::click());
                            
                            let mut draw_key = |key: &mut InterpKey, bounds: Rect, remove: Option<&mut bool>| {
                                let p = point_transformer(key.x, key.y);
                                let prect = Rect::from_center_size(p, Vec2::splat(10.0));
                                ui.push_id(next_id(), |ui| {
                                    let mut presp = ui.allocate_rect(prect, Sense::click_and_drag());
                                    let color = presp.hovered().select(Color32::WHITE, Color32::from_rgb(175, 175, 175));
                                    if presp.dragged() {
                                        let pointer = ui.input(|i| i.pointer.hover_pos());
                                        if let Some(pointer) = pointer {
                                            let clamped = pointer.clamp(bounds.min, bounds.max);
                                            let new_key = key_transformer(clamped);
                                            if new_key.0 != key.x && new_key.1 != key.y {
                                                key.x = new_key.0;
                                                key.y = new_key.1;
                                                presp.mark_changed();
                                            }
                                        }
                                    } else {
                                        let mut v = false;
                                        presp.context_menu(|ui| {
                                            ui.allocate_exact_size(Vec2::new(150.0, 0.0), Sense::hover());
                                            if let Some(remove) = remove {
                                                if ui.button("Remove").clicked() {
                                                    *remove = true;
                                                    ui.close_menu();
                                                    v = true;
                                                }
                                            }
                                            ui.group(|ui| {
                                                if ui.selectable_value(&mut key.interpolation, Interpolation::CatmullRom, "CatmullRom").changed() {
                                                    v = true;
                                                    ui.close_menu();
                                                }
                                                if ui.selectable_value(&mut key.interpolation, Interpolation::Cosine, "Cosine").changed() {
                                                    v = true;
                                                    ui.close_menu();
                                                }
                                                if ui.selectable_value(&mut key.interpolation, Interpolation::Linear, "Linear").changed() {
                                                    v = true;
                                                    ui.close_menu();
                                                };
                                            });
                                        });
                                        if v {
                                            presp.mark_changed();
                                        }
                                    }
                                    ui.painter().circle_filled(p, 5.0, color);
                                    presp
                                }).inner
                            };
                            let max = point_transformer(self.spline[1].x, 0.0);
                            let min = point_transformer(0.0, 1.0);
                            let bounds = Rect::from_min_max(min, max);
                            let mut draw_resp = draw_key(&mut self.spline[0], bounds, None);
                            let max = point_transformer(1.0, 0.0);
                            let min = point_transformer(self.spline[self.spline.len() - 2].x, 1.0);
                            let bounds = Rect::from_min_max(min, max);
                            let end_index = self.spline.len() - 1;
                            draw_resp = draw_resp.union(draw_key(&mut self.spline[end_index], bounds, None));
                            let mut remove_index = None;
                            for i in 1..self.spline.len() - 1 {
                                let min = point_transformer(self.spline[i - 1].x, 1.0);
                                let max = point_transformer(self.spline[i + 1].x, 0.0);
                                let bounds = Rect::from_min_max(min, max);
                                let mut remove = false;
                                let some_rem = if i > 1 && i < self.spline.len() - 2 {
                                    Some(&mut remove)
                                } else {
                                    None
                                };
                                draw_resp = draw_resp.union(draw_key(&mut self.spline[i], bounds, some_rem));
                                if remove {
                                    remove_index.replace(i);
                                }
                            }
                            if let Some(index) = remove_index {
                                self.spline.remove(index);
                            }
                            if !(draw_resp.hovered() || draw_resp.dragged()) {
                                if resp.clicked() && ! resp.dragged() {
                                    draw_resp.mark_changed();
                                    let pointer = ui.input(|i| i.pointer.hover_pos());
                                    if let Some(pointer) = pointer {
                                        if inner_rect.contains(pointer) {
                                            let (x, y) = key_transformer(pointer);
                                            // find the keys that x exists between
                                            for i in 1..self.spline.len() {
                                                let left = self.spline[i - 1].x;
                                                let right = self.spline[i].x;
                                                if x >= left && x < right {
                                                    self.spline.insert(i, InterpKey::new(x, y, Interpolation::CatmullRom));
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            } else if draw_resp.dragged() {
                                draw_resp.mark_changed();
                            }
                            let spline = splines::Spline::from_iter(self.spline.iter().cloned().map(|interp| {
                                Key::new(interp.x, interp.y, interp.interpolation.into())
                            }));
                            let x = 0 as f32 + inner_rect.left();
                            const SAMPLES: u32 = 150;
                            const SAMPDIV: f64 = SAMPLES as f64;
                            let i_mult = inner_rect.width() / SAMPDIV as f32;
                            let t = 0 as f64 / SAMPDIV;
                            let y = spline.sample(t).unwrap_or_default();
                            let y = -y + 1.0;
                            let y = y as f32 * inner_rect.height() + inner_rect.top();
                            let mut previous = Pos2::new(x, y);
                            for i in 1..=SAMPLES {
                                let x = i as f32 * i_mult + inner_rect.left();
                                let t = i as f64 / SAMPDIV;
                                let y = spline.sample(t).unwrap_or_default();
                                let y = -y + 1.0;
                                let y = y as f32 * inner_rect.height() + inner_rect.top();
                                let current = Pos2::new(x, y);
                                ui.painter().line_segment([previous, current], Stroke::new(1.0, Color32::WHITE));
                                previous = current;
                            }
                            resp.union(draw_resp)
                        }).inner
                    }).inner
            });
        let old_enabled = self.enabled;
        self.enabled = !resp.fully_closed();
        let mut resp = resp.body_returned.unwrap_or_else(|| {
            ui.allocate_response(Vec2::ZERO, Sense::hover())
        });
        if old_enabled != self.enabled {
            resp.mark_changed();
        }
        resp
    }
}

impl Widget for &mut NoiseGenIntervalConfig {
    fn ui(self, ui: &mut Ui) -> Response {
        ui.vertical(|ui| {
            let mut resp = ui.checkbox(&mut self.enabled, "Interval Enabled");
            if !self.enabled {
                ui.disable();
            }
            // ui.label(RichText::from("TODO: Add Spline Editor").color(Color32::RED).strong());
            resp = resp.union(ui.add(&mut self.spline));
            ui.labeled(LABEL_WIDTH, "Layer Weight", |ui| {
                let drag = egui::DragValue::new(&mut self.weight)
                    .speed(0.0025)
                    .range(0.0..=f64::INFINITY);
                resp.join(ui.add(drag));
            });
            ui.labeled(LABEL_WIDTH, "Octaves", |ui| {
                let drag = egui::DragValue::new(&mut self.octaves)
                    .speed(0.05)
                    .range(1..=4);
                resp = resp.union(ui.add(drag));
            });
            resp.join(ui.add(&mut self.octave_gen));
            ui.labeled(LABEL_WIDTH, "Low Bound", |ui| {
                let drag = egui::DragValue::new(&mut self.bounds.low.t)
                    .speed(0.0025)
                    .range(0.0..=self.bounds.high.t - 0.001);
                resp = resp.union(ui.add(drag));
                let mut cresp = ui.selectable_value(&mut self.bounds.low.mode, NoiseBoundMode::Clamp, "Clamp");
                cresp.join(ui.selectable_value(&mut self.bounds.low.mode, NoiseBoundMode::Range, "Range"));
                cresp.join(ui.selectable_value(&mut self.bounds.low.mode, NoiseBoundMode::Cutoff, "Cutoff"));
                if cresp.changed() {
                    resp.mark_changed();
                }
            });
            ui.labeled(LABEL_WIDTH, "High Bound", |ui| {
                let drag = egui::DragValue::new(&mut self.bounds.high.t)
                    .speed(0.0025)
                    .range(self.bounds.low.t + 0.001..=1.0);
                resp = resp.union(ui.add(drag));
                let mut cresp = ui.selectable_value(&mut self.bounds.high.mode, NoiseBoundMode::Clamp, "Clamp");
                cresp.join(ui.selectable_value(&mut self.bounds.high.mode, NoiseBoundMode::Range, "Range"));
                cresp.join(ui.selectable_value(&mut self.bounds.high.mode, NoiseBoundMode::Cutoff, "Cutoff"));
                if cresp.changed() {
                    resp.mark_changed();
                }
            });
            resp.join(ui.toggle_value(&mut self.invert, "Invert"));
            resp
        }).inner

    }
}

impl SimplexConfig {
    pub fn ui(&mut self, index: usize, ui: &mut Ui) -> Response {
        let mut resp = ui.checkbox(&mut self.enabled, "Simplex Enabled");
        if !self.enabled {
            ui.disable();
        }
        ui.labeled(LABEL_WIDTH, "Layer Weight", |ui| {
            let drag = egui::DragValue::new(&mut self.weight)
                .speed(0.0025)
                .range(0.0..=f64::INFINITY);
            resp.join(ui.add(drag));
        });
        ui.labeled(LABEL_WIDTH, "Seed", |ui| {
            resp = resp.union(ui.text_edit_singleline(&mut self.seed));
        });
        resp = resp.union(ui.add(&mut self.octave_gen));
        if ui.button("Add Interval").clicked() {
            self.intervals.push(NoiseGenIntervalConfig::default());
            resp.mark_changed();
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

impl Widget for &mut NoiseGenConfig {
    fn ui(self, ui: &mut Ui) -> Response {
        let mut resp = ui.add(&mut self.octave_gen);
        if ui.button("Add Simplex").clicked() {
            self.simplexes.push(SimplexConfig::default());
            resp.mark_changed();
        }
        ui.group(|ui| {
            ScrollArea::vertical()
            .auto_shrink([false, false])
            .max_height(f32::INFINITY).show(ui, |ui| {
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

impl Widget for &mut OctaveGen {
    fn ui(self, ui: &mut Ui) -> Response {
        let mut resp = ui.labeled(LABEL_WIDTH, "Initial Amplitude", |ui| {
            let drag = DragValue::new(&mut self.initial_amplitude)
                .speed(0.01)
                .range(0.01..=30.0);
            ui.add(drag)
        });
        let mut resp = resp.union(ui.labeled(LABEL_WIDTH, "Scale", |ui| {
            let drag = DragValue::new(&mut self.scale)
                .speed(0.01)
                .range(0.01..=30.0);;
            ui.add(drag)
        }));
        let mut resp = resp.union(ui.labeled(LABEL_WIDTH, "Persistence", |ui| {
            let drag = DragValue::new(&mut self.persistence)
                .speed(0.0025)
                .range(-4.0..=4.0);
            ui.add(drag)
        }));
        let mut resp = resp.union(ui.labeled(LABEL_WIDTH, "Lacunarity", |ui| {
            let drag = DragValue::new(&mut self.lacunarity)
                .speed(0.0025);
            ui.add(drag)
        }));
        ui.labeled(LABEL_WIDTH, "X Multiplier", |ui| {
            let drag = egui::DragValue::new(&mut self.x_mult)
                .speed(0.01)
                .range(0.0..=100.0);
            resp = resp.union(ui.add(drag));
        });
        ui.labeled(LABEL_WIDTH, "Y Multiplier", |ui| {
            let drag = egui::DragValue::new(&mut self.y_mult)
                .speed(0.01)
                .range(0.0..=100.0);
            resp = resp.union(ui.add(drag));
        });
        resp.join(ui.labeled(LABEL_WIDTH, "Rotation", |ui| {
            let drag = egui::DragValue::new(&mut self.rotation)
                .speed(0.01);
            ui.add(drag)
        }));
        ui.labeled(LABEL_WIDTH, "Offset", |ui| {
            ui.label("X");
            let drag = egui::DragValue::new(&mut self.offset.0)
                .speed(0.01);
            resp.join(ui.add(drag));
            ui.label("Y");
            let drag = egui::DragValue::new(&mut self.offset.1)
                .speed(0.01);
            resp.join(ui.add(drag));
        });
        ui.labeled(LABEL_WIDTH, "Blend Mode", |ui| {
            resp.join(ui.selectable_value(&mut self.blend_mode, OctaveBlend::Scale, "Scale"));
            resp.join(ui.selectable_value(&mut self.blend_mode, OctaveBlend::Average, "Average"));
            resp.join(ui.selectable_value(&mut self.blend_mode, OctaveBlend::Multiply, "Multiply"));
        });
        resp
    }
}

impl NoiseLayer {
    pub const fn new(amplitude: f64, frequency: f64, noise: f64) -> Self {
        Self {
            amplitude,
            frequency,
            noise,
            total_amplitude: 0.0,
        }
    }
}

impl OctaveGen {
    pub fn weighted_sample<F: NoiseSampler, It: IntoIterator<Item = F>>(&self, point: Point, weights: &[f64], layers: It) -> f64 {
        let mut scale = 1.0;
        let angle = self.rotation.to_radians();
        let point = Point::new(point.x + self.offset.0, point.y + self.offset.1);
        let point = if self.rotation != 0.0 {
            Point::new(
                point.x * angle.cos() - point.y * angle.sin(),
                point.x * angle.sin() + point.y * angle.cos()
            )
        } else {
            point
        };
        let point = Point::new(point.x * self.x_mult, point.y * self.y_mult);
        match self.blend_mode {
            OctaveBlend::Scale => {
                let init = NoiseLayer::new(
                    self.initial_amplitude,
                    self.scale,
                    0.0,  
                );
                layers.into_iter().enumerate().fold(init, |mut accum, (i, noise)| {
                    accum.noise += scale * noise.sample_noise(Point::new(point.x * accum.frequency, point.y * accum.frequency)) * accum.amplitude * weights[i];
                    scale *= 0.5;
                    accum.amplitude *= self.persistence;
                    accum.frequency *= self.lacunarity;
                    accum
                }).noise
            },
            OctaveBlend::Multiply => {
                let init = NoiseLayer::new(
                    self.initial_amplitude,
                    self.scale,
                    1.0,  
                );
                layers.into_iter().enumerate().fold(init, |mut accum, (i, noise)| {
                    accum.noise *= noise.sample_noise(Point::new(point.x * accum.frequency, point.y * accum.frequency)) * accum.amplitude * weights[i];
                    accum.amplitude *= self.persistence;
                    accum.frequency *= self.lacunarity;
                    accum
                }).noise
            },
            OctaveBlend::Average => {
                let init = NoiseLayer::new(
                    self.initial_amplitude,
                    self.scale,
                    0.0,  
                );
                let result = layers.into_iter().enumerate().fold(init, |mut accum, (i, noise)| {
                    accum.noise += noise.sample_noise(Point::new(point.x * accum.frequency, point.y * accum.frequency)) * accum.amplitude * weights[i];
                    accum.total_amplitude += accum.amplitude;
                    accum.amplitude *= self.persistence;
                    accum.frequency *= self.lacunarity;
                    accum
                });
                result.noise / result.total_amplitude
            }
        }
    }

    pub fn sample<F: NoiseSampler, It: IntoIterator<Item = F>>(&self, point: Point, layers: It) -> f64 {
        let mut scale = 1.0;
        let angle = self.rotation.to_radians();
        let point = Point::new(point.x + self.offset.0, point.y + self.offset.1);
        let point = if self.rotation != 0.0 {
            Point::new(
                point.x * angle.cos() - point.y * angle.sin(),
                point.x * angle.sin() + point.y * angle.cos()
            )
        } else {
            point
        };
        let point = Point::new(point.x * self.x_mult, point.y * self.y_mult);
        match self.blend_mode {
            OctaveBlend::Scale => {
                let init = NoiseLayer::new(
                    self.initial_amplitude,
                    self.scale,
                    0.0,  
                );
                layers.into_iter().fold(init, |mut accum, noise| {
                    accum.noise += scale * noise.sample_noise(Point::new(point.x * accum.frequency, point.y * accum.frequency)) * accum.amplitude;
                    scale *= 0.5;
                    accum.amplitude *= self.persistence;
                    accum.frequency *= self.lacunarity;
                    accum
                }).noise
            },
            OctaveBlend::Multiply => {
                let init = NoiseLayer::new(
                    self.initial_amplitude,
                    self.scale,
                    1.0,  
                );
                layers.into_iter().fold(init, |mut accum, noise| {
                    accum.noise *= noise.sample_noise(Point::new(point.x * accum.frequency, point.y * accum.frequency)) * accum.amplitude;
                    accum.amplitude *= self.persistence;
                    accum.frequency *= self.lacunarity;
                    accum
                }).noise
            },
            OctaveBlend::Average => {
                let init = NoiseLayer::new(
                    self.initial_amplitude,
                    self.scale,
                    0.0,  
                );
                let result = layers.into_iter().fold(init, |mut accum, noise| {
                    accum.noise += noise.sample_noise(Point::new(point.x * accum.frequency, point.y * accum.frequency)) * accum.amplitude;
                    accum.total_amplitude += accum.amplitude;
                    accum.amplitude *= self.persistence;
                    accum.frequency *= self.lacunarity;
                    accum
                });
                result.noise / result.total_amplitude
            }
        }
    }
}

impl NoiseSampler for &SimplexSampler {
    fn sample_noise(self, point: Point) -> f64 {
        self.sample(point)
    }
}

impl<'a> NoiseSampler for SimplexInterval<'a> {
    fn sample_noise(self, point: Point) -> f64 {
        self.interval.sample(&self.simplex, point)
    }
}

impl<F: FnMut(Point) -> f64> NoiseSampler for F {
    fn sample_noise(mut self, point: Point) -> f64 {
        let mut f = self;
        f(point)
    }
}

impl NoiseSampler for &OpenSimplex {
    fn sample_noise(self, point: Point) -> f64 {
        self.get([point.x, point.y])
    }
}

impl ResponseExt for Response {
    fn join(&mut self, other: Response) {
        *self = self.union(other);
    }
}

impl Default for IntervalId {
    fn default() -> Self {
        Self {
            id: next_counter()
        }
    }
}

impl Default for OctaveGen {
    fn default() -> Self {
        Self {
            persistence: 0.5,
            lacunarity: 1.0,
            initial_amplitude: 1.0,
            scale: 1.0,
            x_mult: 1.0,
            y_mult: 1.0,
            rotation: 0.0,
            blend_mode: OctaveBlend::Scale,
            offset: (0., 0.),
        }
    }
}

impl Default for NoiseGenConfig {
    fn default() -> Self {
        Self {
            simplexes: vec![SimplexConfig::default()],
            octave_gen: OctaveGen::default(),
        }
    }
}

impl Default for NoiseGenIntervalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            spline: SplineConfig::default(),
            octaves: 4,
            octave_gen: OctaveGen::default(),
            invert: false,
            bounds: NoiseBounds::new(NoiseBound::clamp(0.), NoiseBound::clamp(1.)),
            weight: 1.0,
            id: IntervalId::default(),
        }
    }
}

impl Default for SplineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            spline: vec![
                InterpKey::new(0.0, 0.0, Interpolation::CatmullRom),
                InterpKey::new(0.0, 0.0, Interpolation::CatmullRom),
                InterpKey::new(1.0, 1.0, Interpolation::CatmullRom),
                InterpKey::new(1.0, 1.0, Interpolation::CatmullRom),
            ],
            id: SplineId(next_id()),
        }
    }
}

impl Default for SplineId {
    fn default() -> Self {
        Self(next_id())
    }
}

impl Default for DefId {
    fn default() -> Self {
        Self(next_key_id())
    }
}

impl Default for SimplexConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            seed: String::from(""),
            intervals: vec![NoiseGenIntervalConfig::default()],
            octave_gen: OctaveGen::default(),
            weight: 1.0,
        }
    }
}

impl From<NoiseGenConfig> for NoiseGenSampler {
    fn from(value: NoiseGenConfig) -> Self {
        let mut total_weight = 0.0;
        let mut weights = value.simplexes.iter().map(|simp| {
            total_weight += simp.weight;
            simp.weight
        }).collect_vec();
        let weight_mul = 1.0 / total_weight;
        weights.iter_mut().for_each(|weight| *weight *= weight_mul);
        Self {
            octave_gen: value.octave_gen,
            weights,
            simplexes: value.simplexes.into_iter().filter(|simplex| simplex.enabled).map(|simplex| simplex.into()).collect()
        }
    }
}

impl From<SimplexConfig> for SimplexSampler {
    fn from(value: SimplexConfig) -> Self {
        let seed = make_seed(value.seed);
        let simplex = OpenSimplex::new(seed);
        let mut total_weight = 0.0;
        let mut weights = value.intervals.iter().map(|interval| {
            total_weight += interval.weight;
            interval.weight
        }).collect_vec();
        let weight_mul = 1.0 / total_weight;
        weights.iter_mut().for_each(|weight| *weight *= weight_mul);
        Self {
            enabled: value.enabled,
            simplex,
            weights,
            intervals: value.intervals.into_iter().filter(|interval| interval.enabled).map(NoiseGenIntervalSampler::from).collect(),
            octave_gen: value.octave_gen,
        }
    }
}

impl From<NoiseGenIntervalConfig> for NoiseGenIntervalSampler {
    fn from(value: NoiseGenIntervalConfig) -> Self {
        Self {
            enabled: value.enabled,
            spline: if value.spline.enabled {
                Some(splines::Spline::from_iter(value.spline.spline.into_iter().map(|interp| {
                    Key::new(interp.x, interp.y, interp.interpolation.into())
                })))
            } else {
                None
            },
            octaves: value.octaves,
            octave_gen: value.octave_gen,
            invert: value.invert,
            bounds: value.bounds,
        }
    }
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

pub trait ResponseExt {
    fn join(&mut self, other: Response);
}

pub trait NoiseSampler {
    fn sample_noise(self, point: Point) -> f64;
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

trait UiExt {
    fn labeled<R, Text: Into<widget_text::WidgetText>, F: FnMut(&mut Ui) -> R>(&mut self, label_width: f32, text: Text, add_contents: F) -> R;
}

impl UiExt for Ui {
    fn labeled<R, Text: Into<widget_text::WidgetText>, F: FnMut(&mut Ui) -> R>(&mut self, label_width: f32, text: Text, add_contents: F) -> R {
        let mut add_contents = add_contents;
        self.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(label_width, ui.spacing().interact_size.y), Sense::hover());
            let label = egui::widgets::Label::new(text);
            ui.put(rect, label);
            add_contents(ui)
        }).inner
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