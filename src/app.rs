#![allow(unused)]

use std::path::Path;
use std::sync::atomic::AtomicU64;

use egui::*;
use egui::{Color32, Sense, Vec2};
use image::Rgb;

fn next_index() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

use crate::noisegen::{NoiseGenSampler, NoiseGenConfig, Point};

fn load_image<P: AsRef<Path>>(path: P) -> Result<egui::ColorImage, image::ImageError> {
    let image = image::open(path)?;
    let size = [image.width() as _, image.height() as _];
    let image_buffer = image.to_rgb8();
    let pixels = image_buffer.as_flat_samples();
    Ok(egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice()))
}

fn convert_image_rgb(image: image::RgbImage) -> egui::ColorImage {
    let size = [image.width() as _, image.height() as _];
    let pixels = image.as_flat_samples();
    egui::ColorImage::from_rgb(size, pixels.as_slice())
}

fn convert_image_rgba(image: image::RgbaImage) -> egui::ColorImage {
    let size = [image.width() as _, image.height() as _];
    let pixels = image.as_flat_samples();
    egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice())
}

fn generate_grayscale_noise(noise_gui: &NoiseGenConfig) -> image::RgbImage {
    let mut img = image::RgbImage::new(256, 256);
    let noise_gen: NoiseGenSampler = noise_gui.clone().into();
    for y in 0..256 {
        for x in 0..256 {
            let fx = x as f64 + 0.5;
            let fy = y as f64 + 0.5;
            let noise = noise_gen.sample(Point::new(fx, fy));
            let grey = (255.0 * noise) as u8;
            img.put_pixel(x, y, Rgb([grey, grey, grey]));
        }
    }
    img.into()
}

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct NoiseEditorApp {
    noisegen_gui: NoiseGenConfig,
    auto_generate: bool,
    show_grid: bool,
    #[serde(skip)]
    texture: Option<TextureHandle>,
}

impl Default for NoiseEditorApp {
    fn default() -> Self {
        Self {
            auto_generate: false,
            noisegen_gui: NoiseGenConfig::default(),
            show_grid: true,
            texture: None,
        }
    }
}

impl NoiseEditorApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }

        Default::default()
    }
}

impl eframe::App for NoiseEditorApp {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            
            egui::menu::bar(ui, |ui| {
                // NOTE: no File->Quit on web pages!
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }
                
                // egui::widgets::global_dark_light_mode_buttons(ui);
            });
        });
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(Layout::left_to_right(Align::Min), |ui| {
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        egui::Frame::canvas(ui.style()).fill(Color32::BLACK).show(ui, |ui| {
                            let (rect, _) = ui.allocate_exact_size(Vec2::splat(768.), Sense::hover());
                            ui.allocate_ui_at_rect(rect, |ui| {
                                let painter = ui.painter();
                                if let Some(tex) = &self.texture {
                                    painter.image(tex.id(), rect, Rect::from_min_max(Pos2::new(0., 0.), Pos2::new(1., 1.)), Color32::WHITE);
                                }
                                if self.show_grid {
                                    for i in 0..16 {
                                        let n = i as f32 * (768. / 16.);
                                        painter.line_segment([Pos2::new(rect.left() + n, rect.top()), Pos2::new(rect.left() + n, rect.bottom())], Stroke::new(1.0, Color32::GREEN));
                                        painter.line_segment([Pos2::new(rect.left(), rect.top() + n), Pos2::new(rect.right(), rect.top() + n)], Stroke::new(1.0, Color32::RED));
                                    }
                                }
                            });
                        });
                        ui.checkbox(&mut self.show_grid, "Show Grid");
                    });
                });
                ui.vertical(|ui| {
                    let (rect, _) = ui.allocate_exact_size(Vec2::new(512., ui.spacing().interact_size.y), Sense::hover());
                    let button = egui::Button::new("Save").rounding(Rounding::ZERO);
                    if ui.put(rect, button).clicked() {
                        let img = generate_grayscale_noise(&self.noisegen_gui);
                        let save_file = rfd::FileDialog::new()
                            .add_filter("png", &["png"])
                            .set_file_name("output")
                            .set_directory(std::env::current_dir().unwrap())
                            .save_file();
                        if let Some(path) = save_file {
                            img.save(path);
                        }
    
                    }
    
                    let (rect, _) = ui.allocate_exact_size(Vec2::new(512., ui.spacing().interact_size.y), Sense::hover());
                    let button = egui::Button::new("Generate Noise").rounding(Rounding::ZERO);
                    if ui.put(rect, button).clicked() {
                        let img = generate_grayscale_noise(&self.noisegen_gui);
                        let colorimg = convert_image_rgb(img);
                        self.texture = Some(ctx.load_texture(format!("generation{}", next_index()), colorimg, TextureOptions::LINEAR));
                    }
                    ui.checkbox(&mut self.auto_generate, "Auto Generate");
                    let resp = self.noisegen_gui.ui(ui);
                    if self.auto_generate && resp.changed() {
                        let img = generate_grayscale_noise(&self.noisegen_gui);
                        let colorimg = convert_image_rgb(img);
                        self.texture = Some(ctx.load_texture(format!("generation{}", next_index()), colorimg, TextureOptions::LINEAR));
                    }
                });
            });
        });
    }
}
