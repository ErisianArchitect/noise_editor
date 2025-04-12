#![allow(unused)]

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{self, channel, Sender, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use egui::*;
use egui::{Color32, Sense, Vec2};
use image::Rgb;
use rayon::iter::IntoParallelIterator;

enum NoiseGenAsyncMsg {
    BeginSampling {
        sampler: Box<NoiseGenSampler>,
        size: (u32, u32),
    },
    Terminate,
}

impl NoiseGenAsyncMsg {
    pub fn begin(sampler: NoiseGenSampler, size: (u32, u32)) -> Self {
        Self::BeginSampling { sampler: Box::new(sampler), size }
    }
}

enum NoiseGenAsyncResponse {
    Final(image::RgbImage),
    Progress(f64),
}

struct NoiseGenAsyncTask {
    sender: Sender<NoiseGenAsyncMsg>,
    receiver: Receiver<image::RgbImage>,
}

impl Drop for NoiseGenAsyncTask {
    fn drop(&mut self) {
        self.sender.send(NoiseGenAsyncMsg::Terminate);
    }
}

impl NoiseGenAsyncTask {
    pub fn request(sampler: NoiseGenSampler, size: (u32, u32)) -> NoiseGenAsyncTask {
        let (client_sender, server_recv) = channel();
        let (server_sender, client_recv) = channel();
        let client = NoiseGenAsyncTask {
            sender: client_sender,
            receiver: client_recv,
        };
        let server = NoiseGenAsyncServer {
            sender: server_sender,
            receiver: server_recv,
        };

        client
    }
}

struct NoiseGenAsyncServer {
    sender: Sender<image::RgbImage>,
    receiver: Receiver<NoiseGenAsyncMsg>,
}

struct ThreadPool {
    workers: Vec<NoiseGenAsyncWorker>,
    // sender: Sender<(u32, u32)>,
    // receiver: Receiver<JobResponse>,
    receiver: Receiver<image::RgbImage>,
    terminate: Arc<AtomicBool>,
    progress: Arc<Mutex<f64>>,
    thread: thread::JoinHandle<()>,
}

impl ThreadPool {
    pub fn spawn(count: u32, image_size: (u32, u32), sampler: NoiseGenSampler) -> Self {
        assert!(count > 0);
        let (task_sender, task_receiver) = channel();
        let (response_sender, response_receiver) = channel();
        fn arc_mutex<T>(value: T) -> Arc<Mutex<T>> {
            Arc::new(Mutex::new(value))
        }
        // let pool_sender = arc_mutex(pool_sender);
        // let pool_receiver = arc_mutex(pool_receiver);
        let (pool_sender, pool_receiver) = channel();
        let response_sender = response_sender;
        let task_receiver = arc_mutex(task_receiver);
        let sampler = sampler;
        let terminate = Arc::new(AtomicBool::new(false));
        let progress = arc_mutex(0.0);
        let workers = (0..count).map(|id| NoiseGenAsyncWorker::spawn(id, sampler.clone(), Sender::clone(&response_sender), Arc::clone(&task_receiver), Arc::clone(&terminate))).collect::<Vec<_>>();
        Self {
            workers,
            receiver: pool_receiver,
            terminate: Arc::clone(&terminate),
            progress: Arc::clone(&progress),
            thread: thread::spawn(move || {
                let mut img = image::RgbImage::new(image_size.0, image_size.1);
                let chunk_size = (128, 128);
                let x_count = image_size.0 / chunk_size.0;
                let y_count = image_size.1 / chunk_size.1;
                let x_overflow = image_size.0 % chunk_size.0;
                let y_overflow = image_size.1 % chunk_size.1;
                // task_sender
                // response_receiver
                // pool_sender
                let mut task_count = 0;
                let mut yr = 0;
                for y in 0..y_count {
                    let mut xr = 0;
                    for x in 0..x_count {
                        task_sender.send(Job {
                            offset: (xr, yr),
                            size: chunk_size,
                        }).unwrap();
                        task_count += 1;
                        xr += chunk_size.0;
                    }
                    if x_overflow > 0 {
                        task_sender.send(Job {
                            offset: (xr, yr),
                            size: (x_overflow, chunk_size.1),
                        }).unwrap();
                        task_count += 1;
                    }
                    yr += chunk_size.1;
                }
                if y_overflow > 0 {
                    let mut xr = 0;
                    for x in 0..x_count {
                        task_sender.send(Job {
                            offset: (xr, yr),
                            size: (chunk_size.0, y_overflow),
                        }).unwrap();
                        task_count += 1;
                        xr += chunk_size.0;
                    }
                    if x_overflow > 0 {
                        task_sender.send(Job {
                            offset: (xr, yr),
                            size: (x_overflow, y_overflow),
                        }).unwrap();
                        task_count += 1;
                    }
                }
                let mut completed_tasks = 0;
                'thread_pool: loop {
                    if terminate.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    if let Ok(JobResponse { image, offset, size }) = response_receiver.try_recv() {
                        let mut iy = 0;
                        for y in offset.1..offset.1 + size.1 {
                            let mut ix = 0;
                            for x in offset.0..offset.0 + size.0 {
                                let pixel = image.get_pixel(ix, iy).clone();
                                img.put_pixel(x, y, pixel);
                                ix += 1;
                            }
                            iy += 1;
                        }
                        completed_tasks += 1;
                        // let mut prog = progress.lock().unwrap();
                        // *prog = completed_tasks as f64 / task_count as f64;
                        if completed_tasks == task_count {
                            break 'thread_pool;
                        }
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                pool_sender.send(img).unwrap();
            })
        }
    }

    pub fn terminate(&mut self) {
        self.terminate.store(true, std::sync::atomic::Ordering::Relaxed);
        for worker in self.workers.drain(..) {
            worker.thread.join();
        }
    }

    /// Normalized within the range of `0.0` and `1.0`.
    pub fn progress(&self) -> f64 {
        *self.progress.lock().unwrap()
    }

    pub fn try_recv(&mut self) -> Option<image::RgbImage> {
        if let Ok(img) = self.receiver.try_recv() {
            self.terminate();
            Some(img)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod testing_sandbox {
    use std::io::Write;

    // TODO: Remove this sandbox when it is no longer in use.
    use super::*;
    #[test]
    fn sandbox() {
        println!("Building image...");
        std::io::stdout().flush().unwrap();
        let config = NoiseGenConfig::import("./test_sampler.simp").unwrap();
        let sampler = NoiseGenSampler::from(config.clone());
        println!("Num CPUS: {}", num_cpus::get());
        std::io::stdout().flush().unwrap();
        let count = num_cpus::get() as u32;
        let count = 12;
        let image_size = 2048;
        let mut pool = ThreadPool::spawn(count, (image_size, image_size), sampler);
        let start_time = std::time::Instant::now();
        let mut print_timer = std::time::Instant::now();
        let result = loop {
            if print_timer.elapsed().as_millis() >= 500 {
                print_timer = std::time::Instant::now();
                // println!("Progress: {:.2}%", pool.progress() * 100.0);
                // std::io::stdout().flush().unwrap();
            }
            if let Some(img) = pool.try_recv() {
                break img;
            }
        };
        let elapsed = start_time.elapsed();
        result.save("./test_output.png").unwrap();
        println!("[Multi-Threaded] Built image in {:.2} seconds!", elapsed.as_secs_f64());
        let start_time = std::time::Instant::now();
        let img = generate_grayscale_noise(&config, image_size);
        let elapsed = start_time.elapsed();
        println!("[Single-Threaded] Built image in {:.2} seconds!", elapsed.as_secs_f64());
    }
}

struct NoiseGenAsyncWorker {
    // receiver: Arc<Mutex<Receiver<NoiseGenAsyncJob>>>,
    // sender: Arc<Mutex<Sender<JobResponse>>>,
    id: u32,
    thread: thread::JoinHandle<()>,
}

impl NoiseGenAsyncWorker {
    pub fn spawn(
        id: u32,
        sampler: NoiseGenSampler,
        sender: Sender<JobResponse>,
        receiver: Arc<Mutex<Receiver<Job>>>,
        terminate: Arc<AtomicBool>,
    ) -> Self {
        let thread = thread::spawn(move || 'worker: loop {
            let term = terminate.load(std::sync::atomic::Ordering::Relaxed);
            if term {
                return;
            }
            if let Ok(Job { offset, size }) = receiver.lock().unwrap().try_recv() {
                // let image = generate_offset_grayscale_noise(sampler.as_ref(), offset, size);
                let mut img = image::RgbImage::new(size.0, size.1);
                let mut yi = 0;
                for y in offset.1..offset.1 + size.1 {
                    let mut xi = 0;
                    for x in offset.0..offset.0 + size.0 {
                        let fx = x as f64;
                        let fy = y as f64;
                        let noise = sampler.sample(Point::new(fx, fy));
                        let grey = (255.0 * noise) as u8;
                        img.put_pixel(xi, yi, Rgb([grey, grey, grey]));
                        xi += 1;
                    }
                    yi += 1;
                }
                sender.send(JobResponse::new(img, offset, size)).unwrap();
            }
        });
        Self {
            id,
            thread,
        }
    }
}

struct Job {
    offset: (u32, u32),
    size: (u32, u32),
}

struct JobResponse {
    image: image::RgbImage,
    offset: (u32, u32),
    size: (u32, u32),
}

impl JobResponse {
    pub fn new(image: image::RgbImage, offset: (u32, u32), size: (u32, u32)) -> Self {
        Self {
            image,
            offset,
            size,
        }
    }
}

enum NoiseGenAsyncJob {
    Job((u32, u32)),
    Terminate,
}

fn async_generate_grayscale_noise() {
    let (send, recv) = mpsc::channel::<i32>();

}

fn next_index() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

use crate::message_queue::MessageQueue;
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

fn generate_offset_grayscale_noise(sampler: &NoiseGenSampler, offset: (u32, u32), size: (u32, u32)) -> image::RgbImage {
    let mut img = image::RgbImage::new(size.0, size.1);
    for y in offset.1..offset.1 + size.1 {
        for x in offset.0..offset.0 + size.0 {
            let fx = x as f64;
            let fy = y as f64;
            let noise = sampler.sample(Point::new(fx, fy));
            let grey = (255.0 * noise) as u8;
            img.put_pixel(x - offset.0, y - offset.1, Rgb([grey, grey, grey]));
        }
    }
    img.into()
}

fn generate_grayscale_noise(noise_gui: &NoiseGenConfig, size: u32) -> image::RgbImage {
    let mut img = image::RgbImage::new(size, size);
    let noise_gen: NoiseGenSampler = noise_gui.clone().into();
    use rayon::prelude::*;
    use crate::size::Size;
    let size = Size::new(size, size);
    img.par_pixels_mut().enumerate().for_each(move |(i, pix)| {
        let (x, y) = size.inv_index(i as u32);
        let fx = x as f64;
        let fy = y as f64;
        let noise = noise_gen.sample(Point::new(fx, fy));
        let grey = (255.0 * noise) as u8;
        *pix = Rgb([grey, grey, grey]);
    });
    // if you ever want to see how much faster the parallel version is, you can comment out
    // the parallel version and uncomment the following code.
    // for y in 0..size.height {
    //     for x in 0..size.width {
    //         let fx = x as f64;
    //         let fy = y as f64;
    //         let noise = noise_gen.sample(Point::new(fx, fy));
    //         let grey = (255.0 * noise) as u8;
    //         img.put_pixel(x, y, Rgb([grey, grey, grey]));
    //     }
    // }
    img
}

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct NoiseEditorApp {
    noisegen_config: NoiseGenConfig,
    auto_generate: bool,
    show_grid: bool,
    image_size: u32,
    #[serde(skip)]
    texture: Option<TextureHandle>,
    #[serde(skip)]
    messages: MessageQueue,
}

impl Default for NoiseEditorApp {
    fn default() -> Self {
        Self {
            auto_generate: false,
            noisegen_config: NoiseGenConfig::default(),
            show_grid: true,
            image_size: 256,
            texture: None,
            messages: MessageQueue::new(500),
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
            let mut editor: Self = eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
            if editor.auto_generate {
                editor.generate_noise(&cc.egui_ctx, editor.image_size);
            }
            return editor;
        }

        Default::default()
    }
}

impl NoiseEditorApp {
    pub fn generate_noise(&mut self, ctx: &Context, size: u32) {
        let img = generate_grayscale_noise(&self.noisegen_config, size);
        let colorimg = convert_image_rgb(img);
        self.texture = Some(ctx.load_texture(format!("generation{}", next_index()), colorimg, TextureOptions::LINEAR));
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
                        if ui.button("Import").clicked() {
                            let file = rfd::FileDialog::new().add_filter("simp", &["simp"])
                                .set_title("Import Noise Gen Config")
                                .pick_file();
                            if let Some(path) = file {
                                if let Ok(config) = NoiseGenConfig::import(path) {
                                    self.noisegen_config = config;
                                } else {
                                    println!("Failed to import config!");
                                }
                            }
                        }
                        if ui.button("Export").clicked() {
                            let output = rfd::FileDialog::new().add_filter("simp", &["simp"])
                                .set_title("Export Noise Gen Config")
                                .save_file();
                            if let Some(path) = output {
                                if let Ok(_) = self.noisegen_config.export(path) {
                                    println!("Noise Gen Exported.");
                                } else {
                                    println!("Failed to export.");
                                }
                            }
                        }
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.menu_button("Diagnostics", |ui| {
                        if ui.button("Min and Max Height").clicked() {
                            let sampler: NoiseGenSampler = self.noisegen_config.clone().into();
                            let mut min = f64::MAX;
                            let mut max = f64::MIN;
                            for y in 0..self.image_size {
                                for x in 0..self.image_size {
                                    let xf = x as f64;
                                    let yf = y as f64;
                                    let sample = sampler.sample(Point::new(xf, yf));
                                    min = min.min(sample);
                                    max = max.max(sample);
                                }
                            }
                            self.messages.push(format!("The Min is {min} and the Max is {max}."));
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
                                    let grid_size = self.image_size / 16;
                                    let grid_width = 768. / grid_size as f32;
                                    for i in 0..(self.image_size / 16) {
                                        let n = i as f32 * grid_width;
                                        painter.line_segment([Pos2::new(rect.left(), rect.top() + n), Pos2::new(rect.right(), rect.top() + n)], Stroke::new(1.0, Color32::RED));
                                        painter.line_segment([Pos2::new(rect.left() + n, rect.top()), Pos2::new(rect.left() + n, rect.bottom())], Stroke::new(1.0, Color32::GREEN));
                                    }
                                }
                            });
                        });
                        ui.horizontal(|ui| {
                            let cur_size = self.image_size;
                            let resp = ComboBox::new("image_size_combo", "Image Size")
                            .selected_text(format!("{cur_size}x{cur_size}"))
                            .show_ui(ui, |ui| {
                                for i in 4..14 {
                                    let size = 1 << i;
                                    ui.selectable_value(&mut self.image_size, size, format!("{size}x{size}"));
                                }
                            });
                            if self.auto_generate && cur_size != self.image_size {
                                self.generate_noise(ctx, self.image_size);
                            }
                            ui.checkbox(&mut self.show_grid, "Show Grid");
                        });
                        ScrollArea::new([false, true])
                            .drag_to_scroll(false)
                            .max_width(f32::INFINITY)
                            .id_source("messages_scroll_area")
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                for msg in self.messages.iter() {
                                    ui.label(msg);
                                }
                            });
                    });
                });
                ui.vertical(|ui| {
                    let (rect, _) = ui.allocate_exact_size(Vec2::new(512., ui.spacing().interact_size.y), Sense::hover());
                    let button = egui::Button::new("Save").rounding(Rounding::ZERO);
                    if ui.put(rect, button).clicked() {
                        let img = generate_grayscale_noise(&self.noisegen_config, self.image_size);
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
                        // let img = generate_grayscale_noise(&self.noisegen_config);
                        // let colorimg = convert_image_rgb(img);
                        // self.texture = Some(ctx.load_texture(format!("generation{}", next_index()), colorimg, TextureOptions::LINEAR));
                        self.messages.push("Generating noise...");
                        let st = std::time::Instant::now();
                        self.generate_noise(ctx, self.image_size);
                        let el = st.elapsed();
                        self.messages.push(format!("Generation finished in {:.4} seconds.", el.as_secs_f64()));
                    }
                    ui.checkbox(&mut self.auto_generate, "Auto Generate");
                    let resp = self.noisegen_config.ui(ui);
                    if self.auto_generate && resp.changed() {
                        // let img = generate_grayscale_noise(&self.noisegen_config);
                        // let colorimg = convert_image_rgb(img);
                        // self.texture = Some(ctx.load_texture(format!("generation{}", next_index()), colorimg, TextureOptions::LINEAR));
                        self.generate_noise(ctx, self.image_size);
                    }
                });
            });
        });
    }
}
