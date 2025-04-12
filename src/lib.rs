#![warn(clippy::all, rust_2018_idioms)]

mod app;
mod noisegen;
mod gen;
mod message_queue;
mod size;
pub use app::NoiseEditorApp;