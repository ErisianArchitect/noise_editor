#![allow(unused)]

use crate::noisegen::Point;


#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct OctaveGen {
    persistence: f64,
    lacunarity: f64,
    initial_amplitude: f64,
    initial_frequency: f64,
}

struct NoiseLayer {
    noise: f64,
    amplitude: f64,
    frequency: f64
}

impl NoiseLayer {
    pub const fn new(noise: f64, amplitude: f64, frequency: f64) -> Self {
        Self {
            noise,
            amplitude,
            frequency
        }
    }
}

impl OctaveGen {
    pub fn sample<F: NoiseFn, It: IntoIterator<Item = F>>(&self, point: Point, layers: It) -> f64 {
        let init = NoiseLayer::new(
            0.0,
            self.initial_amplitude,
            self.initial_frequency
        );
        let result = layers.into_iter().fold(init, |mut accum, noise| {
            accum.noise += accum.amplitude * noise.sample_noise(Point::new(point.x * accum.frequency, point.y * accum.frequency));
            accum.amplitude *= self.persistence;
            accum.frequency *= self.lacunarity;
            accum
        });
        result.noise
    }
}

pub trait NoiseFn {
    fn sample_noise(&self, point: Point) -> f64;
}