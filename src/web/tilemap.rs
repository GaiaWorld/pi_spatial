use nalgebra::Point2;
use parry2d::bounding_volume::Aabb;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::tilemap::TileMap as TileMapInner;

use super::util::ID;

#[wasm_bindgen]
pub struct TileMapTree(TileMapInner<ID, i32>);

#[wasm_bindgen]
impl TileMapTree {
    pub fn new(ab_min: &[f32], ab_max: &[f32], row: u32, column: u32) -> Self {
        let ab = Aabb::new(
            Point2::new(ab_min[0], ab_min[1]),
            Point2::new(ab_max[0], ab_max[0]),
        );
        Self(TileMapInner::new(ab, row as usize, column as usize))
    }

    pub fn add(&mut self, id: f64, min: &[f32], y: &[f32]) {
        let min = Point2::new(min[0], min[1]);
        let max = Point2::new(y[0], y[1]);
        self.0.add(ID(id), Aabb::new(min, max), 1);
    }

    pub fn remove(&mut self, id: f64) {
        self.0.remove(ID(id));
    }

    pub fn update(&mut self, id: f64, min: &[f32], y: &[f32]) {
        let min = Point2::new(min[0], min[1]);
        let max = Point2::new(y[0], y[1]);
        self.0.update(ID(id), Aabb::new(min, max));
    }

    pub fn query(&self, min: &[f32], y: &[f32]) -> Vec<u32> {
        let min = Point2::new(min[0], min[1]);
        let max = Point2::new(y[0], y[1]);
        let (len, iter) = self.0.query_iter(&Aabb::new(min, max));
        let mut result = Vec::with_capacity(len);
        for item in iter.into_iter() {
            result.push(item as u32);
        }
        result
    }
}
