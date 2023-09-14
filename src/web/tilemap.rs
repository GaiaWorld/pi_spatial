use nalgebra::Point2;
use parry2d::bounding_volume::Aabb;
use pi_slotmap::{SlotMap, DefaultKey, Key, KeyData};
use wasm_bindgen::prelude::wasm_bindgen;

use crate::tilemap::TileMap as TileMapInner;

#[wasm_bindgen]
pub struct TileMapTree(TileMapInner<DefaultKey, i32>, SlotMap<DefaultKey, ()>);

#[wasm_bindgen]
impl TileMapTree {
    pub fn new(min_x: f32, min_y: f32, max_x: f32, max_y: f32, width: u32, height: u32) -> Self {
        let ab = Aabb::new(
            Point2::new(min_x, min_y),
            Point2::new(max_x, max_y),
        );
        Self(TileMapInner::new(ab, width as usize, height as usize), SlotMap::new())
    }

    pub fn add(&mut self, min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> f64 {
        let min = Point2::new(min_x, min_y);
        let max = Point2::new(max_x, max_y);
        let id = self.1.insert(());
        let res = id.data().as_ffi() as f64;
        self.0.add(id, Aabb::new(min, max), 1);
        res
    }

    pub fn remove(&mut self, id: f64) {
        self.0.remove(DefaultKey::from(KeyData::from_ffi(id as u64)));
    }

    pub fn update(&mut self, id: f64, min_x: f32, min_y: f32, max_x: f32, max_y: f32,) {
        let min = Point2::new(min_x, min_y);
        let max = Point2::new(max_x, max_y);
        self.0.update(DefaultKey::from(KeyData::from_ffi(id as u64)), Aabb::new(min, max));
    }

    pub fn query(&self, min_x: f32, min_y: f32, max_x: f32, max_y: f32,) -> Vec<u32> {
        let min = Point2::new(min_x, min_y);
        let max = Point2::new(max_x, max_y);
        let (len, iter) = self.0.query_iter(&Aabb::new(min, max));
        let mut result = Vec::with_capacity(len);
        for item in iter.into_iter() {
            result.push(item as u32);
        }
        result
    }
}
