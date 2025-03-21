use nalgebra::Point2;
use parry2d::bounding_volume::Aabb;
use parry2d::bounding_volume::Aabb as AABB;
use pi_slotmap::{SlotMap, DefaultKey, Key, KeyData};
use wasm_bindgen::prelude::wasm_bindgen;
use nalgebra::Vector2;
use crate::quad_helper::intersects;
use crate::tilemap::TileMap as TileMapInner;
use super::quad_tree::{AbQueryArgs, ab_query_func};

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
    pub fn shift(&mut self, id: f64, x: f32, y: f32) {
        self.0.shift(DefaultKey::from(KeyData::from_ffi(id as u64)), Vector2::new(x, y));
    }
    pub fn move_to(&mut self, id: f64, x: f32, y: f32) {
        self.0.move_to(DefaultKey::from(KeyData::from_ffi(id as u64)), Point2::new(x, y));
    }

    pub fn query(&self, min_x: f32, min_y: f32, max_x: f32, max_y: f32,) -> Vec<f64> {
        let min = Point2::new(min_x, min_y);
        let max = Point2::new(max_x, max_y);
        let ab = AABB::new(min, max);
        let mut args = AbQueryArgs::new(ab, usize::MAX);
        self.0.query(&AABB::new(min, max), &mut args, ab_query_func);
        args.result
    }
}
