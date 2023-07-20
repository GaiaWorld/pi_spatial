use super::util::ID;
use crate::quad_helper::{intersects, QuadTree as QuadTreeInner};
use nalgebra::Point2;
use parry2d::bounding_volume::Aabb as AABB;
use wasm_bindgen::prelude::wasm_bindgen;

/// aabb的查询函数的参数
pub struct AbQueryArgs {
    pub aabb: AABB,
    len: usize,
    pub result: Vec<f64>,
}
impl AbQueryArgs {
    pub fn new(aabb: AABB, len: usize) -> AbQueryArgs {
        AbQueryArgs {
            aabb: aabb,
            len,
            result: vec![],
        }
    }
}

/// ab节点的查询函数, 这里只是一个简单范本，使用了quad节点的查询函数intersects
/// 应用方为了功能和性能，应该实现自己需要的ab节点的查询函数， 比如点查询， 球查询-包含或相交， 视锥体查询...
pub fn ab_query_func(arg: &mut AbQueryArgs, id: ID, aabb: &AABB, bind: &i32) {
    // println!("ab_query_func: id: {}, bind:{:?}, arg: {:?}", id, bind, arg.result);
    if intersects(&arg.aabb, aabb) {
        if arg.result.len() <= arg.len {
            arg.result.push(id.0);
        }
    }
}

#[wasm_bindgen]
pub struct QuadTree(QuadTreeInner<ID, i32>);

#[wasm_bindgen]
impl QuadTree {
    pub fn default() -> Self {
        let max = nalgebra::Vector2::new(100f32, 100f32);
        let min = max / 100f32;

        Self(QuadTreeInner::new(
            AABB::new(
                Point2::new(-1024f32, -1024f32),
                Point2::new(3072f32, 3072f32),
            ),
            max,
            min,
            0,
            0,
            0,
        ))
    }

    /*
     * min_x & min_y: 场景最小边界
     * max_x & max_y: 场景最大边界
     * min_loose_x & min_loose_y: 场景物体最小尺寸
     * max_loose_x & max_loose_y: 场景物体最大尺寸
     */
    pub fn new(
        min_x: f32,
        min_y: f32,
        max_x: f32,
        max_y: f32,
        min_loose_x: f32,
        min_loose_y: f32,
        max_loose_x: f32,
        max_loose_y: f32,
    ) -> Self {
        let max = nalgebra::Vector2::new(max_loose_x, max_loose_y);
        let min = nalgebra::Vector2::new(min_loose_x, min_loose_y);

        Self(QuadTreeInner::new(
            AABB::new(Point2::new(min_x, min_y), Point2::new(max_x, max_y)),
            max,
            min,
            0,
            0,
            0,
        ))
    }

    pub fn add(&mut self, id: f64, min_x: f32, min_y: f32, max_x: f32, max_y: f32) {
        let min = Point2::new(min_x, min_y);
        let max = Point2::new(max_x, max_y);
        self.0.add(ID(id), AABB::new(min, max), 1);
    }

    pub fn remove(&mut self, id: f64) {
        self.0.remove(ID(id));
    }

    pub fn update(&mut self, id: f64, min_x: f32, min_y: f32, max_x: f32, max_y: f32) {
        let min = Point2::new(min_x, min_y);
        let max = Point2::new(max_x, max_y);
        self.0.update(ID(id), AABB::new(min, max));
    }

    pub fn query(&self, min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Vec<f64> {
        let min = Point2::new(min_x, min_y);
        let max = Point2::new(max_x, max_y);
        let ab = AABB::new(min, max);
        let mut args = AbQueryArgs::new(ab, usize::MAX);
        self.0
            .query(&AABB::new(min, max), intersects, &mut args, ab_query_func);

        let mut r = vec![];
        for i in args.result {
            r.push(i);
        }
        r
    }

    pub fn query_max(
        &self,
        min_x: f32,
        min_y: f32,
        max_x: f32,
        max_y: f32,
        result: &mut [u32],
        max_len: u32,
    ) -> f64 {
        let min = Point2::new(min_x, min_y);
        let max = Point2::new(max_x, max_y);
        let ab = AABB::new(min, max);
        let mut args = AbQueryArgs::new(ab, max_len as usize);
        self.0
            .query(&AABB::new(min, max), intersects, &mut args, ab_query_func);

        for i in 0..args.result.len() {
            result[i] = args.result[i] as u32;
        }
        args.result.len() as f64
    }
}
