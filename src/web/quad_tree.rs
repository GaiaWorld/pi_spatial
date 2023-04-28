use super::util::ID;
use crate::quad_helper::{intersects, QuadTree as QuadTreeInner};
use nalgebra::Point2;
use parry2d::bounding_volume::Aabb as AABB;
use wasm_bindgen::prelude::wasm_bindgen;

/// aabb的查询函数的参数
pub struct AbQueryArgs {
    pub aabb: AABB,
    pub result: Vec<f64>,
}
impl AbQueryArgs {
    pub fn new(aabb: AABB) -> AbQueryArgs {
        AbQueryArgs {
            aabb: aabb,
            result: vec![],
        }
    }
}

/// ab节点的查询函数, 这里只是一个简单范本，使用了quad节点的查询函数intersects
/// 应用方为了功能和性能，应该实现自己需要的ab节点的查询函数， 比如点查询， 球查询-包含或相交， 视锥体查询...
pub fn ab_query_func(arg: &mut AbQueryArgs, id: ID, aabb: &AABB, bind: &i32) {
    // println!("ab_query_func: id: {}, bind:{:?}, arg: {:?}", id, bind, arg.result);
    if intersects(&arg.aabb, aabb) {
        arg.result.push(id.0);
    }
}

#[wasm_bindgen]
pub struct QuadTree(QuadTreeInner<ID, i32>);

#[wasm_bindgen]
impl QuadTree {
    pub fn new() -> Self {
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

    pub fn add(&mut self, id: f64, min: &[f32], y: &[f32]) {
        let min = Point2::new(min[0], min[1]);
        let max = Point2::new(y[0], y[1]);
        self.0.add(ID(id), AABB::new(min, max), 1);
    }

    pub fn remove(&mut self, id: f64) {
        self.0.remove(ID(id));
    }

    pub fn update(&mut self, id: f64, min: &[f32], y: &[f32]) {
        let min = Point2::new(min[0], min[1]);
        let max = Point2::new(y[0], y[1]);
        self.0.update(ID(id), AABB::new(min, max));
    }

    pub fn query(&self, min: &[f32], max: &[f32]) -> Vec<f64> {
        let min = Point2::new(min[0], min[1]);
        let max = Point2::new(max[0], max[1]);
        let ab = AABB::new(min, max);
        let mut args = AbQueryArgs::new(ab);
        self.0
            .query(&AABB::new(min, max), intersects, &mut args, ab_query_func);

        let mut r = vec![];
        for i in args.result {
            r.push(i);
        }
        r
    }
}
