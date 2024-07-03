use crate::quad_helper::{intersects, QuadTree as QuadTreeInner};
use nalgebra::Point2;
use parry2d::bounding_volume::Aabb as AABB;
use pi_slotmap::{DefaultKey, Key, KeyData, SlotMap};
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
pub fn ab_query_func(arg: &mut AbQueryArgs, id: DefaultKey, aabb: &AABB, bind: &i32) {
    // println!("ab_query_func: id: {}, bind:{:?}, arg: {:?}", id, bind, arg.result);
    if intersects(&arg.aabb, aabb) {
        if arg.result.len() <= arg.len {
            arg.result.push(id.data().as_ffi() as f64);
        }
    }
}

#[wasm_bindgen]
pub struct QuadTree(QuadTreeInner<DefaultKey, i32>, SlotMap<DefaultKey, ()>);

#[wasm_bindgen]
impl QuadTree {
    pub fn default() -> Self {
        let max = nalgebra::Vector2::new(100f32, 100f32);
        let min = max / 100f32;

        Self(
            QuadTreeInner::new(
                AABB::new(
                    Point2::new(-1024f32, -1024f32),
                    Point2::new(3072f32, 3072f32),
                ),
                max,
                min,
                0,
                0,
                0,
            ),
            SlotMap::new(),
        )
    }

    /*
     * min_x & min_y: 场景最小边界
     * max_x & max_y: 场景最大边界
     * min_loose_x & min_loose_y: 场景物体最小尺寸
     * max_loose_x & max_loose_y: 场景物体最大尺寸
     */
    pub fn new(
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
        min_loose_x: f64,
        min_loose_y: f64,
        max_loose_x: f64,
        max_loose_y: f64,
    ) -> Self {
        let max = nalgebra::Vector2::new(max_loose_x as f32, max_loose_y as f32);
        let min = nalgebra::Vector2::new(min_loose_x as f32, min_loose_y as f32);

        Self(
            QuadTreeInner::new(
                AABB::new(
                    Point2::new(min_x as f32, min_y as f32),
                    Point2::new(max_x as f32, max_y as f32),
                ),
                max,
                min,
                0,
                0,
                0,
            ),
            SlotMap::new(),
        )
    }

    pub fn add(&mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> f64 {
        let min = Point2::new(min_x as f32, min_y as f32);
        let max = Point2::new(max_x as f32, max_y as f32);
        let id = self.1.insert(());
        let res = id.data().as_ffi() as f64;
        self.0.add(id, AABB::new(min, max), 1);
        res
    }

    pub fn remove(&mut self, id: f64) {
        self.0
            .remove(DefaultKey::from(KeyData::from_ffi(id as u64)));
    }

    pub fn update(&mut self, id: f64, min_x: f64, min_y: f64, max_x: f64, max_y: f64) {
        let min = Point2::new(min_x as f32, min_y as f32);
        let max = Point2::new(max_x as f32, max_y as f32);
        self.0.update(
            DefaultKey::from(KeyData::from_ffi(id as u64)),
            AABB::new(min, max),
        );
    }

    pub fn query(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<f64> {
        let min = Point2::new(min_x as f32, min_y as f32);
        let max = Point2::new(max_x as f32, max_y as f32);
        let ab = AABB::new(min, max);
        let mut args = AbQueryArgs::new(ab, usize::MAX);
        self.0
            .query(&AABB::new(min, max), intersects, &mut args, ab_query_func);

        args.result
    }

    pub fn query_max(
        &self,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
        result: &mut [f64],
        max_len: u32,
    ) -> f64 {
        let min = Point2::new(min_x as f32, min_y as f32);
        let max = Point2::new(max_x as f32, max_y as f32);
        let ab = AABB::new(min, max);
        let mut args = AbQueryArgs::new(ab, max_len as usize);
        self.0
            .query(&AABB::new(min, max), intersects, &mut args, ab_query_func);

        for i in 0..args.result.len() {
            result[i] = args.result[i] as f64;
        }
        args.result.len() as f64
    }
}

// #[test]
// fn test1() {
//     let mut tree = QuadTree::default();
//     // let id1 = tree.add(
//     //     35.6057014465332,
//     //     36.574710845947266,
//     //     46.752288818359375,
//     //     36.574710845947266,
//     // );
//     // let id2 = tree.add(
//     //     35.6057014465332,
//     //     36.574710845947266,
//     //     39.15443420410156,
//     //     36.574710845947266,
//     // );

//     let id1 = tree.add(
//         -50.,
//         -50.00000762939453,
//         49.95399475097656,
//         -10.000000953674316,
//     );
//     let id2 = tree.add(
//         10.,
//         -50.00000762939453,
//         49.95399475097656,
//         -10.000000953674316,
//     );

//     let mut result = vec![0; 423];
//     tree.query_max(1., 1., 1., 1., &mut result, 423);
// }
