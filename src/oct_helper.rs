//! 八叉相关接口

use std::marker::PhantomData;
use std::mem;

use nalgebra::*;
use ncollide3d::bounding_volume::*;
use num_traits::{Float, FromPrimitive};

use crate::*;

/// 八叉树
pub type OctTree<K, S, T> = Tree<K, OctHelper<S>, T, 8>;

#[derive(Debug, Clone)]
pub struct OctHelper<S: Scalar + RealField + Float> {
    phantom: PhantomData<S>,
}

impl<S: Scalar + RealField + Float> Helper<8> for OctHelper<S> {
    type Point = Point3<S>;
    type Vector = Vector3<S>;
    type Aabb = AABB<S>;

    /// 获得AABB的差
    fn aabb_extents(aabb: &AABB<S>) -> Vector3<S> {
        aabb.extents()
    }
    /// 移动AABB
    fn aabb_shift(aabb: &AABB<S>, distance: &Vector3<S>) -> AABB<S> {
        AABB::new(aabb.mins + distance, aabb.maxs + distance)
    }
    /// 判断指定的aabb是否包含另一个aabb
    fn aabb_contains(aabb: &AABB<S>, other: &AABB<S>) -> bool {
        aabb.contains(other)
    }
    /// 判断2个aabb是否相交
    fn aabb_intersects(aabb: &AABB<S>, other: &AABB<S>) -> bool {
        aabb.intersects(other)
    }
    /// 计算八叉树的深度
    fn get_deap(
        d: &mut Vector3<S>,
        loose_layer: usize,
        max_loose: &Vector3<S>,
        deep: usize,
        min_loose: &Vector3<S>,
    ) -> usize {
        let two = S::one() + S::one();
        let x = ComplexField::powf(
            (max_loose.x / d.x + S::one()) / two,
            FromPrimitive::from_usize(loose_layer).unwrap(),
        );
        let y = ComplexField::powf(
            (max_loose.y / d.y + S::one()) / two,
            FromPrimitive::from_usize(loose_layer).unwrap(),
        );
        let z = ComplexField::powf(
            (max_loose.z / d.z + S::one()) / two,
            FromPrimitive::from_usize(loose_layer).unwrap(),
        );
        d.x *= x;
        d.y *= y;
        d.z *= z;
        let deep = if loose_layer < deep {
            // 高于该层的节点，松散值都是用最小值， 也可计算其下每层的八叉节点的大小
            // 八叉节点的大小如果小于最小松散值的2倍， 应该停止向下划分， 因为最小松散值占据了八叉节点的大部分
            // 最大层由设置值和该停止划分的层的最小值
            let mut calc_deep = loose_layer;
            let min = min_loose * two;
            while calc_deep < deep && d.x >= min.x && d.y >= min.y && d.z >= min.z {
                *d = (*d + min_loose) / two;
                calc_deep += 1;
            }
            calc_deep
        } else {
            deep
        };
        deep
    }

    #[inline]
    /// 判定指定向量是否小于最小“松散”尺寸
    fn smaller_than_min_loose(d: &Vector3<S>, min_loose: &Vector3<S>) -> bool {
        if d.x <= min_loose.x && d.y <= min_loose.y && d.z <= min_loose.z {
            return true;
        };
        return false;
    }

    #[inline]
    /// 指定向量以及最大松散尺寸计算对应的层
    fn calc_layer(loose: &Vector3<S>, el: &Vector3<S>) -> usize {
        let x = if el.x == S::zero() {
            usize::max_value()
        } else {
            (loose.x / el.x).to_usize().unwrap()
        };
        let y = if el.y == S::zero() {
            usize::max_value()
        } else {
            (loose.y / el.y).to_usize().unwrap()
        };
        let z = if el.z == S::zero() {
            usize::max_value()
        } else {
            (loose.z / el.z).to_usize().unwrap()
        };
        let min = x.min(y).min(z);
        if min == 0 {
            return 0;
        }
        (mem::size_of::<usize>() << 3) - (min.leading_zeros() as usize) - 1
    }

    #[inline]
    /// 判断所在的子节点
    fn get_child(point: &Point3<S>, aabb: &AABB<S>) -> usize {
        let mut i: usize = 0;
        if aabb.maxs.x > point.x {
            i += 1;
        }
        if aabb.maxs.y > point.y {
            i += 2;
        }
        if aabb.maxs.z > point.z {
            i += 4;
        }
        i
    }

    #[inline]
    fn get_max_half_loose(aabb: &AABB<S>, loose: &Vector3<S>) -> Point3<S> {
        let two = S::one() + S::one();
        let x = (aabb.mins.x + aabb.maxs.x + loose.x) / two;
        let y = (aabb.mins.y + aabb.maxs.y + loose.y) / two;
        let z = (aabb.mins.z + aabb.maxs.z + loose.z) / two;
        Point3::new(x, y, z)
    }

    /// 创建ab的子节点集合
    fn make_childs(aabb: &AABB<S>, loose: &Vector3<S>) -> [AABB<S>; 8] {
        let two = S::one() + S::one();
        let x = (aabb.mins.x + aabb.maxs.x - loose.x) / two;
        let y = (aabb.mins.y + aabb.maxs.y - loose.y) / two;
        let z = (aabb.mins.z + aabb.maxs.z - loose.z) / two;
        let p1 = Point3::new(x, y, z);
        let p2 = Self::get_max_half_loose(&aabb, &loose);
        [
            AABB::new(aabb.mins, p2),
            AABB::new(
                Point3::new(p1.x, aabb.mins.y, aabb.mins.z),
                Point3::new(aabb.maxs.x, p2.y, p2.z),
            ),
            AABB::new(
                Point3::new(aabb.mins.x, p1.y, aabb.mins.z),
                Point3::new(p2.x, aabb.maxs.y, p2.z),
            ),
            AABB::new(
                Point3::new(p1.x, p1.y, aabb.mins.z),
                Point3::new(aabb.maxs.x, aabb.maxs.y, p2.z),
            ),
            AABB::new(
                Point3::new(aabb.mins.x, aabb.mins.y, p1.z),
                Point3::new(p2.x, p2.y, aabb.maxs.z),
            ),
            AABB::new(
                Point3::new(p1.x, aabb.mins.y, p1.z),
                Point3::new(aabb.maxs.x, p2.y, aabb.maxs.z),
            ),
            AABB::new(
                Point3::new(aabb.mins.x, p1.y, p1.z),
                Point3::new(p2.x, aabb.maxs.y, aabb.maxs.z),
            ),
            AABB::new(p1, aabb.maxs),
        ]
    }

    /// 指定创建ab的子节点
    fn create_child(
        aabb: &AABB<S>,
        loose: &Vector3<S>,
        layer: usize,
        loose_layer: usize,
        min_loose: &Vector3<S>,
        index: usize,
    ) -> (AABB<S>, Vector3<S>) {
        let two = S::one() + S::one();
        macro_rules! c1 {
            ($c:ident) => {
                (aabb.mins.$c + aabb.maxs.$c - loose.$c) / two
            };
        }
        macro_rules! c2 {
            ($c:ident) => {
                (aabb.mins.$c + aabb.maxs.$c + loose.$c) / two
            };
        }
        let a = match index {
            0 => AABB::new(aabb.mins, Point3::new(c2!(x), c2!(y), c2!(z))),
            1 => AABB::new(
                Point3::new(c1!(x), aabb.mins.y, aabb.mins.z),
                Point3::new(aabb.maxs.x, c2!(y), c2!(z)),
            ),
            2 => AABB::new(
                Point3::new(aabb.mins.x, c1!(y), aabb.mins.z),
                Point3::new(c2!(x), aabb.maxs.y, c2!(z)),
            ),
            3 => AABB::new(
                Point3::new(c1!(x), c1!(y), aabb.mins.z),
                Point3::new(aabb.maxs.x, aabb.maxs.y, c2!(z)),
            ),
            4 => AABB::new(
                Point3::new(aabb.mins.x, aabb.mins.y, c1!(z)),
                Point3::new(c2!(x), c2!(y), aabb.maxs.z),
            ),
            5 => AABB::new(
                Point3::new(c1!(x), aabb.mins.y, c1!(z)),
                Point3::new(aabb.maxs.x, c2!(y), aabb.maxs.z),
            ),
            6 => AABB::new(
                Point3::new(aabb.mins.x, c1!(y), c1!(z)),
                Point3::new(c2!(x), aabb.maxs.y, aabb.maxs.z),
            ),
            _ => AABB::new(Point3::new(c1!(x), c1!(y), c1!(z)), aabb.maxs),
        };
        let loose = if layer < loose_layer {
            loose / two
        } else {
            min_loose.clone()
        };
        (a, loose)
    }
}

/// oct节点查询函数的范本，aabb是否相交，参数a是查询参数，参数b是oct节点的aabb， 所以最常用的判断是左闭右开
/// 应用方为了功能和性能，应该实现自己需要的oct节点的查询函数， 比如点查询， 球查询， 视锥体查询...
#[inline]
pub fn intersects<S: Scalar + RealField + Float>(a: &AABB<S>, b: &AABB<S>) -> bool {
    a.mins.x <= b.maxs.x
        && a.maxs.x > b.mins.x
        && a.mins.y <= b.maxs.y
        && a.maxs.y > b.mins.y
        && a.mins.z <= b.maxs.z
        && a.maxs.z > b.mins.z
}

/// aabb的查询函数的参数
pub struct AbQueryArgs<S: Scalar + RealField + Float, T> {
    pub aabb: AABB<S>,
    pub result: Vec<(usize, T)>,
}
impl<S: Scalar + RealField + Float, T: Clone> AbQueryArgs<S, T> {
    pub fn new(aabb: AABB<S>) -> AbQueryArgs<S, T> {
        AbQueryArgs {
            aabb: aabb,
            result: Vec::new(),
        }
    }
}

/// ab节点的查询函数, 这里只是一个简单范本，使用了oct节点的查询函数intersects
/// 应用方为了功能和性能，应该实现自己需要的ab节点的查询函数， 比如点查询， 球查询-包含或相交， 视锥体查询...
pub fn ab_query_func<S: Scalar + RealField + Float, T: Clone>(
    arg: &mut AbQueryArgs<S, T>,
    id: usize,
    aabb: &AABB<S>,
    bind: &T,
) {
    if intersects(&arg.aabb, aabb) {
        arg.result.push((id, bind.clone()));
    }
}



// #[test]
// fn test1() {
//     println!("test1-----------------------------------------");
//     let max = Vector3::new(100f32, 100f32, 100f32);
//     let min = max / 100f32;
//     let mut tree = OctTree::new(
//         AABB::new(
//             Point3::new(-1024f32, -1024f32, -4194304f32),
//             Point3::new(3072f32, 3072f32, 4194304f32),
//         ),
//         max,
//         min,
//         0,
//         0,
//         0,
//     );
//     for i in 0..1 {
//         tree.add(
//             i + 1,
//             AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//             i + 1,
//         );
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("00000, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     tree.update(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("00000, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     tree.collect();
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("00000, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     for i in 1..5 {
//         tree.add(
//             i + 1,
//             AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//             i + 3,
//         );
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("00001, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     tree.update(
//         2,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         3,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );

//     tree.update(
//         4,
//         AABB::new(
//             Point3::new(0.0, 700.0, 0.0),
//             Point3::new(1000.0, 1400.0, 1.0),
//         ),
//     );

//     tree.update(
//         5,
//         AABB::new(
//             Point3::new(0.0, 1400.0, 0.0),
//             Point3::new(1000.0, 1470.0, 1.0),
//         ),
//     );
//     tree.update(
//         6,
//         AABB::new(
//             Point3::new(0.0, 1470.0, 0.0),
//             Point3::new(1000.0, 1540.0, 1.0),
//         ),
//     );
//     tree.update(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.collect();
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("00002, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     //   tree.update(1, AABB::new(Point3::new(0.0,0.0,0.0), Point3::new(1000.0, 800.0, 1.0)));
//     //   tree.update(2, AABB::new(Point3::new(0.0,0.0,0.0), Point3::new(1000.0, 800.0, 1.0)));
//     //   tree.update(3, AABB::new(Point3::new(0.0,0.0,0.0), Point3::new(1000.0, 800.0, 1.0)));
//     //   tree.update(4, AABB::new(Point3::new(0.0,0.0,0.0), Point3::new(1000.0, 800.0, 1.0)));

//     //   tree.update(5, AABB::new(Point3::new(0.0,800.0,0.0), Point3::new(1000.0, 1600.0, 1.0)));

//     //    tree.update(6, AABB::new(Point3::new(0.0,1600.0,0.0), Point3::new(1000.0, 2400.0, 1.0)));
//     //   tree.update(7, AABB::new(Point3::new(0.0,2400.0,0.0), Point3::new(1000.0, 3200.0, 1.0)));
//     //   for i in 1..tree.ab_map.len() + 1 {
//     //   println!("22222, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     //  }
//     // tree.collect();
//     for i in 1..tree.slab.len() + 1 {
//         println!(
//             "000000 000000, id:{}, oct: {:?}",
//             i,
//             tree.slab.get(i).unwrap()
//         );
//     }
//     println!("outer:{:?}", tree.outer);
//     let aabb = AABB::new(
//         Point3::new(500f32, 500f32, -4194304f32),
//         Point3::new(500f32, 500f32, 4194304f32),
//     );
//     let mut args: AbQueryArgs<f32, usize> = AbQueryArgs::new(aabb.clone());
//     tree.query(&aabb, intersects, &mut args, ab_query_func);
//     //assert_eq!(args.result(), [1, 3, 4]);
// }

// #[test]
// fn test2() {
//     println!("test2-----------------------------------------");
//     let max = Vector3::new(100f32, 100f32, 100f32);
//     let min = max / 100f32;
//     let mut tree = OctTree::new(
//         AABB::new(
//             Point3::new(-1024f32, -1024f32, -4194304f32),
//             Point3::new(3072f32, 3072f32, 4194304f32),
//         ),
//         max,
//         min,
//         0,
//         0,
//         0,
//     );
//     for i in 0..9 {
//         tree.add(
//             i + 1,
//             AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//             i + 1,
//         );
//     }
//     for i in 1..tree.slab.len() + 1 {
//         println!("000000, id:{}, oct: {:?}", i, tree.slab.get(i).unwrap());
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("00000, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     tree.update(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 1.0)),
//     );
//     tree.collect();
//     for i in 1..tree.slab.len() + 1 {
//         println!(
//             "000000 000000, id:{}, oct: {:?}",
//             i,
//             tree.slab.get(i).unwrap()
//         );
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!(
//             "000000 000000, id:{}, ab: {:?}",
//             i,
//             tree.ab_map.get(i).unwrap()
//         );
//     }
//     println!("tree -new ------------------------------------------");
//     let max = Vector3::new(100f32, 100f32, 100f32);
//     let min = max / 100f32;
//     let mut tree = OctTree::new(
//         AABB::new(
//             Point3::new(-1024f32, -1024f32, -4194304f32),
//             Point3::new(3072f32, 3072f32, 4194304f32),
//         ),
//         max,
//         min,
//         0,
//         0,
//         0,
//     );
//     for i in 0..6 {
//         tree.add(
//             i + 1,
//             AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(0.1, 0.1, 0.1)),
//             i + 1,
//         );
//     }
//     for i in 1..tree.slab.len() + 1 {
//         println!("test1, id:{}, oct: {:?}", i, tree.slab.get(i).unwrap());
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("test1, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     tree.collect();
//     for i in 1..tree.slab.len() + 1 {
//         println!("test2, id:{}, oct: {:?}", i, tree.slab.get(i).unwrap());
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("test2, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     tree.shift(4, Vector3::new(2.0, 2.0, 1.0));
//     tree.shift(5, Vector3::new(4.0, 4.0, 1.0));
//     tree.shift(6, Vector3::new(10.0, 10.0, 1.0));
//     for i in 1..tree.slab.len() + 1 {
//         println!("test3, id:{}, oct: {:?}", i, tree.slab.get(i).unwrap());
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("test3, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     tree.collect();
//     for i in 1..tree.slab.len() + 1 {
//         println!("test4, id:{}, oct: {:?}", i, tree.slab.get(i).unwrap());
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("test4, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     println!("outer:{:?}", tree.outer);
//     let aabb = AABB::new(
//         Point3::new(0.05f32, 0.05f32, 0f32),
//         Point3::new(0.05f32, 0.05f32, 1000f32),
//     );
//     let mut args: AbQueryArgs<f32, usize> = AbQueryArgs::new(aabb.clone());
//     tree.query(&aabb, intersects, &mut args, ab_query_func);
//     //assert_eq!(args.result(), [1, 2, 3]);
// }

// #[test]
// fn test3() {
//     let z_max: f32 = 4194304.0;
//     let aabb = AABB::new(
//         Point3::new(700.0, 100.0, -z_max),
//         Point3::new(700.0, 100.0, z_max),
//     );
//     let mut args: AbQueryArgs<f32, usize> = AbQueryArgs::new(aabb.clone());

//     // let mut tree = Tree::new(AABB::new(Point3::new(0f32,0f32,0f32), Point3::new(1000f32,1000f32,1000f32)),
//     // 	0,
//     // 	0,
//     // 	0,
//     // 	0,
//     // );
//     let max = Vector3::new(100f32, 100f32, 100f32);
//     let min = max / 100f32;
//     let mut tree = OctTree::new(
//         AABB::new(
//             Point3::new(-1024f32, -1024f32, -4194304f32),
//             Point3::new(3072f32, 3072f32, 4194304f32),
//         ),
//         max,
//         min,
//         0,
//         0,
//         0,
//     );
//     tree.add(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         2,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.collect();

//     tree.update(
//         0,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         2,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );

//     tree.add(
//         3,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         4,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         5,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         6,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         7,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         8,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         9,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.collect();
//     tree.update(
//         3,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 350.0, 1.0)),
//     );
//     tree.update(
//         4,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(800.0, 175.0, 1.0)),
//     );
//     tree.update(
//         5,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(640.0, 140.0, 1.0)),
//     );
//     tree.update(
//         6,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(512.0, 112.0, 1.0)),
//     );
//     tree.update(
//         7,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(410.0, 90.0, 1.0)),
//     );
//     tree.update(
//         8,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1600.0, 175.0, 1.0),
//         ),
//     );
//     tree.update(
//         9,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1440.0, 140.0, 1.0),
//         ),
//     );
//     tree.update(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         2,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.query(&aabb, intersects, &mut args, ab_query_func);
//     tree.remove(7);
//     tree.remove(6);
//     tree.remove(5);

//     tree.add(
//         5,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         6,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         7,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.collect();
//     tree.update(
//         5,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(640.0, 140.0, 1.0)),
//     );
//     tree.update(
//         6,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(512.0, 112.0, 1.0)),
//     );
//     tree.update(
//         7,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(410.0, 90.0, 1.0)),
//     );
//     tree.update(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         2,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         3,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 350.0, 1.0)),
//     );
//     tree.update(
//         4,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(800.0, 175.0, 1.0)),
//     );
//     tree.update(
//         8,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1600.0, 175.0, 1.0),
//         ),
//     );
//     tree.update(
//         9,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1440.0, 140.0, 1.0),
//         ),
//     );
//     for i in 1..tree.slab.len() + 1 {
//         println!("test||||||, id:{}, oct: {:?}", i, tree.slab.get(i));
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("test----------, id:{}, ab: {:?}", i, tree.ab_map.get(i));
//     }
//     println!(
//         "-------------------------------------------------------dirtys:{:?}",
//         tree.dirty
//     );
//     tree.query(&aabb, intersects, &mut args, ab_query_func);

//     tree.remove(7);
//     tree.remove(6);
//     tree.remove(5);

//     tree.add(
//         5,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         6,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         7,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.collect();

//     tree.update(
//         5,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(640.0, 140.0, 1.0)),
//     );
//     tree.update(
//         6,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(512.0, 112.0, 1.0)),
//     );
//     tree.update(
//         7,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(410.0, 90.0, 1.0)),
//     );
//     tree.update(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         2,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         3,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 350.0, 1.0)),
//     );
//     tree.update(
//         4,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(800.0, 175.0, 1.0)),
//     );
//     tree.update(
//         8,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1600.0, 175.0, 1.0),
//         ),
//     );
//     tree.update(
//         9,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1440.0, 140.0, 1.0),
//         ),
//     );
//     tree.query(&aabb, intersects, &mut args, ab_query_func);
//     for i in 1..tree.slab.len() + 1 {
//         println!("test||||||, id:{}, oct: {:?}", i, tree.slab.get(i));
//     }
//     for i in 1..tree.ab_map.len() + 10 {
//         println!("test----------, id:{}, ab: {:?}", i, tree.ab_map.get(i));
//     }
//     println!(
//         "-------------------------------------------------------outer:{:?}",
//         tree.outer
//     );
// }

// #[cfg(test)]
// extern crate pcg_rand;
// #[cfg(test)]
// extern crate rand;
// #[test]
// fn test4() {
//     use rand;
//     use rand::Rng;
//     let z_max: f32 = 4194304.0;
//     let aabb = AABB::new(
//         Point3::new(700.0, 100.0, -z_max),
//         Point3::new(700.0, 100.0, z_max),
//     );
//     let mut args: AbQueryArgs<f32, usize> = AbQueryArgs::new(aabb.clone());

//     let max = Vector3::new(100f32, 100f32, 100f32);
//     let min = max / 100f32;
//     let mut tree = OctTree::new(
//         AABB::new(
//             Point3::new(-1024f32, -1024f32, -4194304f32),
//             Point3::new(3072f32, 3072f32, 4194304f32),
//         ),
//         max,
//         min,
//         0,
//         0,
//         0,
//     );
//     tree.add(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         2,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.collect();

//     tree.update(
//         0,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         2,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );

//     tree.add(
//         3,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         4,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         5,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         6,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         7,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         8,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         9,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.collect();
//     tree.update(
//         3,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 350.0, 1.0)),
//     );
//     tree.update(
//         4,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(800.0, 175.0, 1.0)),
//     );
//     tree.update(
//         5,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(640.0, 140.0, 1.0)),
//     );
//     tree.update(
//         6,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(512.0, 112.0, 1.0)),
//     );
//     tree.update(
//         7,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(410.0, 90.0, 1.0)),
//     );
//     tree.update(
//         8,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1600.0, 175.0, 1.0),
//         ),
//     );
//     tree.update(
//         9,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1440.0, 140.0, 1.0),
//         ),
//     );
//     tree.update(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         2,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );

//     tree.add(
//         10,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         11,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         12,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.collect();
//     tree.update(
//         10,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(640.0, 140.0, 1.0)),
//     );
//     tree.update(
//         11,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(512.0, 112.0, 1.0)),
//     );
//     tree.update(
//         12,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(410.0, 90.0, 1.0)),
//     );
//     tree.update(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         2,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         3,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 350.0, 1.0)),
//     );
//     tree.update(
//         4,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(800.0, 175.0, 1.0)),
//     );
//     tree.update(
//         8,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1600.0, 175.0, 1.0),
//         ),
//     );
//     tree.update(
//         9,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1440.0, 140.0, 1.0),
//         ),
//     );

//     tree.add(
//         13,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         14,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         15,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)),
//         1,
//     );
//     tree.collect();

//     //log(&tree.slab, &tree.ab_map, 10000);
//     tree.update(
//         13,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(640.0, 140.0, 1.0)),
//     );
//     //log(&tree.slab, &tree.ab_map, 13);
//     println!("oct========================");
//     tree.update(
//         14,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(512.0, 112.0, 1.0)),
//     );
//     tree.update(
//         15,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(410.0, 90.0, 1.0)),
//     );
//     tree.update(
//         1,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         2,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 700.0, 1.0)),
//     );
//     tree.update(
//         3,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1000.0, 350.0, 1.0)),
//     );
//     tree.update(
//         4,
//         AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(800.0, 175.0, 1.0)),
//     );
//     tree.update(
//         8,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1600.0, 175.0, 1.0),
//         ),
//     );
//     tree.update(
//         9,
//         AABB::new(
//             Point3::new(800.0, 0.0, 0.0),
//             Point3::new(1440.0, 140.0, 1.0),
//         ),
//     );
//     let mut rng = rand::thread_rng();
//     for _ in 0..1000000 {
//         let r = rand::thread_rng().gen_range(1, 16);
//         let x = rand::thread_rng().gen_range(0, 128) - 64;
//         let y = rand::thread_rng().gen_range(0, 128) - 64;
//         let old = (tree.slab.clone(), tree.ab_map.clone());
//         tree.shift(r, Vector3::new(x as f32, y as f32, 0.0));
//         //assert_eq!(check_tree(&tree.slab, &tree.ab_map, old, r), false);
//         if x + y > 120 {
//             //let old = clone_tree(tree.slab.clone(), tree.ab_map.clone());
//             tree.collect();
//             //assert_eq!(check_tree(&tree.slab, &tree.ab_map, old, r), false);
//         }
//     }
// }


// // #[cfg(test)]
// // fn check_tree(
// //     slab: &Slab<OctNode<f32>>,
// //     ab_map: &VecMap<AbNode<f32, usize>>,
// //     old: (Slab<OctNode<f32>>, VecMap<AbNode<f32, usize>>),
// //     r: usize,
// // ) -> bool {
// //     for (id, _n) in slab.iter() {
// //         if check_ab(slab, ab_map, id) {
// //             log(&old.0, &old.1, r);
// //             log(slab, ab_map, id);
// //             return true;
// //         }
// //     }
// //     false
// // }
// // #[cfg(test)]
// // fn check_ab(
// //     slab: &Slab<OctNode<f32>>,
// //     ab_map: &VecMap<AbNode<f32, usize>>,
// //     oct_id: usize,
// // ) -> bool {
// //     let node = unsafe { slab.get_unchecked(oct_id) };
// //     let mut old = VecMap::default();
// //     for c in 0..8 {
// //         match &node.childs[c] {
// //             ChildNode::Ab(list) => {
// //                 if check_list(ab_map, oct_id, c, &list, &mut old) {
// //                     return true;
// //                 };
// //             }
// //             _ => (),
// //         }
// //     }
// //     check_list(ab_map, oct_id, 8, &node.nodes, &mut old)
// // }
// // #[cfg(test)]
// // fn check_list(
// //     ab_map: &VecMap<AbNode<f32, usize>>,
// //     parent: usize,
// //     parent_child: usize,
// //     list: &NodeList,
// //     old: &mut VecMap<usize>,
// // ) -> bool {
// //     let mut id = list.head;
// //     let mut prev = 0;
// //     let mut i = 0;
// //     while id > 0 {
// //         old.insert(id, id);
// //         let ab = unsafe { ab_map.get_unchecked(id) };
// //         if ab.prev != prev {
// //             println!("------------0-oct_id: {}, ab_id: {}", parent, id);
// //             return true;
// //         }
// //         if ab.parent != parent {
// //             println!("------------1-oct_id: {}, ab_id: {}", parent, id);
// //             return true;
// //         }
// //         if ab.parent_child != parent_child {
// //             println!("------------2-oct_id: {}, ab_id: {}", parent, id);
// //             return true;
// //         }
// //         if old.contains(ab.next) {
// //             println!("------------3-oct_id: {}, ab_id: {}", parent, id);
// //             return true;
// //         }
// //         prev = id;
// //         id = ab.next;
// //         i += 1;
// //     }
// //     if i != list.len {
// //         println!("------------4-oct_id: {}, ab_id: {}", parent, id);
// //         return true;
// //     }
// //     return false;
// // }

// // #[cfg(test)]
// // fn log(slab: &Slab<OctNode<f32>>, ab_map: &VecMap<AbNode<f32, usize>>, oct_id: usize) {
// //     println!("oct_id----------, id:{}", oct_id);
// //     let mut i = 0;
// //     for or in ab_map.iter() {
// //         i += 1;
// //         if let Some(r) = or {
// //             //let r = ab_map.get(r).unwrap();
// //             println!("ab----------, id:{}, ab: {:?}", i, r);
// //         }
// //     }
// //     for (id, n) in slab.iter() {
// //         //let r = ab_map.get(r).unwrap();
// //         println!("oct=========, id:{}, oct: {:?}", id, n);
// //     }
// // }

// #[test]
// fn test_update() {
//     use pcg_rand::Pcg32;
//     use rand::{Rng, SeedableRng};

//     let max_size = 1000.0;

//     let max = Vector3::new(100f32, 100f32, 100f32);
//     let min = max / 100f32;
//     let mut tree = OctTree::new(
//         AABB::new(
//             Point3::new(0.0, 0.0, 0.0),
//             Point3::new(max_size, max_size, max_size),
//         ),
//         max,
//         min,
//         0,
//         0,
//         10,
//     );

//     let mut rng = pcg_rand::Pcg32::seed_from_u64(1111);
//     //println!("rr = {}", rr);
//     for i in 0..10000 {
//         //println!("i = {}", i);

//         let x = rng.gen_range(0.0, max_size);
//         let y = rng.gen_range(0.0, max_size);
//         let z = rng.gen_range(0.0, max_size);

//         tree.add(
//             i + 1,
//             AABB::new(Point3::new(x, y, z), Point3::new(x, y, z)),
//             i + 1,
//         );

//         tree.collect();

//         let x_: f32 = rng.gen_range(0.0, max_size);
//         let y_: f32 = rng.gen_range(0.0, max_size);
//         let z_: f32 = rng.gen_range(0.0, max_size);

//         // TODO: 改成 7.0 就可以了。
//         let size: f32 = 1.0;
//         let aabb = AABB::new(
//             Point3::new(x_, y_, z_),
//             Point3::new(x_ + size, y_ + size, z_ + size),
//         );

//         tree.update(i + 1, aabb.clone());
//         //tree.remove(i + 1);
//         //tree.add(i + 1, aabb.clone(), i + 1);
//         // if i == 25 {
//         //     let old = clone_tree(&tree.slab, &tree.ab_map);
//         //     assert_eq!(check_tree(&tree.slab, &tree.ab_map, old, i), false);
//         //     log(&tree.slab, &tree.ab_map, i);
//         // }
//         tree.collect();
//         // let aabb = AABB::new(
//         //     Point3::new(aabb.min.x - 1.0, aabb.min.y - 1.0, aabb.min.z - 1.0),
//         //     Point3::new(aabb.min.x + 1.0, aabb.min.y + 1.0, aabb.min.z + 1.0),
//         // );
//         // if i == 25 {
//         //     let old = clone_tree(&tree.slab, &tree.ab_map);
//         //     assert_eq!(check_tree(&tree.slab, &tree.ab_map, old, i), false);
//         //     log(&tree.slab, &tree.ab_map,i);
//         // }
//         let mut args: AbQueryArgs<f32, usize> = AbQueryArgs::new(aabb.clone());
//         tree.query(&aabb, intersects, &mut args, ab_query_func);
//         assert!(args.result.len() > 0);
//     }
// }