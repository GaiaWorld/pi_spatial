//! 四叉相关接口

use std::fmt;
use std::mem;

use nalgebra::*;
use parry2d::{bounding_volume::*, math::Real};
use num_traits::{FromPrimitive, One, Zero, AsPrimitive};
use pi_slotmap::Key;

use crate::*;

/// 四叉树
pub type QuadTree<K, T> = Tree<K, QuadHelper, T, 4>;

#[derive(Debug, Clone)]
pub struct QuadHelper();

impl Helper<4> for QuadHelper {
    type Point = Point2<Real>;
    type Vector = Vector2<Real>;
    type Aabb = Aabb;

    /// 获得AABB的差
    fn aabb_extents(aabb: &Aabb) -> Vector2<Real> {
        aabb.extents()
    }
    /// 移动AABB
    fn aabb_shift(aabb: &Aabb, distance: &Vector2<Real>) -> Aabb {
        Aabb::new(aabb.mins + distance, aabb.maxs + distance)
    }
    /// 判断指定的aabb是否包含另一个aabb
    fn aabb_contains(aabb: &Aabb, other: &Aabb) -> bool {
        aabb.contains(other)
    }
    /// 判断2个aabb是否相交
    fn aabb_intersects(aabb: &Aabb, other: &Aabb) -> bool {
        aabb.intersects(other)
    }
    /// 计算四叉树的深度
    fn get_deap(
        d: &mut Vector2<Real>,
        loose_layer: usize,
        max_loose: &Vector2<Real>,
        deep: usize,
        min_loose: &Vector2<Real>,
    ) -> usize {
        let two = Real::one() + Real::one();
        let x = ComplexField::powf(
            (max_loose.x / d.x + Real::one()) / two,
            FromPrimitive::from_usize(loose_layer).unwrap(),
        );
        let y = ComplexField::powf(
            (max_loose.y / d.y + Real::one()) / two,
            FromPrimitive::from_usize(loose_layer).unwrap(),
        );
        d.x *= x;
        d.y *= y;
        let deep = if loose_layer < deep {
            // 高于该层的节点，松散值都是用最小值， 也可计算其下每层的八叉节点的大小
            // 八叉节点的大小如果小于最小松散值的2倍， 应该停止向下划分， 因为最小松散值占据了八叉节点的大部分
            // 最大层由设置值和该停止划分的层的最小值
            let mut calc_deep = loose_layer;
            let min = min_loose * two;
            while calc_deep < deep && d.x >= min.x && d.y >= min.y {
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
    fn smaller_than_min_loose(d: &Vector2<Real>, min_loose: &Vector2<Real>) -> bool {
        if d.x <= min_loose.x && d.y <= min_loose.y {
            return true;
        };
        return false;
    }

    #[inline]
    /// 指定向量以及最大松散尺寸计算对应的层
    fn calc_layer(loose: &Vector2<Real>, el: &Vector2<Real>) -> usize {
        let x = if el.x == Real::zero() {
            usize::max_value()
        } else {
            (loose.x / el.x).as_()
        };
        let y = if el.y == Real::zero() {
            usize::max_value()
        } else {
            (loose.y / el.y).as_()
        };

        let min = x.min(y);
        if min == 0 {
            return 0;
        }
        (mem::size_of::<usize>() << 3) - (min.leading_zeros() as usize) - 1
    }

    #[inline]
    /// 判断所在的子节点
    fn get_child(point: &Point2<Real>, aabb: &Aabb) -> usize {
        let mut i: usize = 0;
        if aabb.maxs.x > point.x {
            i += 1;
        }
        if aabb.maxs.y > point.y {
            i += 2;
        }
        i
    }

    #[inline]
    fn get_max_half_loose(aabb: &Aabb, loose: &Vector2<Real>) -> Point2<Real> {
        let two = Real::one() + Real::one();
        let x = (aabb.mins.x + aabb.maxs.x + loose.x) / two;
        let y = (aabb.mins.y + aabb.maxs.y + loose.y) / two;
        Point2::new(x, y)
    }

    /// 创建ab的子节点集合
    fn make_childs(aabb: &Aabb, loose: &Vector2<Real>) -> [Aabb; 4] {
        let two = Real::one() + Real::one();
        let x = (aabb.mins.x + aabb.maxs.x - loose.x) / two;
        let y = (aabb.mins.y + aabb.maxs.y - loose.y) / two;
        let p1 = Point2::new(x, y);
        let p2 = Self::get_max_half_loose(&aabb, &loose);
	[
            Aabb::new(aabb.mins, p2),
            Aabb::new(
                Point2::new(p1.x, aabb.mins.y),
                Point2::new(aabb.maxs.x, p2.y),
            ),
            Aabb::new(
                Point2::new(aabb.mins.x, p1.y),
                Point2::new(p2.x, aabb.maxs.y),
            ),
            Aabb::new(p1, aabb.maxs),
        ]
    }

    /// 指定创建ab的子节点
    fn create_child(
        aabb: &Aabb,
        loose: &Vector2<Real>,
        layer: usize,
        loose_layer: usize,
        min_loose: &Vector2<Real>,
        index: usize,
    ) -> (Aabb, Vector2<Real>) {
        let two = Real::one() + Real::one();
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
            0 => Aabb::new(aabb.mins, Point2::new(c2!(x), c2!(y))),
            1 => Aabb::new(
                Point2::new(c1!(x), aabb.mins.y),
                Point2::new(aabb.maxs.x, c2!(y)),
            ),
            2 => Aabb::new(
                Point2::new(aabb.mins.x, c1!(y)),
                Point2::new(c2!(x), aabb.maxs.y),
            ),
            _ => Aabb::new(Point2::new(c1!(x), c1!(y)), aabb.maxs),
        };
        let loose = if layer < loose_layer {
            loose / two
        } else {
            min_loose.clone()
        };
        (a, loose)
    }
}


/// quad节点查询函数的范本，aabb是否相交，参数a是查询参数，参数b是quad节点的aabb， 所以最常用的判断是左闭右开
/// 应用方为了功能和性能，应该实现自己需要的quad节点的查询函数， 比如点查询， 球查询， 视锥体查询...
#[inline]
pub fn intersects(a: &Aabb, b: &Aabb) -> bool {
    a.mins.x <= b.maxs.x
        && a.maxs.x > b.mins.x
        && a.mins.y <= b.maxs.y
        && a.maxs.y > b.mins.y
}

/// aabb的查询函数的参数
pub struct AbQueryArgs<K: Key, T: Clone + PartialOrd> {
    pub aabb: Aabb,
    pub result: (K, T),
}
impl<K: Key, T: Clone + PartialOrd> AbQueryArgs<K, T> {
    pub fn new(aabb: Aabb, min: T) -> AbQueryArgs<K, T> {
        AbQueryArgs {
            aabb: aabb,
            result: (K::null(), min),
        }
    }
}

/// ab节点的查询函数, 这里只是一个简单范本，使用了quad节点的查询函数intersects
/// 应用方为了功能和性能，应该实现自己需要的ab节点的查询函数， 比如点查询， 球查询-包含或相交， 视锥体查询...
pub fn ab_query_func<K: Key, T: Clone + PartialOrd + fmt::Debug>(
    arg: &mut AbQueryArgs<K, T>,
    id: K,
    aabb: &Aabb,
    bind: &T,
) {
    // println!("ab_query_func: id: {}, bind:{:?}, arg: {:?}", id, bind, arg.result);
    if intersects(&arg.aabb, aabb) {
        if bind > &arg.result.1 {
            arg.result.0 = id;
            arg.result.1 = bind.clone();
        }
    }
}

#[test]
fn test1() {
	use pi_slotmap::{SlotMap, DefaultKey};

    println!("test1-----------------------------------------");
    let max = Vector2::new(100f32, 100f32);
    let min = max / 100f32;
    let mut tree = QuadTree::new(
        Aabb::new(
            Point2::new(-1024f32, -1024f32),
            Point2::new(3072f32, 3072f32),
        ),
        max,
        min,
        0,
        0,
        0,
    );
	let mut slot_map = SlotMap::new();
	let mut keys = Vec::new();
	keys.push(DefaultKey::null());
    for i in 0..1 {
		keys.push(slot_map.insert(()));
        tree.add(
            keys.last().unwrap().clone(),
            Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
            i + 1,
        );
    }
    for i in 1..tree.ab_map.len() + 1 {
        println!("00000, id:{}, ab: {:?}", i, tree.ab_map.get(keys[i]).unwrap());
    }
    tree.update(
        keys[1],
        Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
    );
    for i in 1..tree.ab_map.len() + 1 {
        println!("00000, id:{}, ab: {:?}", i, tree.ab_map.get(keys[i]).unwrap());
    }
    tree.collect();
    for i in 1..tree.ab_map.len() + 1 {
        println!("00000, id:{}, ab: {:?}", i, tree.ab_map.get(keys[i]).unwrap());
    }
    for i in 1..6 {
		keys.push(slot_map.insert(()));
        tree.add(
            keys.last().unwrap().clone(),
            Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
            i + 3,
        );
    }
    for i in 1..tree.ab_map.len() + 1 {
        println!("00001, id:{}, ab: {:?}", i, tree.ab_map.get(keys[i]).unwrap());
    }
    tree.update(
        keys[2],
        Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
    );
    tree.update(
        keys[3],
        Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
    );

    tree.update(
        keys[4],
        Aabb::new(
            Point2::new(0.0, 700.0),
            Point2::new(1000.0, 1400.0),
        ),
    );

    tree.update(
        keys[5],
        Aabb::new(
            Point2::new(0.0, 1400.0),
            Point2::new(1000.0, 1470.0),
        ),
    );
    tree.update(
        keys[6],
        Aabb::new(
            Point2::new(0.0, 1470.0),
            Point2::new(1000.0, 1540.0),
        ),
    );
    tree.update(
        keys[1],
        Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
    );
    tree.collect();
    for i in 1..tree.ab_map.len() + 1 {
        println!("00002, id:{}, ab: {:?}", i, tree.ab_map.get(keys[i]).unwrap());
    }
    //   tree.update(1, Aabb::new(Point2::new(0.0,0.0,0.0), Point2::new(1000.0, 800.0, 1.0)));
    //   tree.update(2, Aabb::new(Point2::new(0.0,0.0,0.0), Point2::new(1000.0, 800.0, 1.0)));
    //   tree.update(3, Aabb::new(Point2::new(0.0,0.0,0.0), Point2::new(1000.0, 800.0, 1.0)));
    //   tree.update(4, Aabb::new(Point2::new(0.0,0.0,0.0), Point2::new(1000.0, 800.0, 1.0)));

    //   tree.update(5, Aabb::new(Point2::new(0.0,800.0,0.0), Point2::new(1000.0, 1600.0, 1.0)));

    //    tree.update(6, Aabb::new(Point2::new(0.0,1600.0,0.0), Point2::new(1000.0, 2400.0, 1.0)));
    //   tree.update(7, Aabb::new(Point2::new(0.0,2400.0,0.0), Point2::new(1000.0, 3200.0, 1.0)));
    //   for i in 1..tree.ab_map.len() + 1 {
    //   println!("22222, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
    //  }
    // tree.collect();
    for i in 1..tree.slab.len() + 1 {
        println!(
            "000000 000000, id:{}, quad: {:?}",
            i,
            tree.slab.get(keys[i]).unwrap()
        );
    }
    println!("outer:{:?}", tree.outer);
    let aabb = Aabb::new(
        Point2::new(500f32, 500f32),
        Point2::new(500f32, 500f32),
    );
    let mut args: AbQueryArgs<DefaultKey, usize> = AbQueryArgs::new(aabb.clone(), 0);
    tree.query(&aabb, intersects, &mut args, ab_query_func);
    //assert_eq!(args.result(), [1, 3, 4]);
}

// #[test]
// fn test2() {
//     println!("test2-----------------------------------------");
//     let max = Vector2::new(100f32, 100f32);
//     let min = max / 100f32;
//     let mut tree = QuadTree::new(
//         Aabb::new(
//             Point2::new(-1024f32, -1024f32),
//             Point2::new(3072f32, 3072f32),
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
//             Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//             i + 1,
//         );
//     }
//     for i in 1..tree.slab.len() + 1 {
//         println!("000000, id:{}, quad: {:?}", i, tree.slab.get(i).unwrap());
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("00000, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     tree.update(
//         1,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(0.0, 0.0)),
//     );
//     tree.collect();
//     for i in 1..tree.slab.len() + 1 {
//         println!(
//             "000000 000000, id:{}, quad: {:?}",
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
//     let max = Vector2::new(100f32, 100f32);
//     let min = max / 100f32;
//     let mut tree = QuadTree::new(
//         Aabb::new(
//             Point2::new(-1024f32, -1024f32),
//             Point2::new(3072f32, 3072f32),
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
//             Aabb::new(Point2::new(0.0, 0.0), Point2::new(0.1, 0.1)),
//             i + 1,
//         );
//     }
//     for i in 1..tree.slab.len() + 1 {
//         println!("test1, id:{}, quad: {:?}", i, tree.slab.get(i).unwrap());
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("test1, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     tree.collect();
//     for i in 1..tree.slab.len() + 1 {
//         println!("test2, id:{}, quad: {:?}", i, tree.slab.get(i).unwrap());
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("test2, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     tree.shift(4, Vector2::new(2.0, 2.0));
//     tree.shift(5, Vector2::new(4.0, 4.0));
//     tree.shift(6, Vector2::new(10.0, 10.0));
//     for i in 1..tree.slab.len() + 1 {
//         println!("test3, id:{}, quad: {:?}", i, tree.slab.get(i).unwrap());
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("test3, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     tree.collect();
//     for i in 1..tree.slab.len() + 1 {
//         println!("test4, id:{}, quad: {:?}", i, tree.slab.get(i).unwrap());
//     }
//     for i in 1..tree.ab_map.len() + 1 {
//         println!("test4, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
//     }
//     println!("outer:{:?}", tree.outer);
//     let aabb = Aabb::new(
//         Point2::new(0.05f32, 0.05f32),
//         Point2::new(0.05f32, 0.05f32),
//     );
//     let mut args: AbQueryArgs<f32, usize> = AbQueryArgReal::new(aabb.clone(), 0);
//     tree.query(&aabb, intersects, &mut args, ab_query_func);
//     assert_eq!(args.result.1, 3);
// }

// #[test]
// fn test3() {
//     let aabb = Aabb::new(
//         Point2::new(700.0, 100.0),
//         Point2::new(700.0, 100.0),
//     );
//     let mut args: AbQueryArgs<f32, usize> = AbQueryArgReal::new(aabb.clone(), 0);

//     // let mut tree = Tree::new(Aabb::new(Point2::new(0f32,0f32,0f32), Point2::new(1000f32,1000f32,1000f32)),
//     // 	0,
//     // 	0,
//     // 	0,
//     // 	0,
//     // );
//     let max = Vector2::new(100f32, 100f32);
//     let min = max / 100f32;
//     let mut tree = QuadTree::new(
//         Aabb::new(
//             Point2::new(-1024f32, -1024f32),
//             Point2::new(3072f32, 3072f32),
//         ),
//         max,
//         min,
//         0,
//         0,
//         0,
//     );
//     tree.add(
//         1,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         2,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.collect();

//     tree.update(
//         0,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         1,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         2,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );

//     tree.add(
//         3,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         4,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         5,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         6,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         7,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         8,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         9,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.collect();
//     tree.update(
//         3,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 350.0)),
//     );
//     tree.update(
//         4,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(800.0, 175.0)),
//     );
//     tree.update(
//         5,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(640.0, 140.0)),
//     );
//     tree.update(
//         6,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(512.0, 112.0)),
//     );
//     tree.update(
//         7,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(410.0, 90.0)),
//     );
//     tree.update(
//         8,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1600.0, 175.0),
//         ),
//     );
//     tree.update(
//         9,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1440.0, 140.0),
//         ),
//     );
//     tree.update(
//         1,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         2,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.query(&aabb, intersects, &mut args, ab_query_func);
//     tree.remove(7);
//     tree.remove(6);
//     tree.remove(5);

//     tree.add(
//         5,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         6,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         7,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.collect();
//     tree.update(
//         5,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(640.0, 140.0)),
//     );
//     tree.update(
//         6,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(512.0, 112.0)),
//     );
//     tree.update(
//         7,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(410.0, 90.0)),
//     );
//     tree.update(
//         1,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         2,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         3,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 350.0)),
//     );
//     tree.update(
//         4,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(800.0, 175.0)),
//     );
//     tree.update(
//         8,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1600.0, 175.0),
//         ),
//     );
//     tree.update(
//         9,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1440.0, 140.0),
//         ),
//     );
//     for i in 1..tree.slab.len() + 1 {
//         println!("test||||||, id:{}, quad: {:?}", i, tree.slab.get(i));
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
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         6,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         7,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.collect();

//     tree.update(
//         5,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(640.0, 140.0)),
//     );
//     tree.update(
//         6,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(512.0, 112.0)),
//     );
//     tree.update(
//         7,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(410.0, 90.0)),
//     );
//     tree.update(
//         1,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         2,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         3,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 350.0)),
//     );
//     tree.update(
//         4,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(800.0, 175.0)),
//     );
//     tree.update(
//         8,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1600.0, 175.0),
//         ),
//     );
//     tree.update(
//         9,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1440.0, 140.0),
//         ),
//     );
//     tree.query(&aabb, intersects, &mut args, ab_query_func);
//     for i in 1..tree.slab.len() + 1 {
//         println!("test||||||, id:{}, quad: {:?}", i, tree.slab.get(i));
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
//     let aabb = Aabb::new(
//         Point2::new(700.0, 100.0),
//         Point2::new(700.0, 100.0),
//     );
//     let mut args: AbQueryArgs<f32, usize> = AbQueryArgReal::new(aabb.clone(), 0);

//     let max = Vector2::new(100f32, 100f32);
//     let min = max / 100f32;
//     let mut tree = QuadTree::new(
//         Aabb::new(
//             Point2::new(-1024f32, -1024f32),
//             Point2::new(3072f32, 3072f32),
//         ),
//         max,
//         min,
//         0,
//         0,
//         0,
//     );
//     tree.add(
//         1,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         2,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.collect();

//     tree.update(
//         0,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         1,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         2,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );

//     tree.add(
//         3,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         4,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         5,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         6,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         7,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         8,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         9,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.collect();
//     tree.update(
//         3,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 350.0)),
//     );
//     tree.update(
//         4,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(800.0, 175.0)),
//     );
//     tree.update(
//         5,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(640.0, 140.0)),
//     );
//     tree.update(
//         6,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(512.0, 112.0)),
//     );
//     tree.update(
//         7,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(410.0, 90.0)),
//     );
//     tree.update(
//         8,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1600.0, 175.0),
//         ),
//     );
//     tree.update(
//         9,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1440.0, 140.0),
//         ),
//     );
//     tree.update(
//         1,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         2,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );

//     tree.add(
//         10,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         11,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         12,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.collect();
//     tree.update(
//         10,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(640.0, 140.0)),
//     );
//     tree.update(
//         11,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(512.0, 112.0)),
//     );
//     tree.update(
//         12,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(410.0, 90.0)),
//     );
//     tree.update(
//         1,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         2,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         3,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 350.0)),
//     );
//     tree.update(
//         4,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(800.0, 175.0)),
//     );
//     tree.update(
//         8,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1600.0, 175.0),
//         ),
//     );
//     tree.update(
//         9,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1440.0, 140.0),
//         ),
//     );

//     tree.add(
//         13,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         14,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.add(
//         15,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
//         1,
//     );
//     tree.collect();

//     //log(&tree.slab, &tree.ab_map, 10000);
//     tree.update(
//         13,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(640.0, 140.0)),
//     );
//     //log(&tree.slab, &tree.ab_map, 13);
//     println!("quad========================");
//     tree.update(
//         14,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(512.0, 112.0)),
//     );
//     tree.update(
//         15,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(410.0, 90.0)),
//     );
//     tree.update(
//         1,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         2,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
//     );
//     tree.update(
//         3,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 350.0)),
//     );
//     tree.update(
//         4,
//         Aabb::new(Point2::new(0.0, 0.0), Point2::new(800.0, 175.0)),
//     );
//     tree.update(
//         8,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1600.0, 175.0),
//         ),
//     );
//     tree.update(
//         9,
//         Aabb::new(
//             Point2::new(800.0, 0.0),
//             Point2::new(1440.0, 140.0),
//         ),
//     );
//     let mut rng = rand::thread_rng();
//     for _ in 0..1000000 {
//         let r = rand::thread_rng().gen_range(1, 16);
//         let x = rand::thread_rng().gen_range(0, 128) - 64;
//         let y = rand::thread_rng().gen_range(0, 128) - 64;
//         let old = (tree.slab.clone(), tree.ab_map.clone());
//         tree.shift(r, Vector2::new(x as f32, y as f32));
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
// //     slab: &Slab<quadNode<f32>>,
// //     ab_map: &VecMap<AbNode<f32, usize>>,
// //     old: (Slab<quadNode<f32>>, VecMap<AbNode<f32, usize>>),
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
// //     slab: &Slab<quadNode<f32>>,
// //     ab_map: &VecMap<AbNode<f32, usize>>,
// //     quad_id: usize,
// // ) -> bool {
// //     let node = unsafe { slab.get_unchecked(quad_id) };
// //     let mut old = VecMap::default();
// //     for c in 0..8 {
// //         match &node.childs[c] {
// //             ChildNode::Ab(list) => {
// //                 if check_list(ab_map, quad_id, c, &list, &mut old) {
// //                     return true;
// //                 };
// //             }
// //             _ => (),
// //         }
// //     }
// //     check_list(ab_map, quad_id, 8, &node.nodes, &mut old)
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
// //             println!("------------0-quad_id: {}, ab_id: {}", parent, id);
// //             return true;
// //         }
// //         if ab.parent != parent {
// //             println!("------------1-quad_id: {}, ab_id: {}", parent, id);
// //             return true;
// //         }
// //         if ab.parent_child != parent_child {
// //             println!("------------2-quad_id: {}, ab_id: {}", parent, id);
// //             return true;
// //         }
// //         if old.contains(ab.next) {
// //             println!("------------3-quad_id: {}, ab_id: {}", parent, id);
// //             return true;
// //         }
// //         prev = id;
// //         id = ab.next;
// //         i += 1;
// //     }
// //     if i != list.len {
// //         println!("------------4-quad_id: {}, ab_id: {}", parent, id);
// //         return true;
// //     }
// //     return false;
// // }

// // #[cfg(test)]
// // fn log(slab: &Slab<quadNode<f32>>, ab_map: &VecMap<AbNode<f32, usize>>, quad_id: usize) {
// //     println!("quad_id----------, id:{}", quad_id);
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
// //         println!("quad=========, id:{}, quad: {:?}", id, n);
// //     }
// // }

// #[test]
// fn test_update() {
//     use pcg_rand::Pcg32;
//     use rand::{Rng, SeedableRng};

//     let max_size = 1000.0;

//     let max = Vector2::new(100f32, 100f32);
//     let min = max / 100f32;
//     let mut tree = QuadTree::new(
//         Aabb::new(
//             Point2::new(0.0, 0.0),
//             Point2::new(max_size, max_size),
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
//             Aabb::new(Point2::new(x, y), Point2::new(x, y)),
//             i + 1,
//         );

//         tree.collect();

//         let x_: f32 = rng.gen_range(0.0, max_size);
//         let y_: f32 = rng.gen_range(0.0, max_size);

//         // TODO: 改成 7.0 就可以了。
//         let size: f32 = 1.0;
//         let aabb = Aabb::new(
//             Point2::new(x_, y_),
//             Point2::new(x_ + size, y_ + size),
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
//         // let aabb = Aabb::new(
//         //     Point2::new(aabb.min.x - 1.0, aabb.min.y - 1.0, aabb.min.z - 1.0),
//         //     Point2::new(aabb.min.x + 1.0, aabb.min.y + 1.0, aabb.min.z + 1.0),
//         // );
//         // if i == 25 {
//         //     let old = clone_tree(&tree.slab, &tree.ab_map);
//         //     assert_eq!(check_tree(&tree.slab, &tree.ab_map, old, i), false);
//         //     log(&tree.slab, &tree.ab_map,i);
//         // }
//         let mut args: AbQueryArgs<f32, usize> = AbQueryArgReal::new(aabb.clone(), 0);
//         tree.query(&aabb, intersects, &mut args, ab_query_func);
//         assert!(args.result.0 > 0);
//     }
// }