//! 瓦片地图，可在瓦片内放多个id的AABB。
//! 要求插入AABB节点时的id， 应该是slotmap的Key。
//! 内部使用SecondaryMap来存储链表，这样内存连续，瓦片地图本身就可以快速拷贝。
//! 通过AABB的中心点计算落在哪个瓦片内，可以查询该瓦片内所有的节点。
//! AABB的范围相交查询时，需要根据最大节点的大小，扩大相应范围，这样如果边界上有节点，也可以被查到相交。

use nalgebra::*;
use num_traits::cast::AsPrimitive;
use parry2d::bounding_volume::*;
use parry2d::math::Real;
use pi_link_list::{Iter, LinkList, Node};
use pi_null::*;
use pi_slotmap::*;

type List<K, T> = LinkList<K, T, SecondaryMap<K, Node<K, T>>>;

pub struct MapInfo {
    // 场景的范围
    pub bounds: Aabb,
    // 该图宽度
    pub width: usize,
    // 该图高度
    pub height: usize,
    // 瓦片总数量
    pub amount: usize,
    // 大小
    size: Vector2<Real>,
}
impl MapInfo {
    /// 计算指定位置的瓦片坐标
    pub fn calc_tile_index(&self, loc: Point2<Real>) -> (usize, usize) {
        let x = if loc[0] <= self.bounds.mins[0] {
            0
        } else if loc[0] >= self.bounds.maxs[0] {
            self.width - 1
        } else {
            ((loc[0] - self.bounds.mins[0]) * self.width as Real / self.size[0]).as_()
        };
        let y = if loc[1] <= self.bounds.mins[1] {
            0
        } else if loc[1] >= self.bounds.maxs[1] {
            self.height - 1
        } else {
            ((loc[1] - self.bounds.mins[1]) * self.height as Real / self.size[1]).as_()
        };
        (x, y)
    }
    /// 获得指定坐标瓦片的tile_index
    pub fn tile_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
    /// 获得指定位置瓦片的坐标
    pub fn tile_xy(&self, tile_index: usize) -> (usize, usize) {
        (tile_index % self.width, tile_index / self.width)
    }
}

///
/// 松散地图结构体
///
/// ### 对`N`的约束
///
/// + 浮点数算术运算，可拷贝，可偏序比较；
/// + 实际使用的时候就是浮点数字类型，比如：f32/f64；
///
pub struct TileMap<K: Key, T> {
    //所有存储aabb的节点
    ab_map: SecondaryMap<K, Node<K, (Aabb, T)>>,
    // 该图所有瓦片
    tiles: Vec<List<K, (Aabb, T)>>,
    // 场景的范围
    pub info: MapInfo,
    // 节点的最大半径
    pub node_max_half_size: Vector2<Real>,
}

impl<K: Key, T> TileMap<K, T> {
    ///
    /// 新建一个瓦片图
    ///
    /// 需传入根节点（即全场景），指定瓦片图的宽度和高度
    pub fn new(bounds: Aabb, width: usize, height: usize) -> Self {
        let amount = width * height;
        let mut tiles = Vec::with_capacity(amount);
        tiles.resize_with(amount, Default::default);
        let size = bounds.extents();
        let info = MapInfo {
            bounds,
            height,
            width,
            amount,
            size,
        };
        TileMap {
            ab_map: Default::default(),
            tiles,
            info,
            node_max_half_size: Vector2::zeros(),
        }
    }
    /// 获得节点最大半径
    pub fn get_node_max_half_size(&self) -> &Vector2<Real> {
        &self.node_max_half_size
    }
    /// 设置节点最大半径
    pub fn set_node_max_half_size(&mut self, half_size: Vector2<Real>) {
        self.node_max_half_size = half_size;
    }
    /// 更新节点最大半径
    fn update_node_max_half_size(&mut self, aabb: Aabb) {
        let size = aabb.half_extents();
        if size.x > self.node_max_half_size.x {
            self.node_max_half_size.x = size.x;
        }
        if size.y > self.node_max_half_size.y {
            self.node_max_half_size.y = size.y;
        }
    }
    /// 获得指定位置的瓦片，超出地图边界则返回最近的边界瓦片
    pub fn get_tile_index(&self, loc: Point2<Real>) -> usize {
        let (x, y) = self.info.calc_tile_index(loc);
        self.info.tile_index(x, y)
    }
    /// 获得指定位置瓦片的节点数量和节点迭代器
    pub fn get_tile_iter<'a>(
        &'a self,
        tile_index: usize,
    ) -> (
        usize,
        Iter<'a, K, (Aabb, T), SecondaryMap<K, Node<K, (Aabb, T)>>>,
    ) {
        let list = &self.tiles[tile_index];
        (list.len(), list.iter(&self.ab_map))
    }
    /// 获得指定范围的tile数量和迭代器
    pub fn query_iter(&self, aabb: &Aabb) -> (usize, QueryIter) {
        // 获得min所在瓦片
        let (x_start, y_start) = self
            .info
            .calc_tile_index(aabb.mins - self.node_max_half_size);
        // 获得max所在瓦片
        let (x_end, y_end) = self
            .info
            .calc_tile_index(aabb.maxs + self.node_max_half_size);
        (
            (x_end - x_start + 1) * (y_end - y_start + 1),
            QueryIter {
                width: self.info.width,
                x_start,
                x_end,
                y_start,
                y_end,
                cur_x: x_start,
            },
        )
    }
    /// 查询空间内及相交的ab节点
    pub fn query<A>(
        &self,
        aabb: &Aabb,
        arg: &mut A,
        ab_func: fn(arg: &mut A, id: K, aabb: &Aabb, bind: &T),
    ) {
        let (_, tile_it) = self.query_iter(aabb);
        for tile_index in tile_it {
            let (_, it) = self.get_tile_iter(tile_index);
            for (id, node) in it {
                ab_func(arg, id, &node.0, &node.1);
            }
        }
    }

    /// 指定id，在地图中添加一个aabb单元及其绑定
    pub fn add(&mut self, id: K, aabb: Aabb, bind: T) -> bool {
        let center = aabb.center();
        // 获得所在瓦片
        let tile_index = self.get_tile_index(center);
        // // 不在网格范围内
        // if tile_index.is_null() {
        //     return false;
        // }
        match self.ab_map.insert(id, Node::new((aabb, bind))) {
            Some(_) => return false,
            None => (),
        }
        self.update_node_max_half_size(aabb);
        self.tiles[tile_index].link_before(id, K::null(), &mut self.ab_map);
        true
    }
    /// 获取所有id的aabb及其绑定的迭代器
    pub fn iter(&self) -> pi_slotmap::secondary::Iter<K, Node<K, (Aabb, T)>> {
        self.ab_map.iter()
    }
    /// 获取指定id的aabb及其绑定
    pub fn get(&self, id: K) -> Option<&(Aabb, T)> {
        match self.ab_map.get(id) {
            Some(node) => Some(&node),
            None => None,
        }
    }

    /// 获取指定id的aabb及其绑定
    pub unsafe fn get_unchecked(&self, id: K) -> &(Aabb, T) {
        &self.ab_map.get_unchecked(id)
    }

    /// 获取指定id的可写绑定
    pub unsafe fn get_mut(&mut self, id: K) -> Option<&mut T> {
        match self.ab_map.get_mut(id) {
            Some(n) => Some(&mut n.1),
            None => None,
        }
    }

    /// 获取指定id的可写绑定
    pub unsafe fn get_unchecked_mut(&mut self, id: K) -> &mut T {
        &mut self.ab_map.get_unchecked_mut(id).1
    }

    /// 检查是否包含某个key
    pub fn contains_key(&self, id: K) -> bool {
        self.ab_map.contains_key(id)
    }

    /// 更新指定id的aabb
    pub fn update(&mut self, id: K, aabb: Aabb) -> bool {
        let node = match self.ab_map.get_mut(id) {
            Some(n) => n,
            _ => return false,
        };
        // 获得所在瓦片的位置
        let (new_x, new_y) = self.info.calc_tile_index(aabb.center());
        // 获得原来所在瓦片的位置
        let (x, y) = self.info.calc_tile_index(node.0.center());
        node.0 = aabb;
        self.move_from_to(id, x, y, new_x, new_y);
        self.update_node_max_half_size(aabb);
        true
    }

    /// 移动指定id的相对位置
    pub fn shift(&mut self, id: K, distance: Vector2<Real>) -> bool {
        let node = match self.ab_map.get_mut(id) {
            Some(n) => n,
            _ => return false,
        };
        // 新aabb
        let aabb = Aabb::new(node.0.mins + distance, node.0.maxs + distance);
        // 获得新的所在瓦片
        let (new_x, new_y) = self.info.calc_tile_index(aabb.center());
        // 获得原来所在瓦片
        let (x, y) = self.info.calc_tile_index(node.0.center());
        node.0 = aabb;
        self.move_from_to(id, x, y, new_x, new_y);
        true
    }
    /// 移动指定id的绝对位置
    pub fn move_to(&mut self, id: K, loc: Point2<Real>) -> bool {
        let node = match self.ab_map.get_mut(id) {
            Some(n) => n,
            _ => return false,
        };
        // 获得新的所在瓦片
        let (new_x, new_y) = self.info.calc_tile_index(loc);
        let center = node.0.center();
        // 获得原来所在瓦片
        let (x, y) = self.info.calc_tile_index(center);
        let d = loc - center;
        node.0 = Aabb::new(node.0.mins + d, node.0.maxs + d);
        self.move_from_to(id, x, y, new_x, new_y);
        true
    }
    fn move_from_to(&mut self, id: K, x: usize, y: usize, new_x: usize, new_y: usize) {
        if x == new_x && y == new_y {
            return;
        }
        let new_tile_index = self.info.tile_index(new_x, new_y);
        let tile_index = self.info.tile_index(x, y);
        self.tiles[tile_index].unlink(id, &mut self.ab_map);
        self.tiles[new_tile_index].link_before(id, K::null(), &mut self.ab_map);
    }
    /// 更新指定id的绑定
    pub fn update_bind(&mut self, id: K, bind: T) -> bool {
        match self.ab_map.get_mut(id) {
            Some(node) => {
                node.1 = bind;
                true
            }
            _ => false,
        }
    }
    /// 移除指定id的aabb及其绑定
    pub fn remove(&mut self, id: K) -> Option<(Aabb, T)> {
        let node = match self.ab_map.get(id) {
            Some(n) => n,
            _ => return None,
        };
        let tile_index = self.get_tile_index(node.0.center());
        self.tiles[tile_index].unlink(id, &mut self.ab_map);
        self.ab_map.remove(id).map(|n| n.take())
    }
    /// 获得指定id的所在的tile
    pub fn get_tile_index_by_id(&self, id: K) -> usize {
        let node = match self.ab_map.get(id) {
            Some(n) => n,
            _ => return Null::null(),
        };
        // 获得新的所在瓦片
        let (x, y) = self.info.calc_tile_index(node.0.center());
        self.info.tile_index(x, y)
    }
    /// 获得节点数量
    pub fn len(&self) -> usize {
        self.ab_map.len()
    }
}

#[derive(Debug, Clone, Default)]
pub struct QueryIter {
    width: usize,
    x_start: usize,
    x_end: usize,
    y_start: usize,
    y_end: usize,
    cur_x: usize,
}

impl Iterator for QueryIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.y_start > self.y_end {
            return None;
        }
        let index = self.y_start * self.width + self.cur_x;
        if self.cur_x < self.x_end {
            self.cur_x += 1;
        } else {
            self.cur_x = self.x_start;
            self.y_start += 1;
        }
        Some(index)
    }
}

#[test]
fn test1() {
    use pi_slotmap::{DefaultKey, SlotMap};

    println!("test1-----------------------------------------");
    let mut tree = TileMap::new(
        Aabb::new(
            Point2::new(-1024f32, -1024f32),
            Point2::new(3072f32, 3072f32),
        ),
        10,
        10,
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
        println!(
            "10000, id:{}, ab: {:?}, index: {:?}",
            i,
            tree.ab_map.get(keys[i]).unwrap(),
            tree.get_tile_index_by_id(keys[i])
        );
    }
    tree.update(
        keys[1],
        Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
    );
    for i in 1..tree.ab_map.len() + 1 {
        println!(
            "20000, id:{}, ab: {:?}, index: {:?}",
            i,
            tree.ab_map.get(keys[i]).unwrap(),
            tree.get_tile_index_by_id(keys[i])
        );
    }

    for i in 1..tree.ab_map.len() + 1 {
        println!(
            "30000, id:{}, ab: {:?}, index: {:?}",
            i,
            tree.ab_map.get(keys[i]).unwrap(),
            tree.get_tile_index_by_id(keys[i])
        );
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
        let index = tree.get_tile_index_by_id(keys[i]);
        let (len, mut iter) = tree.get_tile_iter(index);
        println!(
            "00001, id:{}, ab: {:?}, index: {:?}, iter:{:?}",
            i,
            tree.ab_map.get(keys[i]).unwrap(),
            index,
            (len, iter.next())
        );
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
        Aabb::new(Point2::new(0.0, 700.0), Point2::new(1000.0, 1400.0)),
    );

    tree.update(
        keys[5],
        Aabb::new(Point2::new(0.0, 1400.0), Point2::new(1000.0, 1470.0)),
    );
    tree.update(
        keys[6],
        Aabb::new(Point2::new(0.0, 1470.0), Point2::new(1000.0, 1540.0)),
    );
    tree.update(
        keys[1],
        Aabb::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
    );

    for i in 1..tree.ab_map.len() + 1 {
        println!(
            "00002, id:{}, ab: {:?}, index: {:?}",
            i,
            tree.ab_map.get(keys[i]).unwrap(),
            tree.get_tile_index_by_id(keys[i])
        );
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
    let aabb = Aabb::new(Point2::new(500f32, 500f32), Point2::new(1100f32, 1100f32));
    let (len, iter) = tree.query_iter(&aabb);
    println!("query_iter count:{},", len);
    for i in iter {
        println!(
            "id:{}, xy: {:?}",
            i,
            tree.info.tile_xy(i),
            //get_4d_neighbors(i, tree.info.column, tree.info.count),
            //get_8d_neighbors(i, tree.info.column, tree.info.count)
        );
    }
    //assert_eq!(args.result(), [1, 3, 4]);
}
