//! 瓦片地图，可在瓦片内放多个id的AABB。
//! 要求插入AABB节点时的id， 应该是slotmap的Key。
//! 内部使用SecondaryMap来存储链表，这样内存连续，瓦片地图本身就可以快速拷贝。
//! 通过AABB的中心点计算落在哪个瓦片内，可以查询该瓦片内所有的节点。
//! AABB的范围相交查询时，需要根据最大节点的大小，扩大相应范围，这样如果边界上有节点，也可以被查到相交。

use nalgebra::*;
use ncollide2d::bounding_volume::*;
use num_traits::cast::AsPrimitive;
use num_traits::*;
use pi_null::*;
use pi_slotmap::*;

// 八方向枚举
#[repr(C)]
#[derive(Debug, Clone)]
pub enum Direction {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
    UpLeft = 4,
    UpRight = 5,
    DownLeft = 6,
    DownRight = 7,
}
/// 获得指定位置瓦片的上下左右四个瓦片， 如果为数组元素为null，则超出边界
pub fn get_4d_neighbors(tile_index: usize, column: usize, count: usize) -> [usize; 4] {
    let mut arr = Default::default();
    if tile_index >= count + column {
        return arr;
    }
    let c = tile_index % column;
    if c == column - 1 {
        arr[Direction::Left as usize] = tile_index - 1;
    } else if c > 0 {
        arr[Direction::Left as usize] = tile_index - 1;
        arr[Direction::Right as usize] = tile_index + 1;
    } else {
        arr[Direction::Right as usize] = tile_index + 1;
    }
    if tile_index + column >= count {
        arr[Direction::Up as usize] = tile_index - column;
    } else if tile_index >= column {
        arr[Direction::Up as usize] = tile_index - column;
        arr[Direction::Down as usize] = tile_index + column;
    } else {
        arr[Direction::Down as usize] = tile_index + column;
    }
    arr
}
/// 获得指定位置瓦片周围的八个瓦片， 如果为数组元素为null，则超出边界
pub fn get_8d_neighbors(tile_index: usize, column: usize, count: usize) -> [usize; 8] {
    let mut arr = Default::default();
    if tile_index >= count + column {
        return arr;
    }
    let c = tile_index % column;
    if tile_index >= count {
        arr[Direction::Up as usize] = tile_index - column;
        if c == column - 1 {
            arr[Direction::UpLeft as usize] = arr[Direction::Up as usize] - 1;
        } else if c > 0 {
            arr[Direction::UpLeft as usize] = arr[Direction::Up as usize] - 1;
            arr[Direction::UpRight as usize] = arr[Direction::Up as usize] + 1;
        } else {
            arr[Direction::UpRight as usize] = arr[Direction::Up as usize] + 1;
        }
        return arr;
    }
    if c == column - 1 {
        arr[Direction::Left as usize] = tile_index - 1;
        if tile_index + column >= count {
            arr[Direction::Up as usize] = tile_index - column;
            arr[Direction::UpLeft as usize] = arr[Direction::Up as usize] - 1;
        } else if tile_index >= column {
            arr[Direction::Up as usize] = tile_index - column;
            arr[Direction::Down as usize] = tile_index + column;
            arr[Direction::UpLeft as usize] = arr[Direction::Up as usize] - 1;
            arr[Direction::DownLeft as usize] = arr[Direction::Down as usize] - 1;
        } else {
            arr[Direction::Down as usize] = tile_index + column;
            arr[Direction::DownLeft as usize] = arr[Direction::Down as usize] - 1;
        }
    } else if c > 0 {
        arr[Direction::Left as usize] = tile_index - 1;
        arr[Direction::Right as usize] = tile_index + 1;
        if tile_index + column >= count {
            arr[Direction::Up as usize] = tile_index - column;
            arr[Direction::UpLeft as usize] = arr[Direction::Up as usize] - 1;
            arr[Direction::UpRight as usize] = arr[Direction::Up as usize] + 1;
        } else if tile_index >= column {
            arr[Direction::Up as usize] = tile_index - column;
            arr[Direction::Down as usize] = tile_index + column;
            arr[Direction::UpLeft as usize] = arr[Direction::Up as usize] - 1;
            arr[Direction::UpRight as usize] = arr[Direction::Up as usize] + 1;
            arr[Direction::DownLeft as usize] = arr[Direction::Down as usize] - 1;
            arr[Direction::DownRight as usize] = arr[Direction::Down as usize] + 1;
        } else {
            arr[Direction::Down as usize] = tile_index + column;
            arr[Direction::DownLeft as usize] = arr[Direction::Down as usize] - 1;
            arr[Direction::DownRight as usize] = arr[Direction::Down as usize] + 1;
        }
    } else {
        arr[Direction::Right as usize] = tile_index + 1;
        if tile_index + column >= count {
            arr[Direction::Up as usize] = tile_index - column;
            arr[Direction::UpRight as usize] = arr[Direction::Up as usize] + 1;
        } else if tile_index >= column {
            arr[Direction::Up as usize] = tile_index - column;
            arr[Direction::Down as usize] = tile_index + column;
            arr[Direction::UpRight as usize] = arr[Direction::Up as usize] + 1;
            arr[Direction::DownRight as usize] = arr[Direction::Down as usize] + 1;
        } else {
            arr[Direction::Down as usize] = tile_index + column;
            arr[Direction::DownRight as usize] = arr[Direction::Down as usize] + 1;
        }
    }
    arr
}
pub struct MapInfo<N: Scalar + RealField + Float + AsPrimitive<usize>> {
    // 场景的范围
    pub bounds: AABB<N>,
    // 该图最大行数
    pub row: usize,
    // 该图最大列数
    pub column: usize,
    // 瓦片总数量
    pub count: usize,
    // 大小
    size: Vector2<N>,
    // 最大行数
    row_n: N,
    // 最大列数
    column_n: N,
}
impl<N: Scalar + RealField + Float + AsPrimitive<usize>> MapInfo<N> {
    /// 计算指定位置的瓦片
    pub fn calc_tile_index(&self, loc: Point2<N>) -> (usize, usize) {
        let c = if loc[0] <= self.bounds.mins[0] {
            0
        } else if loc[0] >= self.bounds.maxs[0] {
            self.column - 1
        } else {
            ((loc[0] - self.bounds.mins[0]) * self.column_n / self.size[0]).as_()
        };
        let r = if loc[1] <= self.bounds.mins[1] {
            0
        } else if loc[1] >= self.bounds.maxs[1] {
            self.row - 1
        } else {
            ((loc[1] - self.bounds.mins[1]) * self.row_n / self.size[1]).as_()
        };
        (r, c)
    }
    /// 获得指定行列瓦片的tile_index
    pub fn tile_index(&self, row: usize, column: usize) -> usize {
        row * self.column + column
    }
    /// 获得指定位置瓦片的行列
    pub fn tile_row_column(&self, tile_index: usize) -> (usize, usize) {
        (tile_index / self.column, tile_index % self.column)
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
pub struct TileMap<N: Scalar + RealField + Float + AsPrimitive<usize>, K: Key, T> {
    //所有存储aabb的节点
    ab_map: SecondaryMap<K, AbNode<K, AABB<N>, T>>,
    // 该图所有瓦片
    tiles: Vec<NodeList<K>>,
    // 场景的范围
    pub info: MapInfo<N>,
}

impl<N: Scalar + RealField + Float + AsPrimitive<usize>, K: Key, T> TileMap<N, K, T> {
    ///
    /// 新建一个瓦片图
    ///
    /// 需传入根节点（即全场景），指定瓦片图的行数和列数
    pub fn new(bounds: AABB<N>, row: usize, column: usize) -> Self {
        let len = row * column;
        let mut tiles = Vec::with_capacity(len);
        tiles.resize_with(len, Default::default);
        let size = bounds.extents();
        let row_n = FromPrimitive::from_usize(row).unwrap();
        let column_n = FromPrimitive::from_usize(column).unwrap();
        let info = MapInfo {
            bounds,
            row,
            column,
            count: row * column,
            size,
            row_n,
            column_n,
        };
        TileMap {
            ab_map: Default::default(),
            tiles,
            info,
        }
    }

    /// 获得指定位置的瓦片，超出地图边界则返回最近的边界瓦片
    pub fn get_tile_index(&self, loc: Point2<N>) -> usize {
        let (r, c) = self.info.calc_tile_index(loc);
        self.info.tile_index(r, c)
    }
    /// 获得指定位置瓦片的节点数量和节点迭代器
    pub fn get_tile_iter<'a>(&'a self, tile_index: usize) -> (usize, Iter<'a, N, K, T>) {
        let tile = &self.tiles[tile_index];
        let id = tile.head;
        (
            tile.len,
            Iter {
                next: id,
                container: &self.ab_map,
            },
        )
    }
    /// 获得指定范围的tile数量和迭代器
    pub fn query_iter(&self, aabb: &AABB<N>) -> (usize, QueryIter) {
        // 获得min所在瓦片
        let (row_start, column_start) = self.info.calc_tile_index(aabb.mins);
        // 获得max所在瓦片
        let (row_end, column_end) = self.info.calc_tile_index(aabb.maxs);
        (
            (column_end - column_start + 1) * (row_end - row_start + 1),
            QueryIter {
                column: self.info.column,
                column_start,
                column_end,
                row_start,
                row_end,
                cur_column: column_start,
            },
        )
    }
    /// 指定id，在地图中添加一个aabb单元及其绑定
    pub fn add(&mut self, id: K, aabb: AABB<N>, bind: T) -> bool {
        let center = aabb.center();
        // 获得所在瓦片
        let tile_index = self.get_tile_index(center);
        // 不在网格范围内
        if tile_index.is_null() {
            return false;
        }
        let next = self.tiles[tile_index].head;
        match self.ab_map.insert(id, AbNode::new(aabb, bind, next)) {
            Some(_) => return false,
            None => (),
        }
        self.tiles[tile_index].add(&mut self.ab_map, id);
        true
    }

    /// 获取指定id的aabb及其绑定
    pub fn get(&self, id: K) -> Option<&(AABB<N>, T)> {
        match self.ab_map.get(id) {
            Some(node) => Some(&node.value),
            None => None,
        }
    }

    /// 获取指定id的aabb及其绑定
    pub unsafe fn get_unchecked(&self, id: K) -> &(AABB<N>, T) {
        &self.ab_map.get_unchecked(id).value
    }

    /// 获取指定id的可写绑定
    pub unsafe fn get_mut(&mut self, id: K) -> Option<&mut T> {
        match self.ab_map.get_mut(id) {
            Some(n) => Some(&mut n.value.1),
            None => None,
        }
    }

    /// 获取指定id的可写绑定
    pub unsafe fn get_unchecked_mut(&mut self, id: K) -> &mut T {
        let node = self.ab_map.get_unchecked_mut(id);
        &mut node.value.1
    }

    /// 检查是否包含某个key
    pub fn contains_key(&self, id: K) -> bool {
        self.ab_map.contains_key(id)
    }

    /// 更新指定id的aabb
    pub fn update(&mut self, id: K, aabb: AABB<N>) -> bool {
        let node = match self.ab_map.get_mut(id) {
            Some(n) => n,
            _ => return false,
        };
        // 获得所在瓦片的位置
        let (new_r, new_c) = self.info.calc_tile_index(aabb.center());
        // 获得原来所在瓦片的位置
        let (r, c) = self.info.calc_tile_index(node.value.0.center());
        node.value.0 = aabb;
        if new_r == r && new_c == c {
            return true;
        }
        let tile_index = self.info.tile_index(r, c);
        let new_tile_index = self.info.tile_index(new_r, new_c);
        let prev = node.prev;
        let next = node.next;
        node.prev = K::null();
        node.next = self.tiles[new_tile_index].head;
        self.tiles[tile_index].remove(&mut self.ab_map, prev, next);
        self.tiles[new_tile_index].add(&mut self.ab_map, id);
        true
    }

    /// 移动指定id的aabb
    pub fn shift(&mut self, id: K, distance: Vector2<N>) -> bool {
        let node = match self.ab_map.get_mut(id) {
            Some(n) => n,
            _ => return false,
        };
        // 新aabb
        let aabb = AABB::new(node.value.0.mins + distance, node.value.0.maxs + distance);
        // 获得新的所在瓦片
        let (new_r, new_c) = self.info.calc_tile_index(aabb.center());
        // 获得原来所在瓦片
        let (r, c) = self.info.calc_tile_index(node.value.0.center());
        if c == new_c && r == new_r {
            node.value.0 = aabb;
            return true;
        }
        let new_tile_index = self.info.tile_index(new_r, new_c);
        let tile_index = self.info.tile_index(r, c);
        node.value.0 = aabb;
        let prev = node.prev;
        let next = node.next;
        node.prev = K::null();
        self.tiles[tile_index].remove(&mut self.ab_map, prev, next);
        self.tiles[new_tile_index].add(&mut self.ab_map, id);
        true
    }

    /// 更新指定id的绑定
    pub fn update_bind(&mut self, id: K, bind: T) -> bool {
        match self.ab_map.get_mut(id) {
            Some(node) => {
                node.value.1 = bind;
                true
            }
            _ => false,
        }
    }
    /// 移除指定id的aabb及其绑定
    pub fn remove(&mut self, id: K) -> Option<(AABB<N>, T)> {
        let node = match self.ab_map.remove(id) {
            Some(n) => n,
            _ => return None,
        };
        let tile_index = self.get_tile_index(node.value.0.center());
        self.tiles[tile_index].remove(&mut self.ab_map, node.prev, node.next);
        Some((node.value.0, node.value.1))
    }
    /// 获得指定id的所在的tile
    pub fn get_tile_index_by_id(&self, id: K) -> usize {
        let node = match self.ab_map.get(id) {
            Some(n) => n,
            _ => return Null::null(),
        };
        // 获得新的所在瓦片
        let (r, c) = self.info.calc_tile_index(node.value.0.center());
        self.info.tile_index(r, c)
    }
    /// 获得节点数量
    pub fn len(&self) -> usize {
        self.ab_map.len()
    }
}

//////////////////////////////////////////////////////本地/////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy, Default)]
struct NodeList<K> {
    head: K,
    len: usize,
}
impl<K: Key> NodeList<K> {
    #[inline]
    pub fn add<Aabb, T>(&mut self, map: &mut SecondaryMap<K, AbNode<K, Aabb, T>>, id: K) {
        if !self.head.is_null() {
            let n = unsafe { map.get_unchecked_mut(self.head) };
            n.prev = id;
        }
        self.head = id;
        self.len += 1;
    }
    #[inline]
    pub fn remove<Aabb, T>(
        &mut self,
        map: &mut SecondaryMap<K, AbNode<K, Aabb, T>>,
        prev: K,
        next: K,
    ) {
        if !prev.is_null() {
            let node = unsafe { map.get_unchecked_mut(prev) };
            node.next = next;
        } else {
            self.head = next;
        }
        if !next.is_null() {
            let node = unsafe { map.get_unchecked_mut(next) };
            node.prev = prev;
        }
        self.len -= 1;
    }
}

#[derive(Debug, Clone)]
pub struct AbNode<K: Key, Aabb, T> {
    value: (Aabb, T), // 包围盒
    prev: K,          // 前ab节点
    next: K,          // 后ab节点
}
impl<K: Key, Aabb, T> AbNode<K, Aabb, T> {
    pub fn new(aabb: Aabb, bind: T, next: K) -> Self {
        AbNode {
            value: (aabb, bind),
            prev: K::null(),
            next,
        }
    }
}

#[derive(Clone)]
pub struct Iter<'a, N: Scalar + RealField + Float + AsPrimitive<usize>, K: Key, T> {
    next: K,
    container: &'a SecondaryMap<K, AbNode<K, AABB<N>, T>>,
}

impl<'a, N: Scalar + RealField + Float + AsPrimitive<usize>, K: Key, T> Iterator
    for Iter<'a, N, K, T>
{
    type Item = (K, &'a AABB<N>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.is_null() {
            return None;
        }
        let node = unsafe { self.container.get_unchecked(self.next) };
        let id = self.next;
        self.next = node.next;
        Some((id, &node.value.0, &node.value.1))
    }
}
#[derive(Debug, Clone, Default)]
pub struct QueryIter {
    column: usize,
    column_start: usize,
    column_end: usize,
    row_start: usize,
    row_end: usize,
    cur_column: usize,
}

impl Iterator for QueryIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_start > self.row_end {
            return None;
        }
        let index = self.column * self.row_start + self.cur_column;
        if self.cur_column < self.column_end {
            self.cur_column += 1;
        } else {
            self.cur_column = self.column_start;
            self.row_start += 1;
        }
        Some(index)
    }
}

#[test]
fn test1() {
    use pi_slotmap::{DefaultKey, SlotMap};

    println!("test1-----------------------------------------");
    let mut tree = TileMap::new(
        AABB::new(
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
            AABB::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
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
        AABB::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
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
            AABB::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
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
        AABB::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
    );
    tree.update(
        keys[3],
        AABB::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
    );

    tree.update(
        keys[4],
        AABB::new(Point2::new(0.0, 700.0), Point2::new(1000.0, 1400.0)),
    );

    tree.update(
        keys[5],
        AABB::new(Point2::new(0.0, 1400.0), Point2::new(1000.0, 1470.0)),
    );
    tree.update(
        keys[6],
        AABB::new(Point2::new(0.0, 1470.0), Point2::new(1000.0, 1540.0)),
    );
    tree.update(
        keys[1],
        AABB::new(Point2::new(0.0, 0.0), Point2::new(1000.0, 700.0)),
    );

    for i in 1..tree.ab_map.len() + 1 {
        println!(
            "00002, id:{}, ab: {:?}, index: {:?}",
            i,
            tree.ab_map.get(keys[i]).unwrap(),
            tree.get_tile_index_by_id(keys[i])
        );
    }
    //   tree.update(1, AABB::new(Point2::new(0.0,0.0,0.0), Point2::new(1000.0, 800.0, 1.0)));
    //   tree.update(2, AABB::new(Point2::new(0.0,0.0,0.0), Point2::new(1000.0, 800.0, 1.0)));
    //   tree.update(3, AABB::new(Point2::new(0.0,0.0,0.0), Point2::new(1000.0, 800.0, 1.0)));
    //   tree.update(4, AABB::new(Point2::new(0.0,0.0,0.0), Point2::new(1000.0, 800.0, 1.0)));

    //   tree.update(5, AABB::new(Point2::new(0.0,800.0,0.0), Point2::new(1000.0, 1600.0, 1.0)));

    //    tree.update(6, AABB::new(Point2::new(0.0,1600.0,0.0), Point2::new(1000.0, 2400.0, 1.0)));
    //   tree.update(7, AABB::new(Point2::new(0.0,2400.0,0.0), Point2::new(1000.0, 3200.0, 1.0)));
    //   for i in 1..tree.ab_map.len() + 1 {
    //   println!("22222, id:{}, ab: {:?}", i, tree.ab_map.get(i).unwrap());
    //  }
    // tree.collect();
    let aabb = AABB::new(Point2::new(500f32, 500f32), Point2::new(1100f32, 1100f32));
    let (len, iter) = tree.query_iter(&aabb);
    println!("query_iter count:{},", len);
    for i in iter {
        println!(
            "id:{}, r_c: {:?} 4: {:?} 8: {:?}",
            i,
            tree.info.tile_row_column(i),
            get_4d_neighbors(i, tree.info.column, tree.info.count),
            get_8d_neighbors(i, tree.info.column, tree.info.count)
        );
    }
    //assert_eq!(args.result(), [1, 3, 4]);
}
