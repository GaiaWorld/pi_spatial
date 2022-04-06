//! 高性能的松散叉树
//! 采用二进制掩码 表达xyz的大小， child&1 == 0 表示x为小，否则为大。
//! 采用Slab，内部用偏移量来分配八叉节点。这样内存连续，八叉树本身可以快速拷贝。
//! 要求插入AABB节点时的id， 应该是可以用在数组索引上的。

use pi_slotmap::{SlotMap, SecondaryMap, Key};

pub trait Helper<const N: usize> {
    type Point;
    type Vector: Clone;
    type Aabb: Clone;

    /// 获得AABB的差
    fn aabb_extents(aabb: &Self::Aabb) -> Self::Vector;
    /// 移动AABB
    fn aabb_shift(aabb: &Self::Aabb, distance: &Self::Vector) -> Self::Aabb;
    /// 判断指定的aabb是否包含另一个aabb
    fn aabb_contains(aabb: &Self::Aabb, other: &Self::Aabb) -> bool;
    /// 判断2个aabb是否相交
    fn aabb_intersects(aabb: &Self::Aabb, other: &Self::Aabb) -> bool;
    /// 计算叉树的深度
    fn get_deap(
        d: &mut Self::Vector,
        loose_layer: usize,
        max_loose: &Self::Vector,
        deep: usize,
        min_loose: &Self::Vector,
    ) -> usize;
    /// 判定指定向量是否小于最小“松散”尺寸
    fn smaller_than_min_loose(d: &Self::Vector, min_loose: &Self::Vector) -> bool;
    /// 获得指定向量以及最大松散尺寸计算对应的层
    fn calc_layer(loose: &Self::Vector, el: &Self::Vector) -> usize;
    ///  判断所在的子节点
    fn get_child(point: &Self::Point, aabb: &Self::Aabb) -> usize;
    ///  获得所在的AABB的最大half loose
    fn get_max_half_loose(aabb: &Self::Aabb, loose: &Self::Vector) -> Self::Point;
    ///  获得所在的AABB的最小half loose
    //fn get_min_half_loose(aabb: &Self::Aabb, loose: &Self::Vector) -> Self::Point;
    /// 创建ab的子节点集合
    fn make_childs(aabb: &Self::Aabb, loose: &Self::Vector) -> [Self::Aabb; N];
    /// 指定创建ab的子节点
    fn create_child(
        aabb: &Self::Aabb,
        loose: &Self::Vector,
        layer: usize,
        loose_layer: usize,
        min_loose: &Self::Vector,
        child_index: usize,
    ) -> (Self::Aabb, Self::Vector);
}

const DEEP_MAX: usize = 16;
const ADJUST_MIN: usize = 4;
const ADJUST_MAX: usize = 7;
///
/// 叉树结构体
///
/// ### 对`N`的约束
///
/// + 浮点数算术运算，可拷贝，可偏序比较；
/// + 实际使用的时候就是浮点数字类型，比如：f32/f64；
///
pub struct Tree<K: Key, H: Helper<N>, T, const N: usize> {
    pub slab: SlotMap<K, BranchNode<K, H, N>>, //所有分支节点（分支节点中包含该层ab节点列表）
    pub ab_map: SecondaryMap<K, AbNode<K, H::Aabb, T>>,     //所有存储ab碰撞单位的节点
    max_loose: H::Vector,                //最大松散值，第一层的松散大小
    min_loose: H::Vector,                //最小松散值
    adjust: (usize, usize),              //小于min，节点收缩; 大于max，节点分化。默认(4, 7)
    loose_layer: usize,                  // 最小松散值所在的深度
    deep: usize,                         // 最大深度, 推荐12-16
	root_key: K,
    pub outer: NodeList<K>, // 和根节点不相交的ab节点列表，及节点数量。 相交的放在root的nodes上了。 该AbNode的parent为0
    pub dirty: (Vec<Vec<K>>, usize, usize), // 脏的BranchNode节点, 及脏节点数量，及脏节点的起始层
}

impl<K: Key, H: Helper<N>, T, const N: usize> Tree<K, H, T, N> {
    ///构建树
    ///
    /// 需传入根节点（即全场景）AB碰撞范围；N维实际距离所表示的最大及最小松散参数；叉树收缩及分裂的阈值；叉树的深度限制
    ///
    /// ### 对`N`的约束
    ///
    /// + 浮点数算术运算，可拷贝，可偏序比较；
    /// + 实际使用的时候就是浮点数字类型，比如：f32/f64；
    ///
    pub fn new(
        root: H::Aabb,
        max_loose: H::Vector,
        min_loose: H::Vector,
        adjust_min: usize,
        adjust_max: usize,
        deep: usize,
    ) -> Self {
        let adjust_min = if adjust_min == 0 {
            ADJUST_MIN
        } else {
            adjust_min
        };
        let adjust_max = if adjust_max == 0 {
            ADJUST_MAX
        } else {
            adjust_max
        };
        let adjust_max = adjust_min.max(adjust_max);
        let deep = if deep > DEEP_MAX { DEEP_MAX } else { deep };
        let mut branch_slab: SlotMap<K, BranchNode<K, H, N>> = SlotMap::with_key();
        let mut d = H::aabb_extents(&root);
        // 根据最大 最小 松散值 计算出最小松散值所在的最大的层
        let loose_layer = H::calc_layer(&max_loose, &min_loose);
        let deep = H::get_deap(&mut d, loose_layer, &max_loose, deep, &min_loose);

        let root = branch_slab.insert(BranchNode::new(root, max_loose.clone(), K::null(), 0, 0));
        return Tree {
            slab: branch_slab,
            ab_map: SecondaryMap::default(),
            max_loose,
            min_loose,
            adjust: (adjust_min, adjust_max),
            loose_layer,
            deep,
			root_key: root,
            outer: NodeList::new(),
            dirty: (Vec::new(), 0, usize::max_value()),
        };
    }

    // /// 获得叉树总的占有内存的字节数
    // pub fn mem_size(&self) -> usize {
    //     self.slab.mem_size()
    //         + self.ab_map.mem_size()
    //         + self.outer.len() * std::mem::size_of::<usize>()

    // }

    /// 获得节点收缩和分化的阈值
    pub fn get_adjust(&self) -> (usize, usize) {
        (self.adjust.0, self.adjust.1)
    }

    /// 获得该aabb对应的层
    pub fn get_layer(&self, aabb: &H::Aabb) -> usize {
        let d = H::aabb_extents(aabb);
        if H::smaller_than_min_loose(&d, &self.min_loose) {
            return self.deep;
        };

        H::calc_layer(&self.max_loose, &d)
    }

    /// 指定id，在叉树中添加一个aabb单元及其绑定
    pub fn add(&mut self, id: K, aabb: H::Aabb, bind: T) {
        let layer = self.get_layer(&aabb);
        match self.ab_map.insert(id, AbNode::new(aabb, bind, layer, N)) {
            Some(_) => return,// panic!("duplicate id: {}", id),
            _ => (),
        }
        let next = {
            let node = unsafe { self.ab_map.get_unchecked_mut(id) };
            let root = unsafe { self.slab.get_unchecked_mut(self.root_key) };
            if H::aabb_contains(&root.aabb, &node.value.0) {
                // root的ab内
                set_tree_dirty(
                    &mut self.dirty,
                    down(&mut self.slab, self.adjust.1, self.deep, self.root_key, node, id),
                );
            } else if H::aabb_intersects(&root.aabb, &node.value.0) {
                // 相交的放在root的nodes上
                node.parent = self.root_key;
                node.next = root.nodes.head;
                root.nodes.push(id);
            } else {
                // 和根节点不相交的ab节点, 该AbNode的parent为0
                node.next = self.outer.head;
                self.outer.push(id);
            }
            node.next
        };

        if !next.is_null() {
            let n = unsafe { self.ab_map.get_unchecked_mut(next) };
            n.prev = id;
        }
    }

    /// 获取指定id的aabb及其绑定
    /// + 该接口返回Option
    pub fn get(&self, id: K) -> Option<&(H::Aabb, T)> {
        match self.ab_map.get(id) {
            Some(node) => Some(&node.value),
            _ => None,
        }
    }

	/// 获取指定id的aabb及其绑定
	pub unsafe fn get_unchecked(&self, id: K) -> &(H::Aabb, T) {
		&self.ab_map.get_unchecked (id).value
    }

	/// 获取指定id的可写绑定
    pub unsafe fn get_mut(&mut self, id: K) -> Option<&mut (H::Aabb, T)> {
        match self.ab_map.get_mut(id) {
            Some(n) => Some(&mut n.value),
            _ => None,
        }
    }

    /// 获取指定id的可写绑定
    pub unsafe fn get_unchecked_mut(&mut self, id: K) -> &mut (H::Aabb, T) {
        let node = self.ab_map.get_unchecked_mut(id);
        &mut node.value
    }

	/// 检查是否包含某个key
	pub fn contains_key(&self, id: K) -> bool {
		self.ab_map.contains_key(id)
	}

    /// 更新指定id的aabb
    pub fn update(&mut self, id: K, aabb: H::Aabb) -> bool {
        let layer = self.get_layer(&aabb);
        let r = match self.ab_map.get_mut(id) {
            Some(node) => {
                node.layer = layer;
                node.value.0 = aabb;
                update(
                    &mut self.slab,
                    &self.adjust,
                    self.deep,
                    &mut self.outer,
                    &mut self.dirty,
                    id,
                    node,
					self.root_key,
                )
            }
            _ => return false,
        };
        remove_add(self, id, r);
        true
    }

    /// 移动指定id的aabb，性能比update要略好
    pub fn shift(&mut self, id: K, distance: H::Vector) -> bool {
        let r = match self.ab_map.get_mut(id) {
            Some(node) => {
                node.value.0 = H::aabb_shift(&node.value.0, &distance);
                update(
                    &mut self.slab,
                    &self.adjust,
                    self.deep,
                    &mut self.outer,
                    &mut self.dirty,
                    id,
                    node,
					self.root_key,
                )
            }
            _ => return false,
        };
        remove_add(self, id, r);
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
    pub fn remove(&mut self, id: K) -> Option<(H::Aabb, T)> {
        let node = match self.ab_map.remove(id) {
            Some(n) => n,
            _ => return None,
        };
        if !node.parent.is_null() {
            let (p, c) = {
                let parent = unsafe { self.slab.get_unchecked_mut(node.parent) };
                if node.parent_child < N {
                    // 在节点的childs上
                    match parent.childs[node.parent_child] {
                        ChildNode::Ab(ref mut ab) => {
                            ab.remove(&mut self.ab_map, node.prev, node.next)
                        }
                        _ => panic!("invalid state"),
                    }
                } else {
                    // 在节点的nodes上
                    parent.nodes.remove(&mut self.ab_map, node.prev, node.next);
                }
                (parent.parent, parent.parent_child)
            };
            remove_up(&mut self.slab, self.adjust.0, &mut self.dirty, p, c);
        } else {
            // 表示在outer上
            self.outer.remove(&mut self.ab_map, node.prev, node.next);
        }
        Some((node.value.0, node.value.1))
    }

    /// 整理方法，只有整理方法才会创建或销毁BranchNode
    pub fn collect(&mut self) {
        let mut count = self.dirty.1;
        if count == 0 {
            return;
        }
        let min_loose = self.min_loose.clone();
        for i in self.dirty.2..self.dirty.0.len() {
            let vec = unsafe { self.dirty.0.get_unchecked_mut(i) };
            let c = vec.len();
            if c == 0 {
                continue;
            }
            for j in 0..c {
                let branch_id = unsafe { vec.get_unchecked(j) };
                collect(
                    &mut self.slab,
                    &mut self.ab_map,
                    &self.adjust,
                    self.deep,
                    *branch_id,
                    self.loose_layer,
                    &min_loose,
                );
            }
            vec.clear();
            if count <= c {
                break;
            }
            count -= c;
        }
        self.dirty.1 = 0;
        self.dirty.2 = usize::max_value();
    }

    /// 查询空间内及相交的ab节点
    pub fn query<A, B>(
        &self,
        branch_arg: &A,
        branch_func: fn(arg: &A, aabb: &H::Aabb) -> bool,
        ab_arg: &mut B,
        ab_func: fn(arg: &mut B, id: K, aabb: &H::Aabb, bind: &T),
    ) {
        query(
            &self.slab,
            &self.ab_map,
            self.root_key,
            branch_arg,
            branch_func,
            ab_arg,
            ab_func,
        )
    }

    /// 查询空间外的ab节点
    pub fn query_outer<B>(
        &self,
        arg: &mut B,
        func: fn(arg: &mut B, id: K, aabb: &H::Aabb, bind: &T),
    ) {
        let mut id = self.outer.head;
        while !id.is_null() {
            let ab = unsafe { self.ab_map.get_unchecked(id) };
            func(arg, id, &ab.value.0, &ab.value.1);
            id = ab.next;
        }
    }

	pub fn len(&self) -> usize {
		self.ab_map.len()
	}

    // 检查碰撞对，不会检查outer的aabb。一般arg包含1个hashset，用(big, little)做键，判断是否已经计算过。
    // pub fn collision<A>(
    //     &self,
    //     id: K,
    //     _limit_layer: usize,
    //     arg: &mut A,
    //     func: fn(
    //         arg: &mut A,
    //         a_id: usize,
    //         a_aabb: &H::AABB,
    //         a_bind: &T,
    //         b_id: usize,
    //         b_aabb: &H::AABB,
    //         b_bind: &T,
    //     ) -> bool,
    // ) {
    //     let a = match self.ab_map.get(id) {
    //         Some(ab) => ab,
    //         _ => return,
    //     };
    //     // 先判断root.nodes是否有节点，如果有则遍历root的nodes
    //     let node = unsafe { self.branch_slab.get_unchecked(1) };
    //     collision_list(
    //         &self.ab_map,
    //         id,
    //         &a.aabb,
    //         &a.value.1,
    //         arg,
    //         func,
    //         node.nodes.head,
    //     );
    //     // 和同列表节点碰撞
    //     collision_list(&self.ab_map, id, &a.aabb, &a.value.1, arg, func, a.next);
    //     let mut prev = a.prev;
    //     while prev > 0 {
    //         let b = unsafe { self.ab_map.get_unchecked(prev) };
    //         func(arg, id, &a.aabb, &a.value.1, prev, &b.aabb, &b.value.1);
    //         prev = b.prev;
    //     }
    //     // 需要计算是否在重叠区，如果在，则需要上溯检查重叠的兄弟节点。不在，其实也需要上溯检查父的匹配节点，但可以提前计算ab节点的最小层
    //     //}
    // }
}

//////////////////////////////////////////////////////本地/////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy)]
pub struct NodeList<K> {
    head: K,
    len: usize,
}
impl<K: Key> NodeList<K> {
    #[inline]
    pub fn new() -> NodeList<K> {
        NodeList { head: K::null(), len: 0 }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn push(&mut self, id: K) {
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
pub struct BranchNode<K: Key, H: Helper<N>, const N: usize> {
    aabb: H::Aabb,          // 包围盒
    loose: H::Vector,       // 本层的松散值
    parent: K,          // 父八叉节点
    parent_child: usize,    // 对应父八叉节点childs的位置
    childs: [ChildNode<K>; N], // 子八叉节点
    layer: usize,           // 表示第几层， 根据aabb大小，决定最低为第几层
    nodes: NodeList<K>,        // 匹配本层大小的ab节点列表，及节点数量
    dirty: usize, // 脏标记, 1-128对应节点被修改。添加了节点，并且某个子八叉节点(AbNode)的数量超过阈值，可能分化。删除了节点，并且自己及其下ab节点的数量超过阈值，可能收缩
}
impl<K: Key, H: Helper<N>, const N: usize> BranchNode<K, H, N> {
    #[inline]
    pub fn new(aabb: H::Aabb, loose: H::Vector, parent: K, child: usize, layer: usize) -> Self {
        BranchNode {
            aabb,
            loose,
            parent,
            parent_child: child,
            childs: [ChildNode::Ab(NodeList::new()); N],
            layer,
            nodes: NodeList::new(),
            dirty: 0,
        }
    }
}
#[derive(Debug, Clone, Copy)]
enum ChildNode<K> {
    Branch(K, usize), // 对应的BranchNode, 及其下ab节点的数量
    Ab(NodeList<K>),         // ab节点列表，及节点数量
}

#[derive(Debug, Clone)]
pub struct AbNode<K: Key, Aabb, T> {
    value: (Aabb, T),       // 包围盒
    layer: usize,        // 表示第几层， 根据aabb大小，决定最低为第几层
    parent: K,       // 父八叉节点
    parent_child: usize, // 父八叉节点所在的子八叉节点， 8表示不在子八叉节点上
    prev: K,         // 前ab节点
    next: K,         // 后ab节点
}
impl<K: Key, Aabb, T> AbNode<K, Aabb, T> {
    pub fn new(aabb: Aabb, bind: T, layer: usize, n: usize) -> Self {
        AbNode {
            value: (aabb, bind),
            layer: layer,
            parent: K::null(),
            parent_child: n,
            prev: K::null(),
            next: K::null(),
        }
    }
}

// ab节点下降
fn down<K: Key, H: Helper<N>, T, const N: usize>(
    slab: &mut SlotMap<K, BranchNode<K, H, N>>,
    adjust: usize,
    deep: usize,
    branch_id: K,
    node: &mut AbNode<K, H::Aabb, T>,
    id: K,
) -> (usize, K) {
    let parent = unsafe { slab.get_unchecked_mut(branch_id) };
    if parent.layer >= node.layer {
        node.parent = branch_id;
        node.next = parent.nodes.head;
        parent.nodes.push(id);
        return (0, K::null());
    }
    let i = H::get_child(
        &H::get_max_half_loose(&parent.aabb, &parent.loose),
        &node.value.0,
    );
    match parent.childs[i] {
        ChildNode::Branch(branch, ref mut num) => {
            *num += 1;
            return down(slab, adjust, deep, branch, node, id);
        }
        ChildNode::Ab(ref mut list) => {
            node.parent = branch_id;
            node.parent_child = i;
            node.next = list.head;
            list.push(id);
            if list.len > adjust && parent.layer < deep {
                return set_dirty(&mut parent.dirty, i, parent.layer, branch_id);
            }
            return (0, K::null());
        }
    }
}

// 更新aabb
fn update<K: Key, H: Helper<N>, T, const N: usize>(
    slab: &mut SlotMap<K, BranchNode<K, H, N>>,
    adjust: &(usize, usize),
    deep: usize,
    outer: &mut NodeList<K>,
    dirty: &mut (Vec<Vec<K>>, usize, usize),
    id: K,
    node: &mut AbNode<K, H::Aabb, T>,
	root_key: K,
) -> Option<(K, usize, K, K, K)> {
    let old_p = node.parent;
    if !old_p.is_null() {
        let old_c = node.parent_child;
        let mut parent = unsafe { slab.get_unchecked_mut(old_p) };
        if node.layer > parent.layer {
            // ab节点能在当前branch节点的容纳范围
            if H::aabb_contains(&parent.aabb, &node.value.0) {
                // 获得新位置
                let child = H::get_child(
                    &H::get_max_half_loose(&parent.aabb, &parent.loose),
                    &node.value.0,
                );
                if old_c == child {
                    return None;
                }
                if child < N {
                    let prev = node.prev;
                    let next = node.next;
                    node.prev = K::null();
                    // 移动到兄弟节点
                    match parent.childs[child] {
                        ChildNode::Branch(branch, ref mut num) => {
                            *num += 1;
                            node.parent_child = N;
                            set_tree_dirty(dirty, down(slab, adjust.1, deep, branch, node, id));
                            return Some((old_p, old_c, prev, next, node.next));
                        }
                        ChildNode::Ab(ref mut list) => {
                            node.parent_child = child;
                            node.next = list.head;
                            list.push(id);
                            if list.len > adjust.1 && node.layer < deep {
                                set_dirty(&mut parent.dirty, child, parent.layer, id);
                            }
                            return Some((old_p, old_c, prev, next, node.next));
                        }
                    }
                }
            }
        // 需要向上
        } else if node.layer == parent.layer {
            if H::aabb_contains(&parent.aabb, &node.value.0) {
                if old_c == N {
                    return None;
                }
                let prev = node.prev;
                let next = node.next;
                node.prev = K::null();
                // 从child 移到 nodes
                node.parent_child = N;
                node.next = parent.nodes.head;
                parent.nodes.push(id);
                return Some((old_p, old_c, prev, next, node.next));
            }
        // 在当前节点外
        } else {
            // 比当前节点大
        };
        let prev = node.prev;
        let next = node.next;
        if old_p != root_key {
            // 向上移动
            let mut p = parent.parent;
            let mut c = parent.parent_child;
            loop {
                parent = unsafe { slab.get_unchecked_mut(p) };
                match parent.childs[c] {
                    ChildNode::Branch(_, ref mut num) => {
                        *num -= 1;
                        if *num < adjust.0 {
                            let d = set_dirty(&mut parent.dirty, c, parent.layer, p);
                            if !d.1.is_null() {
                                set_tree_dirty(dirty, d);
                            }
                        }
                    }
                    _ => panic!("invalid state"),
                }
                if parent.layer <= node.layer && H::aabb_contains(&parent.aabb, &node.value.0) {
                    node.prev = K::null();
                    node.parent_child = N;
                    set_tree_dirty(dirty, down(slab, adjust.1, deep, p, node, id));
                    return Some((old_p, old_c, prev, next, node.next));
                }
                p = parent.parent;
                c = parent.parent_child;
                if p.is_null() {
                    break;
                }
            }
        }
        // 判断根节点是否相交
        if H::aabb_intersects(&parent.aabb, &node.value.0) {
            if old_p == root_key && old_c == N {
                return None;
            }
            // 相交的放在root的nodes上
            node.parent = root_key;
            node.next = parent.nodes.head;
            parent.nodes.push(id);
        } else {
            node.parent = K::null();
            node.next = outer.head;
            outer.push(id);
        }
        node.prev = K::null();
        node.parent_child = N;
        return Some((old_p, old_c, prev, next, node.next));
    } else {
        // 边界外物体更新
        let root = unsafe { slab.get_unchecked_mut(root_key) };
        if H::aabb_intersects(&root.aabb, &node.value.0) {
            // 判断是否相交或包含
            let prev = node.prev;
            let next = node.next;
            node.prev = K::null();
            node.parent_child = N;
            if H::aabb_contains(&root.aabb, &node.value.0) {
                set_tree_dirty(dirty, down(slab, adjust.1, deep, root_key, node, id));
            } else {
                // 相交的放在root的nodes上
                node.parent = K::null();
                node.next = root.nodes.head;
                root.nodes.push(id);
            }
            Some((K::null(), 0, prev, next, node.next))
        } else {
            // 表示还在outer上
            None
        }
    }
}

/// 从NodeList中移除，并可能添加
pub fn remove_add<K: Key, H: Helper<N>, T, const N: usize>(
    tree: &mut Tree<K, H, T, N>,
    id: K,
    r: Option<(K, usize, K, K, K)>,
) {
    // 从NodeList中移除
    if let Some((rid, child, prev, next, cur_next)) = r {
        if !rid.is_null() {
            let branch = unsafe { tree.slab.get_unchecked_mut(rid) };
            if child < N {
                match branch.childs[child] {
                    ChildNode::Ab(ref mut ab) => ab.remove(&mut tree.ab_map, prev, next),
                    _ => panic!("invalid state"),
                }
            } else {
                branch.nodes.remove(&mut tree.ab_map, prev, next);
            }
        } else {
            tree.outer.remove(&mut tree.ab_map, prev, next);
        }
        if !cur_next.is_null() {
            let n = unsafe { tree.ab_map.get_unchecked_mut(cur_next) };
            n.prev = id;
        }
    }
}

// 移除时，向上修改数量，并可能设脏
#[inline]
fn remove_up<K: Key, H: Helper<N>, const N: usize>(
    slab: &mut SlotMap<K, BranchNode<K, H, N>>,
    adjust: usize,
    dirty: &mut (Vec<Vec<K>>, usize, usize),
    parent: K,
    child: usize,
) {
    if parent.is_null() {
        return;
    }
    let (p, c) = {
        let node = unsafe { slab.get_unchecked_mut(parent) };
        match node.childs[child] {
            ChildNode::Branch(_, ref mut num) => {
                *num -= 1;
                if *num < adjust {
                    let d = set_dirty(&mut node.dirty, child, node.layer, parent);
                    if !d.1.is_null() {
                        set_tree_dirty(dirty, d);
                    }
                }
            }
            _ => panic!("invalid state"),
        }
        (node.parent, node.parent_child)
    };
    remove_up(slab, adjust, dirty, p, c);
}

#[inline]
fn set_dirty<K: Key>(dirty: &mut usize, index: usize, layer: usize, rid: K) -> (usize, K) {
    if *dirty == 0 {
        *dirty |= 1 << index;
        return (layer, rid);
    }
    *dirty |= 1 << index;
    return (0, K::null());
}
// 设置脏标记
#[inline]
fn set_tree_dirty<K: Key>(dirty: &mut (Vec<Vec<K>>, usize, usize), (layer, rid): (usize, K)) {
    if rid.is_null() {
        return;
    }
    dirty.1 += 1;
    if dirty.2 > layer {
        dirty.2 = layer;
    }
    if dirty.0.len() <= layer {
        for _ in dirty.0.len()..layer + 1 {
            dirty.0.push(Vec::new())
        }
    }
    let vec = unsafe { dirty.0.get_unchecked_mut(layer) };
    vec.push(rid);
}

// 创建指定的子节点
fn create_child<K: Key, H: Helper<N>, const N: usize>(
    aabb: &H::Aabb,
    loose: &H::Vector,
    layer: usize,
    parent_id: K,
    loose_layer: usize,
    min_loose: &H::Vector,
    child: usize,
) -> BranchNode<K, H, N> {
    let (ab, loose) = H::create_child(aabb, loose, layer, loose_layer, min_loose, child);
    BranchNode::new(ab, loose, parent_id, child, layer + 1)
}

// 整理方法，只有整理方法才会创建或销毁BranchNode
fn collect<K: Key, H: Helper<N>, T, const N: usize>(
    branch_slab: &mut SlotMap<K, BranchNode<K, H, N>>,
    ab_map: &mut SecondaryMap<K, AbNode<K, H::Aabb, T>>,
    adjust: &(usize, usize),
    deep: usize,
    parent_id: K,
    loose_layer: usize,
    min_loose: &H::Vector,
) {
    let (dirty, childs, ab, loose, layer) = {
        let parent = match branch_slab.get_mut(parent_id) {
            Some(branch) => branch,
            _ => return,
        };
        let dirty = parent.dirty;
        if parent.dirty == 0 {
            return;
        }
        parent.dirty = 0;
        (
            dirty,
            parent.childs.clone(),
            parent.aabb.clone(),
            parent.loose.clone(),
            parent.layer,
        )
    };
    for i in 0..N {
        if dirty & (1 << i) != 0 {
            match childs[i] {
                ChildNode::Branch(branch, num) if num < adjust.0 => {
                    let mut list = NodeList::new();
                    if num > 0 {
                        shrink(branch_slab, ab_map, parent_id, i, branch, &mut list);
                    }
                    let parent = unsafe { branch_slab.get_unchecked_mut(parent_id) };
                    parent.childs[i] = ChildNode::Ab(list);
                }
                ChildNode::Ab(ref list) if list.len > adjust.1 => {
                    let child_id = split(
                        branch_slab,
                        ab_map,
                        adjust,
                        deep,
                        list,
                        &ab,
                        &loose,
                        layer,
                        parent_id,
                        loose_layer,
                        min_loose,
                        i,
                    );
                    let parent = unsafe { branch_slab.get_unchecked_mut(parent_id) };
                    parent.childs[i] = ChildNode::Branch(child_id, list.len);
                }
                _ => (),
            }
        }
    }
}
// 收缩BranchNode
fn shrink<K: Key, H: Helper<N>, T, const N: usize>(
    branch_slab: &mut SlotMap<K, BranchNode<K, H, N>>,
    ab_map: &mut SecondaryMap<K, AbNode<K, H::Aabb, T>>,
    parent: K,
    parent_child: usize,
    branch_id: K,
    result: &mut NodeList<K>,
) {
    let node = branch_slab.remove(branch_id).unwrap();
    if node.nodes.len > 0 {
        shrink_merge(ab_map, parent, parent_child, &node.nodes, result);
    }
    for index in 0..N {
        match node.childs[index] {
            ChildNode::Ab(ref list) if list.len > 0 => {
                shrink_merge(ab_map, parent, parent_child, &list, result);
            }
            ChildNode::Branch(branch, len) if len > 0 => {
                shrink(branch_slab, ab_map, parent, parent_child, branch, result);
            }
            _ => (),
        }
    }
}
// 合并ab列表到结果列表中
#[inline]
fn shrink_merge<K: Key, Aabb, T>(
    ab_map: &mut SecondaryMap<K, AbNode<K, Aabb, T>>,
    parent: K,
    parent_child: usize,
    list: &NodeList<K>,
    result: &mut NodeList<K>,
) {
    let old = result.head;
    result.head = list.head;
    result.len += list.len;
    let mut id = list.head;
    loop {
        let ab = unsafe { ab_map.get_unchecked_mut(id) };
        ab.parent = parent;
        ab.parent_child = parent_child;
        if ab.next.is_null() {
            ab.next = old;
            break;
        }
        id = ab.next;
    }
    if !old.is_null() {
        let ab = unsafe { ab_map.get_unchecked_mut(old) };
        ab.prev = id;
    }
}

// 分裂出BranchNode
#[inline]
fn split<K: Key, H: Helper<N>, T, const N: usize>(
    branch_slab: &mut SlotMap<K, BranchNode<K, H, N>>,
    ab_map: &mut SecondaryMap<K, AbNode<K, H::Aabb, T>>,
    adjust: &(usize, usize),
    deep: usize,
    list: &NodeList<K>,
    parent_ab: &H::Aabb,
    parent_loose: &H::Vector,
    parent_layer: usize,
    parent_id: K,
    loose_layer: usize,
    min_loose: &H::Vector,
    child: usize,
) -> K {
    let branch = create_child(
        parent_ab,
        parent_loose,
        parent_layer,
        parent_id,
        loose_layer,
        min_loose,
        child,
    );
    let branch_id = branch_slab.insert(branch);
    let branch = unsafe { branch_slab.get_unchecked_mut(branch_id) };
    if split_down(ab_map, adjust.1, deep, branch, branch_id, list) > 0 {
        collect(
            branch_slab,
            ab_map,
            adjust,
            deep,
            branch_id,
            loose_layer,
            min_loose,
        );
    }
    branch_id
}
// 将ab节点列表放到分裂出来的八叉节点上
fn split_down<K: Key, H: Helper<N>, T, const N: usize>(
    map: &mut SecondaryMap<K, AbNode<K, H::Aabb, T>>,
    adjust: usize,
    deep: usize,
    parent: &mut BranchNode<K, H, N>,
    parent_id: K,
    list: &NodeList<K>,
) -> usize {
    let point = H::get_max_half_loose(&parent.aabb, &parent.loose);
    let mut id = list.head;
    while !id.is_null() {
        let node = unsafe { map.get_unchecked_mut(id) };
        let nid = id;
        id = node.next;
        node.prev = K::null();
        if parent.layer >= node.layer {
            node.parent = parent_id;
            node.parent_child = N;
            node.next = parent.nodes.head;
            parent.nodes.push(nid);
            continue;
        }
        id = node.next;
        let i = H::get_child(&point, &node.value.0);
        match parent.childs[i] {
            ChildNode::Ab(ref mut list) => {
                node.parent = parent_id;
                node.parent_child = i;
                node.next = list.head;
                list.push(nid);
                if list.len > adjust && parent.layer < deep {
                    set_dirty(&mut parent.dirty, i, parent.layer, parent_id);
                }
                continue;
            }
            _ => panic!("invalid state"),
        }
    }
    fix_prev(map, parent.nodes.head);
    for i in 0..N {
        match parent.childs[i] {
            ChildNode::Ab(ref list) => fix_prev(map, list.head),
            _ => (), // panic
        }
    }
    parent.dirty
}
// 修复prev
#[inline]
fn fix_prev<K: Key, Aabb, T>(map: &mut SecondaryMap<K, AbNode<K, Aabb, T>>, mut head: K) {
    if head.is_null() {
        return;
    }
    let node = unsafe { map.get_unchecked(head) };
    let mut next = node.next;
    while !next.is_null() {
        let node = unsafe { map.get_unchecked_mut(next) };
        node.prev = head;
        head = next;
        next = node.next;
    }
}

// 查询空间内及相交的ab节点
fn query<K: Key, H: Helper<N>, T, A, B, const N: usize>(
    branch_slab: &SlotMap<K, BranchNode<K, H, N>>,
    ab_map: &SecondaryMap<K, AbNode<K, H::Aabb, T>>,
    branch_id: K,
    branch_arg: &A,
    branch_func: fn(arg: &A, aabb: &H::Aabb) -> bool,
    ab_arg: &mut B,
    ab_func: fn(arg: &mut B, id: K, aabb: &H::Aabb, bind: &T),
) {
    let node = unsafe { branch_slab.get_unchecked(branch_id) };
    let mut id = node.nodes.head;
    while !id.is_null() {
        let ab = unsafe { ab_map.get_unchecked(id) };
        ab_func(ab_arg, id, &ab.value.0, &ab.value.1);
        id = ab.next;
    }
    let childs = H::make_childs(&node.aabb, &node.loose);
    let mut i = 0;
    for ab in childs {
        match node.childs[i] {
            ChildNode::Branch(branch, ref num) if *num > 0 => {
                if branch_func(branch_arg, &ab) {
                    query(
                        branch_slab,
                        ab_map,
                        branch,
                        branch_arg,
                        branch_func,
                        ab_arg,
                        ab_func,
                    );
                }
            }
            ChildNode::Ab(ref list) if !list.head.is_null() => {
                if branch_func(branch_arg, &ab) {
                    let mut id = list.head;
                    loop {
                        let ab = unsafe { ab_map.get_unchecked(id) };
                        ab_func(ab_arg, id, &ab.value.0, &ab.value.1);
                        id = ab.next;
                        if id.is_null() {
                            break;
                        }
                    }
                }
            }
            _ => (),
        }
        i += 1;
    }
}

// 和指定的列表进行碰撞
// fn collision_list<H: Helper, T, A>(
//     map: &VecMap<AbNode<S, T>>,
//     id: usize,
//     aabb: &H::AABB,
//     bind: &T,
//     arg: &mut A,
//     func: fn(
//         arg: &mut A,
//         a_id: usize,
//         a_aabb: &H::AABB,
//         a_bind: &T,
//         b_id: usize,
//         b_aabb: &H::AABB,
//         b_bind: &T,
//     ) -> bool,
//     mut head: usize,
// ) {
//     while head > 0 {
//         let b = unsafe { map.get_unchecked(head) };
//         func(arg, id, aabb, bind, head, &b.aabb, &b.value.1);
//         head = b.next;
//     }
// }
