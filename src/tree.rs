//! 高性能的松散叉树
//! 采用二进制掩码 表达xyz的大小， child&1 == 0 表示x为小，否则为大。
//! 采用Slab，内部用偏移量来分配八叉空间。这样内存连续，八叉树本身可以快速拷贝。
//! 要求插入AABB节点时的id， 应该是可以用在数组索引上的。
//! 分裂和收缩：
//!     ChildNode的Branch(BranchKey),
//!     如果BranchKey对应八叉空间下的节点总数量小于收缩阈值，则可以收缩成ChildNode的Ab(List)
//!     ChildNode的Ab(List),
//!     如果List中节点的数量大于分裂阈值，则也可以分裂成Branch(BranchKey)
//!     收缩阈值一般为4，分裂阈值一般为8。 在这个结构下，不会出现反复分裂和收缩。
//!     如果一组节点重叠，并且超过6，则会导致会分裂到节点所能到达的最低的层，应用方应该尽量避免这种情况
//! 更新aabb：
//!     节点只会在3个位置：
//!     1. 如果超出或相交边界，则在tree.outer上
//!        这种情况下，node.parent为null
//!     2. 如果节点大小下不去，则只能在本层活动，则在BranchNode的nodes
//!         这种情况下，node.layer==parent.layer. node.parent_child==N
//!     3. 其余的节点都在ChildNode的Ab(List)中
//!         node.layer<parent.layer. node.parent_child<N
//!     更新节点就是在这3个位置上挪动

use std::mem;

use pi_link_list::{LinkList, Node};
use pi_null::Null;
use pi_slotmap::{new_key_type, Key, SecondaryMap, SlotMap};

new_key_type! {
    pub struct BranchKey;
}

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
    fn get_child(point: &Self::Point, aabb: &Self::Aabb) -> u8;
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
        child_index: u8,
    ) -> (Self::Aabb, Self::Vector);
}

const DEEP_MAX: usize = 16;
const ADJUST_MIN: usize = 4;
const ADJUST_MAX: usize = 8;
const AUTO_COLLECT: usize = 1024;

type List<K, H, T, const N: usize> = LinkList<
    K,
    AbNode<<H as Helper<N>>::Aabb, T>,
    SecondaryMap<K, Node<K, AbNode<<H as Helper<N>>::Aabb, T>>>,
>;
///
/// 叉树结构体
///
/// ### 对`N`的约束
///
/// + 浮点数算术运算，可拷贝，可偏序比较；
/// + 实际使用的时候就是浮点数字类型，比如：f32/f64；
///
pub struct Tree<K: Key, H: Helper<N>, T, const N: usize> {
    pub slab: SlotMap<BranchKey, BranchNode<K, H, T, N>>, //所有分支节点（分支节点中包含该层ab节点列表）
    pub ab_map: SecondaryMap<K, Node<K, AbNode<H::Aabb, T>>>, //所有存储ab碰撞单位的节点
    max_loose: H::Vector,                                 //最大松散值，第一层的松散大小
    min_loose: H::Vector,                                 //最小松散值
    root_key: BranchKey,
    pub outer: List<K, H, T, N>, // 和根空间不包含（相交或在外）的ab节点列表，及节点数量。 该AbNode的parent为Null
    pub dirty: (Vec<Vec<BranchKey>>, DirtyState), // 脏的BranchNode节点, 及脏节点状态
    adjust: (usize, usize), //小于min，节点收缩; 大于max，节点分化。默认(4, 8)
    loose_layer: usize,     // 最小松散值所在的深度
    deep: usize,        // 最大深度, 推荐12-16, 最小松散值设置的好，不设置最大深度也是可以的
    auto_collect: usize, // 自动整理的阈值，默认为1024
}

impl<K: Key, H: Helper<N>, T, const N: usize> Tree<K, H, T, N> {
    ///构建树
    ///
    /// 需传入根空间（即全场景）AB碰撞范围；N维实际距离所表示的最大及最小松散参数；叉树收缩及分裂的阈值；叉树的深度限制
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
        let deep = if deep > DEEP_MAX || deep == 0 {
            DEEP_MAX
        } else {
            deep
        };
        let mut branch_slab: SlotMap<BranchKey, BranchNode<K, H, T, N>> = SlotMap::with_key();
        let mut d = H::aabb_extents(&root);
        // 根据最大 最小 松散值 计算出最小松散值所在的最大的层
        let loose_layer = H::calc_layer(&max_loose, &min_loose);
        let deep = H::get_deap(&mut d, loose_layer, &max_loose, deep, &min_loose);

        let root = branch_slab.insert(BranchNode::new(
            root,
            max_loose.clone(),
            0,
            BranchKey::null(),
            0,
        ));
        return Tree {
            slab: branch_slab,
            ab_map: SecondaryMap::default(),
            max_loose,
            min_loose,
            adjust: (adjust_min, adjust_max),
            loose_layer,
            deep,
            root_key: root,
            outer: LinkList::new(),
            dirty: (
                Vec::new(),
                DirtyState {
                    dirty_count: 0,
                    min_layer: usize::max_value(),
                    max_layer: 0,
                },
            ),
            auto_collect: AUTO_COLLECT,
        };
    }

    // /// 获得叉树总的占有内存的字节数
    // pub fn mem_size(&self) -> usize {
    //     self.slab.mem_size()
    //         + self.ab_map.mem_size()
    //         + self.outer.len() * std::mem::size_of::<usize>()

    // }
    /// 获得自动整理的次数
    pub fn get_auto_collect(&self) -> usize {
        self.auto_collect
    }
    /// 设置自动整理的次数
    pub fn set_auto_collect(&mut self, auto_collect: usize) {
        self.auto_collect = auto_collect;
    }
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
    pub fn add(&mut self, id: K, aabb: H::Aabb, bind: T) -> bool {
        if self.ab_map.contains_key(id) {
            return false;
        }
        let layer = self.get_layer(&aabb);
        self.ab_map.insert(
            id,
            Node::new(AbNode::new(aabb.clone(), bind, layer, N as u8)),
        );
        let root = unsafe { self.slab.get_unchecked_mut(self.root_key) };
        if H::aabb_contains(&root.aabb, &aabb) {
            // root的ab内
            self.down(self.root_key, &aabb, layer, id);
        } else {
            // 和根空间相交或在其外的ab节点, 该AbNode的parent为0
            self.outer.link_before(id, K::null(), &mut self.ab_map);
        }
        true
    }

    /// ab节点下降
    /// ChildNode的Branch(BranchKey, usize), 记录了该八叉空间下的节点总数量
    /// 如果小于阈值，则可以转化成ChildNode的Ab(List)
    /// ChildNode的Ab(List)如果大于阈值，则也可以转化成Branch(BranchKey, usize)
    fn down(&mut self, branch_id: BranchKey, aabb: &H::Aabb, layer: usize, id: K) {
        let parent = unsafe { self.slab.get_unchecked_mut(branch_id) };
        let child = if parent.layer as usize >= layer {
            parent.nodes.link_before(id, K::null(), &mut self.ab_map);
            N as u8
        } else {
            let i = H::get_child(&H::get_max_half_loose(&parent.aabb, &parent.loose), aabb);
            match parent.childs[i as usize] {
                ChildNode::Branch(branch) => {
                    return self.down(branch, aabb, layer, id);
                }
                ChildNode::Ab(ref mut list) => {
                    list.link_before(id, K::null(), &mut self.ab_map);
                    if list.len() >= self.adjust.1 && parent.layer < self.deep {
                        set_dirty(&mut parent.dirty, parent.layer, branch_id, &mut self.dirty);
                    }
                }
            }
            i
        };
        let node = unsafe { self.ab_map.get_unchecked_mut(id) };
        node.parent = branch_id;
        node.parent_child = child;
        if self.dirty.1.dirty_count >= self.auto_collect {
            self.collect();
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
        &self.ab_map.get_unchecked(id).value
    }

    /// 获取指定id的可写绑定
    pub unsafe fn get_mut(&mut self, id: K) -> Option<&mut T> {
        match self.ab_map.get_mut(id) {
            Some(n) => Some(&mut n.value.1),
            _ => None,
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
    pub fn update(&mut self, id: K, aabb: H::Aabb) -> bool {
        let layer = self.get_layer(&aabb);
        if let Some(node) = self.ab_map.get_mut(id) {
            node.layer = layer;
            node.value.0 = aabb.clone();
            let old_p = node.parent;
            let old_c = node.parent_child;
            self.update1(id, layer, old_p, old_c, &aabb);
            true
        } else {
            false
        }
    }

    /// 更新aabb
    /// 节点只会在3个位置：
    ///     1. 如果超出或相交边界，则在outer上
    ///         这种情况下，node.parent为null
    ///     2. 如果节点大小下不去，则只能在本层活动，则在BranchNode的nodes
    ///         这种情况下，node.layer==parent.layer. node.parent_child==N
    ///     3. 其余的节点都在ChildNode的Ab(List)中
    ///         node.layer<parent.layer. node.parent_child<N
    /// 更新节点就是在这3个位置上挪动
    fn update1(&mut self, id: K, layer: usize, old_p: BranchKey, old_c: u8, aabb: &H::Aabb) {
        if old_p.is_null() {
            // 边界外物体更新
            let root = unsafe { self.slab.get_unchecked_mut(self.root_key) };
            if H::aabb_contains(&root.aabb, aabb) {
                self.outer.unlink(id, &mut self.ab_map);
                self.down(self.root_key, aabb, layer, id);
            } else {
                // 不包含，表示还在outer上
            }
            return;
        }
        let mut parent = unsafe { self.slab.get_unchecked_mut(old_p) };
        if layer > parent.layer {
            // ab节点能在当前branch空间的容纳范围
            if H::aabb_contains(&parent.aabb, aabb) {
                // 获得新位置
                let child = H::get_child(&H::get_max_half_loose(&parent.aabb, &parent.loose), aabb);
                if old_c == child {
                    return;
                }
                Self::remove1(&mut self.ab_map, id, old_c, parent);
                // 移动到兄弟节点
                match parent.childs[child as usize] {
                    ChildNode::Branch(branch) => {
                        self.down(branch, aabb, layer, id);
                    }
                    ChildNode::Ab(ref mut list) => {
                        Self::add1(&mut self.ab_map, list, id, old_p, child);
                        if list.len() >= self.adjust.1 && layer < self.deep {
                            set_dirty(&mut parent.dirty, parent.layer, old_p, &mut self.dirty);
                        }
                    }
                }
                return;
            }
            // 需要向上
        } else if layer == parent.layer {
            // 还是继续在本层
            if H::aabb_contains(&parent.aabb, aabb) {
                // 还是继续在本层本空间内
                if (old_c as usize) == N {
                    return;
                }
                // old_c < N 表示是从本空间的ChildNode的Ab(List)移动上来的
                Self::remove1(&mut self.ab_map, id, old_c, parent);
                Self::add1(&mut self.ab_map, &mut parent.nodes, id, old_p, N as u8);
                // Ab(List)变少，但本层空间的节点数量不变，是不需要设脏的
                return;
            }
            // 在当前空间外
        } else {
            // 比当前空间大
        };
        // 从当前空间移走
        Self::remove1(&mut self.ab_map, id, old_c, parent);
        // 如果本空间小于收缩阈值，设置本空间脏标记
        if parent.is_need_merge(self.adjust.0) {
            set_dirty(&mut parent.dirty, parent.layer, old_p, &mut self.dirty);
        }
        // 向上移动
        let mut p = parent.parent;
        while !p.is_null() {
            parent = unsafe { self.slab.get_unchecked_mut(p) };
            if parent.layer <= layer && H::aabb_contains(&parent.aabb, aabb) {
                return self.down(p, aabb, layer, id);
            }
            p = parent.parent;
        }
        // 根空间不包含该节点，相交或超出，放到outer上
        Self::add1(
            &mut self.ab_map,
            &mut self.outer,
            id,
            BranchKey::null(),
            N as u8,
        );
    }
    /// 从旧的Parent中移除
    fn remove1(
        ab_map: &mut SecondaryMap<K, Node<K, AbNode<H::Aabb, T>>>,
        id: K,
        old_c: u8,
        parent: &mut BranchNode<K, H, T, N>,
    ) {
        if (old_c as usize) < N {
            match parent.childs[old_c as usize] {
                ChildNode::Ab(ref mut list) => list.unlink(id, ab_map),
                _ => panic!("invalid state"),
            }
        } else {
            parent.nodes.unlink(id, ab_map);
        }
    }
    /// 设置节点新的Parent
    fn add1(
        ab_map: &mut SecondaryMap<K, Node<K, AbNode<H::Aabb, T>>>,
        list: &mut List<K, H, T, N>,
        id: K,
        parent: BranchKey,
        parent_child: u8,
    ) {
        let node = unsafe { ab_map.get_unchecked_mut(id) };
        node.parent = parent;
        node.parent_child = parent_child;
        list.link_before(id, K::null(), ab_map);
    }
    /// 移动指定id的aabb，性能比update要略好
    pub fn shift(&mut self, id: K, distance: H::Vector) -> bool {
        if let Some(node) = self.ab_map.get_mut(id) {
            let aabb = H::aabb_shift(&node.value.0, &distance);
            let layer = node.layer;
            node.value.0 = aabb.clone();
            let old_p = node.parent;
            let old_c = node.parent_child;
            self.update1(id, layer, old_p, old_c, &aabb);
            true
        } else {
            false
        }
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
        let (parent, parent_child) = match self.ab_map.get(id) {
            Some(n) => (n.parent, n.parent_child),
            _ => return None,
        };
        if !parent.is_null() {
            let branch = unsafe { self.slab.get_unchecked_mut(parent) };
            Self::remove1(&mut self.ab_map, id, parent_child, branch);
            // 如果本空间小于收缩阈值，设置本空间脏标记
            if branch.is_need_merge(self.adjust.0) {
                set_dirty(&mut branch.dirty, branch.layer, parent, &mut self.dirty);
            }
        } else {
            // 表示在outer上
            self.outer.unlink(id, &mut self.ab_map);
        }
        Some(self.ab_map.remove(id).unwrap().take().value)
    }

    /// 整理方法，只有整理方法才会创建或销毁BranchNode
    pub fn collect(&mut self) {
        let state = mem::replace(&mut self.dirty.1, DirtyState::new());
        if state.dirty_count == 0 {
            return;
        }
        for i in state.min_layer..state.max_layer {
            let vec = unsafe { self.dirty.0.get_unchecked_mut(i) };
            let c = vec.len();
            if c == 0 {
                continue;
            }
            for j in 0..c {
                let branch_id = unsafe { vec.get_unchecked(j) };
                Self::collect1(
                    &mut self.slab,
                    &mut self.ab_map,
                    &self.adjust,
                    self.deep,
                    *branch_id,
                    self.loose_layer,
                    &self.min_loose,
                );
            }
            vec.clear();
        }
    }

    /// 整理方法，只有整理方法才会创建或销毁BranchNode
    fn collect1(
        slab: &mut SlotMap<BranchKey, BranchNode<K, H, T, N>>,
        ab_map: &mut SecondaryMap<K, Node<K, AbNode<H::Aabb, T>>>,
        adjust: &(usize, usize),
        deep: usize,
        branch_id: BranchKey,
        loose_layer: usize,
        min_loose: &H::Vector,
    ) {
        let parent = match slab.get_mut(branch_id) {
            Some(branch) => branch,
            _ => return,
        };
        let dirty = mem::replace(&mut parent.dirty, false);
        if !dirty {
            return;
        }
        let parent_id = parent.parent;
        // 判断是否收缩
        if (!parent_id.is_null()) && parent.is_need_merge(adjust.0) {
            let child = parent.parent_child;
            let list = Self::merge_branch(ab_map, parent, LinkList::new());
            slab.remove(branch_id);
            Self::shrink(slab, ab_map, adjust.0, parent_id, child, branch_id, list);
            return;
        }
        let (need, lists) = parent.need_split_list(adjust.1);
        if need {
            let aabb = parent.aabb.clone();
            let loose = parent.loose.clone();
            let layer = parent.layer;
            Self::split(
                slab,
                ab_map,
                adjust.1,
                deep,
                lists,
                &aabb,
                &loose,
                layer,
                branch_id,
                loose_layer,
                min_loose,
            );
        }
    }
    // 合并子空间的所有列表
    fn merge_branch(
        ab_map: &mut SecondaryMap<K, Node<K, AbNode<H::Aabb, T>>>,
        branch: &mut BranchNode<K, H, T, N>,
        mut list: List<K, H, T, N>,
    ) -> List<K, H, T, N> {
        list.append(&mut branch.nodes, ab_map);
        for n in &mut branch.childs {
            match n {
                ChildNode::Ab(other) => list.append(other, ab_map),
                _ => (),
            }
        }
        list
    }

    /// 收缩BranchNode
    fn shrink(
        slab: &mut SlotMap<BranchKey, BranchNode<K, H, T, N>>,
        ab_map: &mut SecondaryMap<K, Node<K, AbNode<H::Aabb, T>>>,
        adjust: usize,
        branch_id: BranchKey,
        parent_child: u8,
        child_id: BranchKey,
        list: List<K, H, T, N>,
    ) {
        let branch = unsafe { slab.get_unchecked_mut(branch_id) };
        // 判断是否继续收缩
        if (!branch.parent.is_null()) && branch.is_need_merge_with_child(adjust, child_id, list.len()) {
            let parent_id = branch.parent;
            let child = branch.parent_child;
            let list = Self::merge_branch(ab_map, branch, list);
            slab.remove(branch_id);
            Self::shrink(slab, ab_map, adjust, parent_id, child, branch_id, list);
        } else {
            for (_, node) in list.iter_mut(ab_map) {
                node.parent = branch_id;
                node.parent_child = parent_child;
            };
            branch.childs[parent_child as usize] = ChildNode::Ab(list);
        }
    }
    // 对列表进行分裂
    #[inline]
    fn split(
        slab: &mut SlotMap<BranchKey, BranchNode<K, H, T, N>>,
        ab_map: &mut SecondaryMap<K, Node<K, AbNode<H::Aabb, T>>>,
        adjust: usize,
        deep: usize,
        lists: [List<K, H, T, N>; N],
        parent_aabb: &H::Aabb,
        parent_loose: &H::Vector,
        parent_layer: usize,
        parent_id: BranchKey,
        loose_layer: usize,
        min_loose: &H::Vector,
    ) {
        let mut branchs = [BranchKey::null(); N];
        for (i, list) in lists.into_iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let branch = BranchNode::create(
                parent_aabb,
                parent_loose,
                parent_layer,
                parent_id,
                loose_layer,
                min_loose,
                i as u8,
            );
            let branch_id = slab.insert(branch);
            Self::split_down(
                slab,
                ab_map,
                adjust,
                deep,
                list,
                branch_id,
                loose_layer,
                min_loose,
            );
            branchs[i] = branch_id;
        }
        let parent = unsafe { slab.get_unchecked_mut(parent_id) };
        for (i, child_id) in branchs.into_iter().enumerate() {
            if !child_id.is_null() {
                parent.childs[i] = ChildNode::Branch(child_id);
            }
        }
    }
    // 将ab节点列表放到分裂出来的八叉空间上
    fn split_down(
        slab: &mut SlotMap<BranchKey, BranchNode<K, H, T, N>>,
        ab_map: &mut SecondaryMap<K, Node<K, AbNode<H::Aabb, T>>>,
        adjust: usize,
        deep: usize,
        list: List<K, H, T, N>,
        parent_id: BranchKey,
        loose_layer: usize,
        min_loose: &H::Vector,
    ) {
        let parent = unsafe { slab.get_unchecked_mut(parent_id) };
        let point = H::get_max_half_loose(&parent.aabb, &parent.loose);
        let mut drain = list.drain();
        let mut id = drain.pop_front(ab_map);
        while !id.is_null() {
            let node = unsafe { ab_map.get_unchecked_mut(id) };
            if parent.layer >= node.layer {
                node.parent = parent_id;
                node.parent_child = N as u8;
                parent.nodes.link_before(id, K::null(), ab_map);
            } else {
                let i = H::get_child(&point, &node.value.0);
                match parent.childs[i as usize] {
                    ChildNode::Ab(ref mut ab) => {
                        node.parent = parent_id;
                        node.parent_child = i;
                        ab.link_before(id, K::null(), ab_map);
                    }
                    _ => panic!("invalid state"),
                }
            }
            id = drain.pop_front(ab_map);
        }
        if parent.layer >= deep {
            return;
        }
        let (need, lists) = parent.need_split_list(adjust);
        if need {
            let aabb: <H as Helper<N>>::Aabb = parent.aabb.clone();
            let loose = parent.loose.clone();
            let layer = parent.layer;
            Self::split(
                slab,
                ab_map,
                adjust,
                deep,
                lists,
                &aabb,
                &loose,
                layer,
                parent_id,
                loose_layer,
                min_loose,
            );
        }
    }

    /// 查询空间内及相交的ab节点
    pub fn query<A, B>(
        &self,
        branch_arg: &A,
        branch_func: fn(arg: &A, aabb: &H::Aabb) -> bool,
        ab_arg: &mut B,
        ab_func: fn(arg: &mut B, id: K, aabb: &H::Aabb, bind: &T),
    ) {
        self.query_outer(ab_arg, ab_func);
        self.query1(self.root_key, branch_arg, branch_func, ab_arg, ab_func)
    }

    // 查询空间内及相交的ab节点
    fn query1<A, B>(
        &self,
        branch_id: BranchKey,
        branch_arg: &A,
        branch_func: fn(arg: &A, aabb: &H::Aabb) -> bool,
        ab_arg: &mut B,
        ab_func: fn(arg: &mut B, id: K, aabb: &H::Aabb, bind: &T),
    ) {
        let node = unsafe { self.slab.get_unchecked(branch_id) };
        for (id, ab) in node.nodes.iter(&self.ab_map) {
            ab_func(ab_arg, id, &ab.value.0, &ab.value.1);
        }
        let childs = H::make_childs(&node.aabb, &node.loose);
        for (i, ab) in childs.iter().enumerate() {
            match node.childs[i] {
                ChildNode::Branch(branch) => {
                    if branch_func(branch_arg, &ab) {
                        self.query1(branch, branch_arg, branch_func, ab_arg, ab_func);
                    }
                }
                ChildNode::Ab(ref list) if !list.is_empty() => {
                    if branch_func(branch_arg, &ab) {
                        for (id, ab) in list.iter(&self.ab_map) {
                            ab_func(ab_arg, id, &ab.value.0, &ab.value.1);
                        }
                    }
                }
                _ => (),
            }
        }
    }
    /// 查询空间外的ab节点
    pub fn query_outer<B>(
        &self,
        arg: &mut B,
        func: fn(arg: &mut B, id: K, aabb: &H::Aabb, bind: &T),
    ) {
        for (id, ab) in self.outer.iter(&self.ab_map) {
            func(arg, id, &ab.value.0, &ab.value.1);
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

#[derive(Clone)]
pub struct BranchNode<K: Key, H: Helper<N>, T, const N: usize> {
    aabb: H::Aabb,                      // 包围盒
    loose: H::Vector,                   // 本层的松散值
    layer: usize,                       // 表示第几层， 根据aabb大小，决定最低为第几层
    parent: BranchKey,                  // 父八叉空间
    childs: [ChildNode<K, H, T, N>; N], // 子八叉空间
    nodes: List<K, H, T, N>,            // 匹配本层大小的ab节点列表，及节点数量
    parent_child: u8,                   // 对应父八叉空间childs的位置
    dirty: bool, // 脏标记. 添加了节点，并且某个子八叉空间(AbNode)的数量超过分裂阈值，可能分裂。删除了节点，并且自己及其下ab节点的数量小于收缩阈值，可能收缩
}
impl<K: Key, H: Helper<N>, T, const N: usize> BranchNode<K, H, T, N> {
    #[inline]
    pub fn new(
        aabb: H::Aabb,
        loose: H::Vector,
        layer: usize,
        parent: BranchKey,
        child: u8,
    ) -> Self {
        let childs = [0; N].map(|_| ChildNode::Ab(Default::default()));
        BranchNode {
            aabb,
            loose,
            layer,
            parent,
            childs,
            nodes: LinkList::new(),
            parent_child: child,
            dirty: false,
        }
    }
    // 创建指定的子节点
    fn create(
        aabb: &H::Aabb,
        loose: &H::Vector,
        layer: usize,
        parent_id: BranchKey,
        loose_layer: usize,
        min_loose: &H::Vector,
        child: u8,
    ) -> Self {
        let (ab, loose) = H::create_child(aabb, loose, layer, loose_layer, min_loose, child);
        BranchNode::new(ab, loose, layer + 1, parent_id, child)
    }
    // 是否需要合并
    pub fn is_need_merge(&self, adjust_min: usize) -> bool {
        if self.parent.is_null() {
            return false;
        }
        let mut len = self.nodes.len();
        for n in &self.childs {
            match n {
                ChildNode::Branch(_) => return false,
                ChildNode::Ab(list) => len += list.len(),
            }
        }
        len <= adjust_min
    }
    // 是否需要合并
    pub fn is_need_merge_with_child(
        &self,
        adjust_min: usize,
        child: BranchKey,
        child_node_len: usize,
    ) -> bool {
        let mut len = self.nodes.len();
        for n in &self.childs {
            match n {
                ChildNode::Branch(b) => {
                    if b != &child {
                        return false;
                    }
                    len += child_node_len;
                }
                ChildNode::Ab(list) => len += list.len(),
            }
        }
        len <= adjust_min
    }
    // 需要劈分的列表
    pub fn need_split_list(&mut self, adjust_max: usize) -> (bool, [List<K, H, T, N>; N]) {
        let mut need = false;
        let mut childs = [0; N].map(|_| Default::default());
        for (i, n) in self.childs.iter_mut().enumerate() {
            match n {
                ChildNode::Ab(list) if list.len() >= adjust_max => {
                    mem::swap(list, &mut childs[i]);
                    need = true;
                }
                _ => (),
            }
        }
        (need, childs)
    }
}
#[derive(Clone)]
enum ChildNode<K: Key, H: Helper<N>, T, const N: usize> {
    Branch(BranchKey),    // 对应的BranchNode, 及其下ab节点的数量
    Ab(List<K, H, T, N>), // ab节点列表，及节点数量
}

#[derive(Debug, Clone)]
pub struct AbNode<Aabb, T> {
    value: (Aabb, T),  // 包围盒
    parent: BranchKey, // 父八叉空间
    layer: usize,      // 表示第几层， 根据aabb大小，决定最低为第几层
    parent_child: u8,  // 父八叉空间所在的子八叉空间， 8表示不在子八叉空间上
}
impl<Aabb, T> AbNode<Aabb, T> {
    pub fn new(aabb: Aabb, bind: T, layer: usize, n: u8) -> Self {
        AbNode {
            value: (aabb, bind),
            layer: layer,
            parent: BranchKey::null(),
            parent_child: n,
        }
    }
}

#[derive(Debug)]
pub struct DirtyState {
    dirty_count: usize,
    min_layer: usize,
    max_layer: usize,
}
impl DirtyState {
    fn new() -> Self {
        DirtyState {
            dirty_count: 0,
            min_layer: usize::max_value(),
            max_layer: 0,
        }
    }
}

#[inline]
fn set_dirty(
    dirty: &mut bool,
    layer: usize,
    rid: BranchKey,
    dirty_list: &mut (Vec<Vec<BranchKey>>, DirtyState),
) {
    dirty_list.1.dirty_count += 1;
    if !*dirty {
        // 该八叉空间首次脏，则放入脏列表
        set_tree_dirty(dirty_list, layer, rid);
    }
    *dirty = true;
}
// 设置脏标记
#[inline]
fn set_tree_dirty(dirty: &mut (Vec<Vec<BranchKey>>, DirtyState), layer: usize, rid: BranchKey) {
    if dirty.1.min_layer > layer {
        dirty.1.min_layer = layer;
    }
    if dirty.1.max_layer <= layer {
        dirty.1.max_layer = layer + 1;
    }
    if dirty.0.len() <= layer as usize {
        for _ in dirty.0.len()..layer as usize + 1 {
            dirty.0.push(Vec::new())
        }
    }
    let vec = unsafe { dirty.0.get_unchecked_mut(layer as usize) };
    vec.push(rid);
}
