//! 高性能的松散叉树
//！采用二进制掩码 表达xyz的大小， child&1 == 0 表示x为小，否则为大。
//！采用Slab，内部用偏移量来分配八叉节点。这样内存连续，八叉树本身可以快速拷贝。

pub mod oct_helper;
pub mod quad_helper;
pub mod tree;
pub mod tilemap;

pub use tree::*;
pub use oct_helper::*;
pub use quad_helper::*;
