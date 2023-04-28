use core::hash::Hash;
use pi_slotmap::{Key, KeyData};
use std::{cmp::Ordering, hash::Hasher};

#[derive(Debug, Clone, Copy, Default)]
pub struct ID(pub f64);

impl Ord for ID {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = unsafe { std::mem::transmute::<f64, u64>(self.0) };
        let b = unsafe { std::mem::transmute::<f64, u64>(other.0) };
        a.cmp(&b)
    }
}

impl PartialOrd for ID {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let a = unsafe { std::mem::transmute::<f64, u64>(self.0) };
        let b = unsafe { std::mem::transmute::<f64, u64>(other.0) };
        a.partial_cmp(&b)
    }
}

impl Hash for ID {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let a = unsafe { std::mem::transmute::<f64, u64>(self.0) };
        a.hash(state)
    }
}

impl Eq for ID {}

impl PartialEq for ID {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl From<KeyData> for ID {
    fn from(data: KeyData) -> Self {
        ID(unsafe { std::mem::transmute(data.as_ffi()) })
    }
}

unsafe impl Key for ID {
    fn data(&self) -> KeyData {
        KeyData::from_ffi(unsafe { std::mem::transmute(self.0) })
    }

    fn null() -> Self {
        ID(f64::MAX)
    }

    fn is_null(&self) -> bool {
        self.0 == f64::MAX
    }
}
