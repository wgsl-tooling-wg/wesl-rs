mod builtin;
mod conv;
mod display;
mod error;
#[allow(clippy::module_inception)]
mod instance;
mod mem;
mod ops;
mod ty;

pub use builtin::*;
pub use conv::*;
pub use error::*;
pub use instance::*;
pub use ty::*;

use derive_more::Display;
use std::{collections::HashMap, rc::Rc};

#[derive(Debug)]
struct ScopeInner<T> {
    local: HashMap<String, T>,
    parent: Option<Rc<ScopeInner<T>>>,
    transparent: bool,
}

#[derive(Debug)]
pub struct Scope<T> {
    inner: Rc<ScopeInner<T>>,
}

impl<T> Default for Scope<T> {
    fn default() -> Self {
        Self {
            inner: Rc::new(ScopeInner {
                local: Default::default(),
                parent: Default::default(),
                transparent: false,
            }),
        }
    }
}

impl<T> ScopeInner<T> {
    pub fn get(&self, name: &str) -> Option<&T> {
        self.local
            .get(name)
            .or_else(|| self.parent.as_ref().and_then(|parent| parent.get(name)))
    }
    pub fn contains(&self, name: &str) -> bool {
        self.local.contains_key(name)
            || self
                .parent
                .as_ref()
                .is_some_and(|parent| parent.contains(name))
    }
}

impl<T> Scope<T> {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(ScopeInner {
                local: Default::default(),
                parent: None,
                transparent: false,
            }),
        }
    }
    pub fn is_root(&self) -> bool {
        self.inner.parent.is_none()
    }
    /// variable in a 'transparent' have the same scope as the parent scope.
    /// this is useful for 'for' loops and function calls which have the same
    /// end-of-scope for initializer and formal parameters as the body.
    ///
    /// see <https://github.com/gpuweb/gpuweb/issues/5024>
    pub fn make_transparent(&mut self) {
        Rc::get_mut(&mut self.inner)
            .expect("cannot edit a parent scope")
            .transparent = true;
    }
    pub fn push(&mut self) {
        self.inner = Rc::new(ScopeInner {
            local: Default::default(),
            parent: Some(self.inner.clone()),
            transparent: false,
        });
    }
    pub fn pop(&mut self) {
        self.inner = self.inner.parent.clone().expect("failed to pop scope");
    }
    pub fn add(&mut self, name: String, value: T) -> bool {
        if self.local_contains(&name) {
            false
        } else {
            Rc::get_mut(&mut self.inner)
                .expect("cannot edit a parent scope")
                .local
                .insert(name, value);
            true
        }
    }
    pub fn local_get_mut(&mut self, name: &str) -> Option<&mut T> {
        Rc::get_mut(&mut self.inner)
            .expect("cannot edit a parent scope")
            .local
            .get_mut(name)
    }
    pub fn get(&self, name: &str) -> Option<&T> {
        self.inner.get(name)
    }
    pub fn local_contains(&self, name: &str) -> bool {
        self.inner.local.contains_key(name)
            || self.inner.transparent
                && self
                    .inner
                    .parent
                    .as_ref()
                    .expect("transparent scope must have a parent")
                    .local
                    .contains_key(name)
    }
    pub fn contains(&self, name: &str) -> bool {
        self.inner.contains(name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Display)]
pub enum ScopeKind {
    #[display("module")]
    Module,
    #[display("function")]
    Function,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalStage {
    /// Shader module creation
    Const,
    /// Pipeline creation
    Override,
    /// Shader execution
    Exec,
}

#[derive(Clone, Copy, Debug)]
pub enum ResourceKind {
    UniformBuffer,
    StorageBuffer,
    Texture,
    Sampler,
}
