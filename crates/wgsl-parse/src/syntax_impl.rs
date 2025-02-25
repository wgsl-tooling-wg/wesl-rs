use itertools::Itertools;

use crate::span::Spanned;

use super::syntax::*;

impl TranslationUnit {
    /// New empty [`TranslationUnit`]
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove all [`GlobalDeclaration::Void`] and [`Statement::Void`]
    pub fn remove_voids(&mut self) {
        self.global_declarations.retain_mut(|decl| match decl {
            GlobalDeclaration::Void => false,
            _ => {
                decl.remove_voids();
                true
            }
        })
    }
}

#[cfg(feature = "imports")]
impl ModulePath {
    /// Create a new module path from components.
    ///
    /// Precondition: the path components must be valid WGSL identifiers.
    pub fn new(origin: PathOrigin, components: Vec<String>) -> Self {
        Self { origin, components }
    }
    /// Create a new module path from a filesystem path.
    ///
    /// * Paths with a root (leading `/` on Unix) produce `package::` paths.
    /// * Relative paths (starting with `.` or `..`) produce `self::` or `super::` paths.
    /// * The file extension is ignored.
    /// * The path is canonicalized and to do so it does NOT follow symlinks.
    ///
    /// Precondition: the path components must be valid WGSL identifiers.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Self {
        use std::path::Component;
        let mut origin = PathOrigin::Package;
        let mut components = Vec::new();

        for comp in path.as_ref().with_extension("").components() {
            match comp {
                Component::Prefix(_) => {}
                Component::RootDir => origin = PathOrigin::Absolute,
                Component::CurDir => {
                    if components.is_empty() && origin.is_package() {
                        origin = PathOrigin::Relative(0);
                    }
                }
                Component::ParentDir => {
                    if components.is_empty() {
                        if let PathOrigin::Relative(n) = &mut origin {
                            *n += 1;
                        } else {
                            origin = PathOrigin::Relative(1)
                        }
                    } else {
                        components.pop();
                    }
                }
                Component::Normal(comp) => components.push(comp.to_string_lossy().to_string()),
            }
        }

        Self { origin, components }
    }
    /// Append a component to the path.
    ///
    /// Precondition: the `item` must be a valid WGSL identifier.
    pub fn push(&mut self, item: &str) {
        self.components.push(item.to_string());
    }
    /// Get the first component of the module path.
    pub fn first(&self) -> Option<&str> {
        self.components.first().map(String::as_str)
    }
    /// Get the last component of the module path.
    pub fn last(&self) -> Option<&str> {
        self.components.last().map(String::as_str)
    }
    /// Append `suffix` to the module path.
    pub fn join(mut self, suffix: impl IntoIterator<Item = String>) -> Self {
        self.components.extend(suffix);
        self
    }
    /// Append `suffix` to the module path.
    /// the suffix must be a relative module path.
    pub fn join_path(&self, path: &Self) -> Option<Self> {
        match path.origin {
            PathOrigin::Relative(n) => {
                let to_keep = self.components.len().min(n) - n;
                let components = self
                    .components
                    .iter()
                    .take(to_keep)
                    .chain(&path.components)
                    .cloned()
                    .collect_vec();
                let origin = match self.origin {
                    PathOrigin::Absolute | PathOrigin::Package => {
                        if n > self.components.len() {
                            PathOrigin::Relative(n - self.components.len())
                        } else {
                            self.origin
                        }
                    }
                    PathOrigin::Relative(m) => {
                        if n > self.components.len() {
                            PathOrigin::Relative(m + n - self.components.len())
                        } else {
                            self.origin
                        }
                    }
                };
                Some(Self { origin, components })
            }
            _ => None,
        }
    }
    pub fn starts_with(&self, prefix: &Self) -> bool {
        self.origin == prefix.origin
            && prefix.components.len() >= self.components.len()
            && prefix.components.iter().zip(&self.components).all_equal()
    }
    pub fn is_empty(&self) -> bool {
        self.origin.is_package() && self.components.is_empty()
    }
}

impl Default for ModulePath {
    /// The path that is represented as ``, i.e. a package import with no components.
    fn default() -> Self {
        Self {
            origin: PathOrigin::Package,
            components: Vec::new(),
        }
    }
}

impl<T: AsRef<std::path::Path>> From<T> for ModulePath {
    fn from(value: T) -> Self {
        ModulePath::from_path(value.as_ref())
    }
}

impl GlobalDeclaration {
    /// Remove all [`Statement::Void`]
    pub fn remove_voids(&mut self) {
        if let GlobalDeclaration::Function(decl) = self {
            decl.body.remove_voids();
        }
    }
}

impl TypeExpression {
    /// New [`TypeExpression`] with no template.
    pub fn new(ident: Ident) -> Self {
        Self {
            #[cfg(feature = "imports")]
            path: None,
            ident,
            template_args: None,
        }
    }
}

impl CompoundStatement {
    /// Remove all [`Statement::Void`]
    pub fn remove_voids(&mut self) {
        self.statements.retain_mut(|stmt| match stmt.node_mut() {
            Statement::Void => false,
            _ => {
                stmt.remove_voids();
                true
            }
        })
    }
}

impl Statement {
    /// Remove all [`Statement::Void`]
    pub fn remove_voids(&mut self) {
        match self {
            Statement::Compound(stmt) => {
                stmt.remove_voids();
            }
            Statement::If(stmt) => {
                stmt.if_clause.body.remove_voids();
                for clause in &mut stmt.else_if_clauses {
                    clause.body.remove_voids();
                }
                if let Some(clause) = &mut stmt.else_clause {
                    clause.body.remove_voids();
                }
            }
            Statement::Switch(stmt) => stmt
                .clauses
                .iter_mut()
                .for_each(|clause| clause.body.remove_voids()),
            Statement::Loop(stmt) => stmt.body.remove_voids(),
            Statement::For(stmt) => stmt.body.remove_voids(),
            Statement::While(stmt) => stmt.body.remove_voids(),
            _ => (),
        }
    }
}

impl AccessMode {
    /// Is [`Self::Read`] or [`Self::ReadWrite`]
    pub fn is_read(&self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }
    /// Is [`Self::Write`] or [`Self::ReadWrite`]
    pub fn is_write(&self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

impl From<Ident> for TypeExpression {
    fn from(name: Ident) -> Self {
        Self {
            #[cfg(feature = "imports")]
            path: None,
            ident: name,
            template_args: None,
        }
    }
}

impl From<ExpressionNode> for ReturnStatement {
    fn from(expression: ExpressionNode) -> Self {
        Self {
            #[cfg(feature = "attributes")]
            attributes: Default::default(),
            expression: Some(expression),
        }
    }
}
impl From<Expression> for ReturnStatement {
    fn from(expression: Expression) -> Self {
        Self::from(ExpressionNode::from(expression))
    }
}

impl From<FunctionCall> for FunctionCallStatement {
    fn from(call: FunctionCall) -> Self {
        Self {
            #[cfg(feature = "attributes")]
            attributes: Default::default(),
            call,
        }
    }
}

impl GlobalDeclaration {
    /// Get the name of the declaration, if it has one.
    pub fn ident(&self) -> Option<&Ident> {
        match self {
            GlobalDeclaration::Void => None,
            GlobalDeclaration::Declaration(decl) => Some(&decl.ident),
            GlobalDeclaration::TypeAlias(decl) => Some(&decl.ident),
            GlobalDeclaration::Struct(decl) => Some(&decl.ident),
            GlobalDeclaration::Function(decl) => Some(&decl.ident),
            GlobalDeclaration::ConstAssert(_) => None,
        }
    }
    /// Get the name of the declaration, if it has one.
    pub fn ident_mut(&mut self) -> Option<&mut Ident> {
        match self {
            GlobalDeclaration::Void => None,
            GlobalDeclaration::Declaration(decl) => Some(&mut decl.ident),
            GlobalDeclaration::TypeAlias(decl) => Some(&mut decl.ident),
            GlobalDeclaration::Struct(decl) => Some(&mut decl.ident),
            GlobalDeclaration::Function(decl) => Some(&mut decl.ident),
            GlobalDeclaration::ConstAssert(_) => None,
        }
    }
}

/// A trait implemented on all types that can be prefixed by attributes.
pub trait Decorated {
    /// List all attributes (`@name`) of a syntax node.
    fn attributes(&self) -> &[Attribute];
    /// List all attributes (`@name`) of a syntax node.
    fn attributes_mut(&mut self) -> &mut [Attribute];
    /// Remove attributes with predicate.
    fn retain_attributes_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut Attribute) -> bool;
}

impl<T: Decorated> Decorated for Spanned<T> {
    fn attributes(&self) -> &[Attribute] {
        self.node().attributes()
    }

    fn attributes_mut(&mut self) -> &mut [Attribute] {
        self.node_mut().attributes_mut()
    }

    fn retain_attributes_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut Attribute) -> bool,
    {
        self.node_mut().retain_attributes_mut(f)
    }
}

macro_rules! impl_decorated_struct {
    ($ty:ty) => {
        impl Decorated for $ty {
            fn attributes(&self) -> &[Attribute] {
                &self.attributes
            }
            fn attributes_mut(&mut self) -> &mut [Attribute] {
                &mut self.attributes
            }
            fn retain_attributes_mut<F>(&mut self, f: F)
            where
                F: FnMut(&mut Attribute) -> bool,
            {
                self.attributes.retain_mut(f)
            }
        }
    };
}

#[cfg(all(feature = "imports", feature = "attributes"))]
impl_decorated_struct!(ImportStatement);

#[cfg(feature = "attributes")]
impl Decorated for GlobalDirective {
    fn attributes(&self) -> &[Attribute] {
        match self {
            GlobalDirective::Diagnostic(directive) => &directive.attributes,
            GlobalDirective::Enable(directive) => &directive.attributes,
            GlobalDirective::Requires(directive) => &directive.attributes,
        }
    }

    fn attributes_mut(&mut self) -> &mut [Attribute] {
        match self {
            GlobalDirective::Diagnostic(directive) => &mut directive.attributes,
            GlobalDirective::Enable(directive) => &mut directive.attributes,
            GlobalDirective::Requires(directive) => &mut directive.attributes,
        }
    }

    fn retain_attributes_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut Attribute) -> bool,
    {
        match self {
            GlobalDirective::Diagnostic(directive) => directive.attributes.retain_mut(f),
            GlobalDirective::Enable(directive) => directive.attributes.retain_mut(f),
            GlobalDirective::Requires(directive) => directive.attributes.retain_mut(f),
        }
    }
}

#[cfg(feature = "attributes")]
impl_decorated_struct!(DiagnosticDirective);

#[cfg(feature = "attributes")]
impl_decorated_struct!(EnableDirective);

#[cfg(feature = "attributes")]
impl_decorated_struct!(RequiresDirective);

#[cfg(feature = "attributes")]
impl Decorated for GlobalDeclaration {
    fn attributes(&self) -> &[Attribute] {
        match self {
            GlobalDeclaration::Void => &[],
            GlobalDeclaration::Declaration(decl) => &decl.attributes,
            GlobalDeclaration::TypeAlias(decl) => &decl.attributes,
            GlobalDeclaration::Struct(decl) => &decl.attributes,
            GlobalDeclaration::Function(decl) => &decl.attributes,
            GlobalDeclaration::ConstAssert(decl) => &decl.attributes,
        }
    }

    fn attributes_mut(&mut self) -> &mut [Attribute] {
        match self {
            GlobalDeclaration::Void => &mut [],
            GlobalDeclaration::Declaration(decl) => &mut decl.attributes,
            GlobalDeclaration::TypeAlias(decl) => &mut decl.attributes,
            GlobalDeclaration::Struct(decl) => &mut decl.attributes,
            GlobalDeclaration::Function(decl) => &mut decl.attributes,
            GlobalDeclaration::ConstAssert(decl) => &mut decl.attributes,
        }
    }

    fn retain_attributes_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut Attribute) -> bool,
    {
        match self {
            GlobalDeclaration::Void => {}
            GlobalDeclaration::Declaration(decl) => decl.attributes.retain_mut(f),
            GlobalDeclaration::TypeAlias(decl) => decl.attributes.retain_mut(f),
            GlobalDeclaration::Struct(decl) => decl.attributes.retain_mut(f),
            GlobalDeclaration::Function(decl) => decl.attributes.retain_mut(f),
            GlobalDeclaration::ConstAssert(decl) => decl.attributes.retain_mut(f),
        }
    }
}

impl_decorated_struct!(Declaration);

#[cfg(feature = "attributes")]
impl_decorated_struct!(TypeAlias);

#[cfg(feature = "attributes")]
impl_decorated_struct!(Struct);

impl_decorated_struct!(StructMember);

impl_decorated_struct!(Function);

impl_decorated_struct!(FormalParameter);

#[cfg(feature = "attributes")]
impl_decorated_struct!(ConstAssert);

#[cfg(feature = "attributes")]
impl Decorated for Statement {
    fn attributes(&self) -> &[Attribute] {
        match self {
            Statement::Void => &[],
            Statement::Compound(stmt) => &stmt.attributes,
            Statement::Assignment(stmt) => &stmt.attributes,
            Statement::Increment(stmt) => &stmt.attributes,
            Statement::Decrement(stmt) => &stmt.attributes,
            Statement::If(stmt) => &stmt.attributes,
            Statement::Switch(stmt) => &stmt.attributes,
            Statement::Loop(stmt) => &stmt.attributes,
            Statement::For(stmt) => &stmt.attributes,
            Statement::While(stmt) => &stmt.attributes,
            Statement::Break(stmt) => &stmt.attributes,
            Statement::Continue(stmt) => &stmt.attributes,
            Statement::Return(stmt) => &stmt.attributes,
            Statement::Discard(stmt) => &stmt.attributes,
            Statement::FunctionCall(stmt) => &stmt.attributes,
            Statement::ConstAssert(stmt) => &stmt.attributes,
            Statement::Declaration(stmt) => &stmt.attributes,
        }
    }

    fn attributes_mut(&mut self) -> &mut [Attribute] {
        match self {
            Statement::Void => &mut [],
            Statement::Compound(stmt) => &mut stmt.attributes,
            Statement::Assignment(stmt) => &mut stmt.attributes,
            Statement::Increment(stmt) => &mut stmt.attributes,
            Statement::Decrement(stmt) => &mut stmt.attributes,
            Statement::If(stmt) => &mut stmt.attributes,
            Statement::Switch(stmt) => &mut stmt.attributes,
            Statement::Loop(stmt) => &mut stmt.attributes,
            Statement::For(stmt) => &mut stmt.attributes,
            Statement::While(stmt) => &mut stmt.attributes,
            Statement::Break(stmt) => &mut stmt.attributes,
            Statement::Continue(stmt) => &mut stmt.attributes,
            Statement::Return(stmt) => &mut stmt.attributes,
            Statement::Discard(stmt) => &mut stmt.attributes,
            Statement::FunctionCall(stmt) => &mut stmt.attributes,
            Statement::ConstAssert(stmt) => &mut stmt.attributes,
            Statement::Declaration(stmt) => &mut stmt.attributes,
        }
    }

    fn retain_attributes_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut Attribute) -> bool,
    {
        match self {
            Statement::Void => {}
            Statement::Compound(stmt) => stmt.attributes.retain_mut(f),
            Statement::Assignment(stmt) => stmt.attributes.retain_mut(f),
            Statement::Increment(stmt) => stmt.attributes.retain_mut(f),
            Statement::Decrement(stmt) => stmt.attributes.retain_mut(f),
            Statement::If(stmt) => stmt.attributes.retain_mut(f),
            Statement::Switch(stmt) => stmt.attributes.retain_mut(f),
            Statement::Loop(stmt) => stmt.attributes.retain_mut(f),
            Statement::For(stmt) => stmt.attributes.retain_mut(f),
            Statement::While(stmt) => stmt.attributes.retain_mut(f),
            Statement::Break(stmt) => stmt.attributes.retain_mut(f),
            Statement::Continue(stmt) => stmt.attributes.retain_mut(f),
            Statement::Return(stmt) => stmt.attributes.retain_mut(f),
            Statement::Discard(stmt) => stmt.attributes.retain_mut(f),
            Statement::FunctionCall(stmt) => stmt.attributes.retain_mut(f),
            Statement::ConstAssert(stmt) => stmt.attributes.retain_mut(f),
            Statement::Declaration(stmt) => stmt.attributes.retain_mut(f),
        }
    }
}

impl_decorated_struct!(CompoundStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(AssignmentStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(IncrementStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(DecrementStatement);

impl_decorated_struct!(IfStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(ElseIfClause);

#[cfg(feature = "attributes")]
impl_decorated_struct!(ElseClause);

impl_decorated_struct!(SwitchStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(SwitchClause);

impl_decorated_struct!(LoopStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(ContinuingStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(BreakIfStatement);

impl_decorated_struct!(ForStatement);

impl_decorated_struct!(WhileStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(BreakStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(ContinueStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(ReturnStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(DiscardStatement);

#[cfg(feature = "attributes")]
impl_decorated_struct!(FunctionCallStatement);
