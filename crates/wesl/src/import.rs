use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use wgsl_parse::syntax::{
    self, GlobalDeclaration, Ident, ImportContent, ImportStatement, ModulePath, TranslationUnit,
    TypeExpression,
};

use crate::{visit::Visit, Mangler, ResolveError, Resolver};

type Imports = HashMap<Ident, (ModulePath, Ident)>;
type Modules = HashMap<ModulePath, Rc<RefCell<Module>>>;

/// Error produced during import resolution.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ImportError {
    #[error("duplicate imported item `{0}`")]
    DuplicateSymbol(String),
    #[error("{0}")]
    ResolveError(#[from] ResolveError),
    #[error("module `{0}` has no declaration `{1}`")]
    MissingDecl(ModulePath, String),
}

type E = ImportError;

#[derive(Debug)]
pub(crate) struct Module {
    pub(crate) source: TranslationUnit,
    pub(crate) path: ModulePath,
    idents: HashMap<Ident, usize>, // lookup (ident, decl_index)
    treated_idents: RefCell<HashSet<Ident>>, // used idents that have already been usage-analyzed
    imports: Imports,
}

impl Module {
    pub(crate) fn new(source: TranslationUnit, path: ModulePath) -> Self {
        let idents = source
            .global_declarations
            .iter()
            .enumerate()
            .filter_map(|(i, decl)| decl.ident().map(|id| (id.clone(), i)))
            .collect();
        let imports = flatten_imports(&source.imports, &path);
        Self {
            source,
            path,
            idents,
            treated_idents: Default::default(),
            imports,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Resolutions {
    modules: Modules,
    order: Vec<ModulePath>,
}

impl Resolutions {
    /// Warning: you *must* call `push_module` right after this.
    pub(crate) fn new() -> Self {
        Resolutions {
            modules: Default::default(),
            order: Default::default(),
        }
    }
    pub(crate) fn root_module(&self) -> Rc<RefCell<Module>> {
        self.modules.get(self.root_path()).unwrap().clone() // safety: new() requires push_module
    }
    pub(crate) fn root_path(&self) -> &ModulePath {
        self.order.first().unwrap() // safety: new() requires push_module
    }
    pub(crate) fn modules(&self) -> impl Iterator<Item = Rc<RefCell<Module>>> + '_ {
        self.order.iter().map(|i| self.modules[i].clone())
    }
    pub(crate) fn push_module(&mut self, module: Module) -> Rc<RefCell<Module>> {
        let path = module.path.clone();
        let module = Rc::new(RefCell::new(module));
        self.modules.insert(path.clone(), module.clone());
        self.order.push(path);
        module
    }
}

fn resolve_inline_path(
    path: &ModulePath,
    parent_path: &ModulePath,
    imports: &Imports,
) -> ModulePath {
    match path.origin {
        syntax::PathOrigin::Absolute => path.clone(),
        syntax::PathOrigin::Relative(_) => parent_path.join_path(path).unwrap(),
        syntax::PathOrigin::Package => {
            let prefix = path.first().unwrap();
            // the path could be either a package, of referencing an imported module alias.
            imports
                .iter()
                .find_map(|(ident, (ext_res, ext_ident))| {
                    if *ident.name() == prefix {
                        // import a::b::c as foo; foo::bar::baz() => a::b::c::bar::baz()
                        let mut res = ext_res.clone(); // a::b
                        res.push(&ext_ident.name()); // c
                        Some(res.join(path.components.iter().skip(1).cloned()))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| path.clone())
        }
    }
}

// XXX: it's quite messy.
/// Load all modules "used" transitively by the root module. Make external idents point at
/// the right declaration in the external module.
///
/// it is "lazy" because external modules are loaded only if used by the `keep` declarations
/// or module-scope `const_assert`s.
///
/// "used": used declarations in the root module are the `keep` parameter. Used declarations
/// in other modules are those reached by `keep` the declaration, recursively.
/// Module-scope `const_assert`s are always included.
///
/// Returns a list of [`Module`]s with the list of their "used" idents.
///
/// See also: [`resolve_eager`]
pub fn resolve_lazy<'a>(
    keep: impl IntoIterator<Item = &'a Ident>,
    resolutions: &mut Resolutions,
    resolver: &impl Resolver,
) -> Result<(), E> {
    fn load_module(
        path: &ModulePath,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<Rc<RefCell<Module>>, E> {
        if let Some(module) = resolutions.modules.get(path) {
            Ok(module.clone())
        } else {
            let source = resolver.resolve_module(path)?;
            let module = Module::new(source, path.clone());

            // const_asserts of used modules must be included.
            // https://github.com/wgsl-tooling-wg/wesl-spec/issues/66
            let const_asserts = module
                .source
                .global_declarations
                .iter()
                .filter(|decl| decl.is_const_assert());

            for decl in const_asserts {
                resolve_decl(&module, decl, resolutions, resolver)?;
            }

            Ok(resolutions.push_module(module))
        }
    }

    fn resolve_ident(
        module: &Module,
        ident: &str,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), E> {
        let (ident, n) = module
            .idents
            .iter()
            .find(|(id, _)| *id.name() == ident)
            .ok_or_else(|| E::MissingDecl(module.path.clone(), ident.to_string()))?;

        if module.treated_idents.borrow().contains(ident) {
            return Ok(());
        } else {
            module.treated_idents.borrow_mut().insert(ident.clone());
        }

        let decl = module.source.global_declarations.get(*n).unwrap();
        resolve_decl(module, decl, resolutions, resolver)
    }

    fn resolve_decl(
        module: &Module,
        decl: &GlobalDeclaration,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), E> {
        for ty in Visit::<TypeExpression>::visit(decl) {
            // first, the recursive call
            for ty in Visit::<TypeExpression>::visit(ty) {
                resolve_ident(module, &ty.ident.name(), resolutions, resolver)?;
            }

            let (ext_path, ext_id) = if let Some(path) = &ty.path {
                let res = resolve_inline_path(path, &module.path, &module.imports);
                (res, ty.ident.clone())
            } else if let Some((path, ident)) = module.imports.get(&ty.ident) {
                (path.clone(), ident.clone())
            } else {
                // points to a local decl, we stop here.
                if let Some(n) = module.idents.get(&ty.ident) {
                    let decl = module.source.global_declarations.get(*n).unwrap();
                    if module.treated_idents.borrow().contains(&ty.ident) {
                        continue;
                    } else {
                        module.treated_idents.borrow_mut().insert(ty.ident.clone());
                        return resolve_decl(module, decl, resolutions, resolver);
                    }
                } else {
                    continue;
                };
            };

            // if the import path points to a local decl, we stop here
            if ext_path == module.path {
                if module.idents.contains_key(&ext_id) {
                    return resolve_ident(module, &ext_id.name(), resolutions, resolver);
                } else {
                    return Err(E::MissingDecl(ext_path, ext_id.to_string()));
                }
            }

            // load the external module for this external ident
            let ext_mod = load_module(&ext_path, resolutions, resolver)?;
            resolve_ident(&ext_mod.borrow(), &ext_id.name(), resolutions, resolver)?;
        }

        Ok(())
    }

    let path = resolutions.root_path().clone();
    let module = load_module(&path, resolutions, resolver)?;

    for id in keep {
        resolve_ident(&module.borrow(), &id.name(), resolutions, resolver)?;
    }

    resolutions.retarget();
    Ok(())
}

pub fn resolve_eager(resolutions: &mut Resolutions, resolver: &impl Resolver) -> Result<(), E> {
    fn resolve_ty(
        module: &Module,
        ty: &TypeExpression,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), E> {
        for ty in Visit::<TypeExpression>::visit(ty) {
            resolve_ty(module, ty, resolutions, resolver)?;
        }

        let (ext_path, ext_id) = if let Some(path) = &ty.path {
            let res = resolve_inline_path(path, &module.path, &module.imports);
            (res, ty.ident.clone())
        } else if let Some((path, ident)) = module.imports.get(&ty.ident) {
            (path.clone(), ident.clone())
        } else {
            // points to a local decl, we stop here.
            return Ok(());
        };

        // if the import path points to a local decl, we stop here
        if ext_path == module.path {
            if module.idents.contains_key(&ty.ident) {
                return Ok(());
            } else {
                return Err(E::MissingDecl(ext_path, ty.ident.to_string()));
            }
        }

        // load the external module for this external ident
        let ext_mod = if let Some(module) = resolutions.modules.get(&ext_path) {
            module.clone()
        } else {
            let source = resolver.resolve_module(&ext_path)?;
            let module = resolutions.push_module(Module::new(source, ext_path.clone()));
            resolve_module(&module.borrow(), resolutions, resolver)?;
            module
        };

        // get the ident of the external declaration pointed to by the type
        if ext_mod
            .borrow() // safety: only 1 module is borrowed at a time, the current one.
            .idents
            .iter()
            .find(|(id, _)| *id.name() == *ext_id.name())
            .is_none()
        {
            return Err(E::MissingDecl(ext_path.clone(), ext_id.to_string()));
        }
        Ok(())
    }
    fn resolve_module(
        module: &Module,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), E> {
        for (path, _) in module.imports.values() {
            if !resolutions.modules.contains_key(path) {
                let source = resolver.resolve_module(path)?;
                let module = resolutions.push_module(Module::new(source, path.clone()));
                resolve_module(&module.borrow(), resolutions, resolver)?;
            }
        }

        for ty in Visit::<TypeExpression>::visit(&module.source) {
            resolve_ty(module, ty, resolutions, resolver)?;
        }
        Ok(())
    }

    let module = resolutions.root_module();
    resolve_module(&module.borrow(), resolutions, resolver)?;
    resolutions.retarget();
    Ok(())
}

/// Flatten imports to a list of module paths.
pub(crate) fn flatten_imports(imports: &[ImportStatement], parent_path: &ModulePath) -> Imports {
    fn rec(content: &ImportContent, path: ModulePath, res: &mut Imports) {
        match content {
            ImportContent::Item(item) => {
                let ident = item.rename.as_ref().unwrap_or(&item.ident).clone();
                res.insert(ident, (path, item.ident.clone()));
            }
            ImportContent::Collection(coll) => {
                for import in coll {
                    let path = path.clone().join(import.path.clone());
                    rec(&import.content, path, res)
                }
            }
        }
    }

    let mut res = Imports::new();

    for import in imports {
        let path = parent_path
            .join_path(&import.path)
            .unwrap_or_else(|| import.path.clone());
        rec(&import.content, path, &mut res);
    }

    res
}

pub(crate) fn mangle_decls<'a>(
    wgsl: &'a mut TranslationUnit,
    path: &'a ModulePath,
    mangler: &impl Mangler,
) {
    wgsl.global_declarations
        .iter_mut()
        .filter_map(|decl| decl.ident_mut())
        .for_each(|ident| {
            let new_name = mangler.mangle(path, &ident.name());
            ident.rename(new_name.clone());
        })
}

impl Resolutions {
    pub fn retarget(&mut self) {
        for module in self.modules.values() {
            let mut module = module.borrow_mut();
            let module = &mut *module;
            for ty in Visit::<TypeExpression>::visit_mut(&mut module.source) {
                let (ext_path, ext_id) = if let Some(path) = &ty.path {
                    let res = resolve_inline_path(path, &module.path, &module.imports);
                    (res, ty.ident.clone())
                } else if let Some((path, ident)) = module.imports.get(&ty.ident) {
                    (path.clone(), ident.clone())
                } else {
                    // points to a local decl, we stop here.
                    continue;
                };

                // if the import path points to a local decl, we stop here
                if ext_path == module.path {
                    continue;
                }

                // load the external module for this external ident
                if let Some(module) = self.modules.get(&ext_path) {
                    // get the ident of the external declaration pointed to by the type
                    let ext_id = module
                        .borrow() // safety: only 1 module is borrowed at a time, the current one.
                        .idents
                        .iter()
                        .find(|(id, _)| *id.name() == *ext_id.name())
                        .map(|(id, _)| id.clone())
                        .expect("external declaration not found");

                    ty.path = None;
                    ty.ident = ext_id;
                }
            }
        }
    }

    pub fn mangle(&mut self, mangler: &impl Mangler) {
        let root_path = self.root_path().clone();
        for (path, module) in self.modules.iter_mut() {
            if path != &root_path {
                let mut module = module.borrow_mut();
                mangle_decls(&mut module.source, path, mangler);
            }
        }
    }

    pub fn assemble(&self, strip: bool) -> TranslationUnit {
        let mut wesl = TranslationUnit::default();
        for module in self.modules() {
            let module = module.borrow();
            if strip {
                wesl.global_declarations.extend(
                    module
                        .source
                        .global_declarations
                        .iter()
                        .filter(|decl| {
                            decl.is_const_assert()
                                || decl
                                    .ident()
                                    .is_some_and(|id| module.treated_idents.borrow().contains(id))
                        })
                        .cloned(),
                );
            } else {
                wesl.global_declarations
                    .extend(module.source.global_declarations.clone());
            }
            wesl.global_directives
                .extend(module.source.global_directives.clone());
        }
        // TODO: <https://github.com/wgsl-tooling-wg/wesl-spec/issues/71>
        // currently the behavior is:
        // * include all directives used (if strip)
        // * include all directives (if not strip)
        wesl.global_directives.dedup();
        wesl
    }
}
