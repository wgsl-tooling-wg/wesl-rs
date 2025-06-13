use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use itertools::Itertools;
use wgsl_parse::syntax::{
    GlobalDeclaration, Ident, ImportContent, ImportStatement, ModulePath, PathOrigin,
    TranslationUnit, TypeExpression,
};

use crate::{Diagnostic, Error, Mangler, ResolveError, Resolver, SyntaxUtil, visit::Visit};

#[derive(Clone, Debug)]
struct ImportItem {
    path: ModulePath,
    ident: Ident,
    public: bool,
}

type Imports = HashMap<Ident, ImportItem>;
type Modules = HashMap<ModulePath, Rc<RefCell<Module>>>;

/// Error produced during import resolution.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ImportError {
    #[error("duplicate declaration of `{0}`")]
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
    is_resolved: bool,
}

impl Module {
    pub(crate) fn new(source: TranslationUnit, path: ModulePath) -> Result<Self, E> {
        let idents = source
            .global_declarations
            .iter()
            .enumerate()
            .filter_map(|(i, decl)| decl.ident().map(|id| (id.clone(), i)))
            .collect::<HashMap<_, _>>();
        let imports = flatten_imports(&source.imports, &path)?;

        // TODO: this is not correct because of conditional compilation.
        // for id in idents.keys() {
        //     if imports
        //         .keys()
        //         .any(|k| k.name().as_str() == id.name().as_str())
        //     {
        //         return Err(E::DuplicateSymbol(id.to_string()));
        //     }
        // }

        Ok(Self {
            source,
            path,
            idents,
            treated_idents: Default::default(),
            imports,
            is_resolved: false,
        })
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
    pub(crate) fn into_module_order(self) -> Vec<ModulePath> {
        self.order
    }
}

fn err_with_module(e: Error, module: &Module, resolver: &impl Resolver) -> Error {
    Error::from(
        Diagnostic::from(e)
            .with_module_path(module.path.clone(), resolver.display_name(&module.path)),
    )
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
) -> Result<(), Error> {
    fn load_module(
        path: &ModulePath,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<Rc<RefCell<Module>>, Error> {
        let module = if let Some(module) = resolutions.modules.get(path) {
            module.clone()
        } else {
            let mut source = resolver.resolve_module(path)?;
            source.retarget_idents();
            let module = Module::new(source, path.clone())?;
            resolutions.push_module(module)
        };

        {
            let mut module = module.borrow_mut();
            if !module.is_resolved {
                module.is_resolved = true;
                // const_asserts of used modules must be included.
                // https://github.com/wgsl-tooling-wg/wesl-spec/issues/66
                let const_asserts = module
                    .source
                    .global_declarations
                    .iter()
                    .filter(|decl| decl.is_const_assert());

                for decl in const_asserts {
                    resolve_decl(&module, decl, resolutions, resolver)
                        .map_err(|e| err_with_module(e, &module, resolver))?;
                }
            }
        }

        Ok(module)
    }

    fn resolve_ident(
        module: &Module,
        name: &Ident,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), Error> {
        let decl = if let Some((ident, n)) = module
            .idents
            .iter()
            .find(|(id, _)| *id.name() == *name.name())
        {
            if module.treated_idents.borrow().contains(ident) {
                return Ok(());
            } else {
                module.treated_idents.borrow_mut().insert(ident.clone());
            }
            let decl = module.source.global_declarations.get(*n).unwrap();
            decl
        } else if let Some((_, item)) = module
            .imports
            .iter()
            .find(|(id, item)| *id.name() == *name.name())
        {
            panic!("TODO")
        } else {
            return Err(E::MissingDecl(module.path.clone(), name.to_string()).into());
        };

        resolve_decl(module, decl, resolutions, resolver)
    }

    // fn resolve_name(
    //     module: &Module,
    //     name: &str,
    //     resolutions: &mut Resolutions,
    //     resolver: &impl Resolver,
    // ) -> Result<(), E> {
    //     let decl = if let Some((ident, n)) = module.idents.iter().find(|(id, _)| *id.name() == name)
    //     {
    //         println!("{name} {}")
    //         if module.treated_idents.borrow().contains(ident) {
    //             return Ok(());
    //         } else {
    //             module.treated_idents.borrow_mut().insert(ident.clone());
    //         }
    //         let decl = module.source.global_declarations.get(*n).unwrap();
    //         decl
    //     } else if let Some((_, item)) = module.imports.iter().find(|(id, item)| *id.name() == name)
    //     {
    //         panic!("TODO")
    //     } else {
    //         return Err(E::MissingDecl(module.path.clone(), name.to_string()));
    //     };

    //     resolve_decl(module, decl, resolutions, resolver)
    // }

    fn resolve_ty(
        module: &Module,
        ty: &TypeExpression,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), Error> {
        // first, the recursive call
        for ty in Visit::<TypeExpression>::visit(ty) {
            resolve_ty(module, ty, resolutions, resolver)?;
        }

        let (ext_path, ext_id) = if let Some(path) = &ty.path {
            let path = resolve_inline_path(path, &module.path, &module.imports);
            (path, ty.ident.clone())
        } else if let Some(item) = module.imports.get(&ty.ident) {
            (item.path.clone(), item.ident.clone())
        } else {
            // points to a local decl, we stop here.
            if let Some(n) = module.idents.get(&ty.ident) {
                let decl = module.source.global_declarations.get(*n).unwrap();
                if module.treated_idents.borrow().contains(&ty.ident) {
                    return Ok(());
                } else {
                    module.treated_idents.borrow_mut().insert(ty.ident.clone());
                    return resolve_decl(module, decl, resolutions, resolver);
                }
            } else {
                return Ok(());
            };
        };

        // if the import path points to a local decl, we stop here
        if ext_path == module.path {
            return resolve_ident(module, &ext_id, resolutions, resolver);
        }

        // load the external module for this external ident
        let ext_mod = load_module(&ext_path, resolutions, resolver)?;
        resolve_ident(&ext_mod.borrow(), &ext_id, resolutions, resolver)?;
        Ok(())
    }

    fn resolve_decl(
        module: &Module,
        decl: &GlobalDeclaration,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), Error> {
        for ty in Visit::<TypeExpression>::visit(decl) {
            resolve_ty(module, ty, resolutions, resolver)?;
        }

        Ok(())
    }

    let path = resolutions.root_path().clone();
    let module = load_module(&path, resolutions, resolver)?;

    for id in keep {
        let module = module.borrow();
        resolve_ident(&module, id, resolutions, resolver)
            .map_err(|e| err_with_module(e, &module, resolver))?;
    }

    resolutions.retarget();
    Ok(())
}

pub fn resolve_eager(resolutions: &mut Resolutions, resolver: &impl Resolver) -> Result<(), Error> {
    fn resolve_ty(
        module: &Module,
        ty: &TypeExpression,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), Error> {
        for ty in Visit::<TypeExpression>::visit(ty) {
            resolve_ty(module, ty, resolutions, resolver)?;
        }

        let (ext_path, ext_id) = if let Some(path) = &ty.path {
            let res = resolve_inline_path(path, &module.path, &module.imports);
            (res, ty.ident.clone())
        } else if let Some(item) = module.imports.get(&ty.ident) {
            (item.path.clone(), item.ident.clone())
        } else {
            // points to a local decl, we stop here.
            return Ok(());
        };

        // if the import path points to a local decl, we stop here
        if ext_path == module.path {
            if module.idents.contains_key(&ty.ident) {
                return Ok(());
            } else {
                return Err(E::MissingDecl(ext_path, ty.ident.to_string()).into());
            }
        }

        // load the external module for this external ident
        let ext_mod = if let Some(module) = resolutions.modules.get(&ext_path) {
            module.clone()
        } else {
            let mut source = resolver.resolve_module(&ext_path)?;
            source.retarget_idents();
            let module = resolutions.push_module(Module::new(source, ext_path.clone())?);
            resolve_module(&module.borrow(), resolutions, resolver)?;
            module
        };

        let ext_mod = ext_mod.borrow(); // safety: only 1 module is borrowed at a time, the current one.
        // get the ident of the external declaration pointed to by the type
        if !ext_mod.idents.keys().any(|id| *id.name() == *ext_id.name())
            // TODO private err msg
            && !ext_mod
                .imports
                .iter()
                .any(|(id, item)| item.public && *id.name() == *ext_id.name())
        {
            return Err(err_with_module(
                E::MissingDecl(ext_path.clone(), ext_id.to_string()).into(),
                &module,
                resolver,
            ));
        }
        Ok(())
    }
    fn resolve_module(
        module: &Module,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), Error> {
        for item in module.imports.values() {
            if !resolutions.modules.contains_key(&item.path) {
                let mut source = resolver.resolve_module(&item.path)?;
                source.retarget_idents();
                let module = resolutions.push_module(Module::new(source, item.path.clone())?);
                let module = module.borrow();
                resolve_module(&module, resolutions, resolver)
                    .map_err(|e| err_with_module(e, &module, resolver))?;
            }
        }

        for ty in Visit::<TypeExpression>::visit(&module.source) {
            resolve_ty(module, ty, resolutions, resolver)?;
        }
        Ok(())
    }

    let module = resolutions.root_module();
    {
        let module = module.borrow();
        resolve_module(&module, resolutions, resolver)
            .map_err(|e| err_with_module(e, &module, resolver))?;
    }
    resolutions.retarget();
    Ok(())
}

fn join_paths(parent_path: &ModulePath, path: &ModulePath) -> ModulePath {
    match (parent_path.origin, path.origin) {
        (PathOrigin::Absolute | PathOrigin::Relative(_), _)
        | (PathOrigin::Package, PathOrigin::Relative(_)) => {
            parent_path.join_path(&path).unwrap_or_else(|| path.clone())
        }
        // Absolute imports from within a package correspond to package imports.
        (PathOrigin::Package, PathOrigin::Absolute) => ModulePath::new(
            PathOrigin::Package,
            parent_path
                .first()
                .map(str::to_string)
                .into_iter()
                .chain(path.components.iter().cloned())
                .collect_vec(),
        ),
        // Importing a sub-package. This is a hack: we rename the package to
        // parent_package/child_package, which cannot be spelled in code.
        (PathOrigin::Package, PathOrigin::Package) => ModulePath::new(
            PathOrigin::Package,
            parent_path
                .first()
                .map(|name| format!("{name}/{}", path.first().unwrap()))
                .into_iter()
                .chain(path.components.iter().skip(1).cloned())
                .collect_vec(),
        ),
    }
}

/// Flatten imports to a list of module paths.
fn flatten_imports(imports: &[ImportStatement], parent_path: &ModulePath) -> Result<Imports, E> {
    fn rec(
        content: &ImportContent,
        path: ModulePath,
        public: bool,
        res: &mut Imports,
    ) -> Result<(), E> {
        match content {
            ImportContent::Item(item) => {
                let ident = item.rename.as_ref().unwrap_or(&item.ident).clone();
                // TODO: this is not correct because of conditional compilation.
                // if res
                //     .keys()
                //     .any(|k| k.name().as_str() == ident.name().as_str())
                // {
                //     return Err(E::DuplicateSymbol(ident.to_string()));
                // }
                res.insert(
                    ident,
                    ImportItem {
                        path,
                        ident: item.ident.clone(),
                        public,
                    },
                );
            }
            ImportContent::Collection(coll) => {
                for import in coll {
                    let path = path.clone().join(import.path.clone());
                    rec(&import.content, path, public, res)?;
                }
            }
        }
        Ok(())
    }

    let mut res = Imports::new();

    for import in imports {
        let path = join_paths(parent_path, &import.path);
        let public = import.attributes.iter().any(|attr| attr.is_publish());
        rec(&import.content, path, public, &mut res)?;
    }
    Ok(res)
}

fn resolve_inline_path(
    path: &ModulePath,
    parent_path: &ModulePath,
    imports: &Imports,
) -> ModulePath {
    let module_path = match path.origin {
        PathOrigin::Package => {
            // the path could be either a package, of referencing an imported module alias.
            let prefix = path.first().unwrap();
            let imported_item = imports.iter().find(|(ident, _)| *ident.name() == prefix);

            if let Some((_, ext_item)) = imported_item {
                // this inline path references an imported item. Example:
                // import a::b::c as foo; foo::bar::baz() => a::b::c::bar::baz()
                let mut res = ext_item.path.clone(); // a::b
                res.push(&ext_item.ident.name()); // c
                res.join(path.components.iter().skip(1).cloned())
            } else {
                join_paths(parent_path, path)
            }
        }
        _ => join_paths(parent_path, path),
    };
    module_path
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
            Visit::<TypeExpression>::visit_rec_mut(&mut module.source, &mut |ty| {
                let (ext_path, ext_id) = if let Some(path) = &ty.path {
                    let res = resolve_inline_path(path, &module.path, &module.imports);
                    (res, ty.ident.clone())
                } else if let Some(item) = module.imports.get(&ty.ident) {
                    (item.path.clone(), item.ident.clone())
                } else {
                    // points to a local decl, we stop here.
                    return;
                };

                // if the import path points to a local decl
                if ext_path == module.path {
                    let ext_id = module
                        .idents
                        .iter()
                        .find(|(id, _)| *id.name() == *ext_id.name())
                        .map(|(id, _)| id.clone())
                        .expect("external declaration not found");
                    ty.path = None;
                    ty.ident = ext_id;
                }
                // load the external module for this external ident
                else if let Some(module) = self.modules.get(&ext_path) {
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
            });
        }
    }

    pub fn mangle(&mut self, mangler: &impl Mangler, mangle_root: bool) {
        let root_path = self.root_path().clone();
        for (path, module) in self.modules.iter_mut() {
            if mangle_root || path != &root_path {
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
