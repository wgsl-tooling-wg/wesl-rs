use std::{
    cell::{Ref, RefCell},
    collections::{HashMap, HashSet},
    ops::DerefMut,
    rc::Rc,
};

use itertools::Itertools;
use wgsl_parse::syntax::{
    self, Ident, ImportContent, ImportStatement, ModulePath, TranslationUnit, TypeExpression,
};

use crate::{visit::Visit, Mangler, ResolveError, Resolver};

type Imports = HashMap<Ident, (ModulePath, Ident)>;
type Decls = HashMap<ModulePath, HashSet<usize>>;
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
    #[error("circular dependency involving `{0}`")]
    CircularDependency(ModulePath),
}

type E = ImportError;

pub(crate) struct Module {
    pub(crate) source: TranslationUnit,
    pub(crate) path: ModulePath,
    idents: HashMap<Ident, usize>,  // lookup (ident, decl_index)
    treated_idents: HashSet<Ident>, // used idents that have already been usage-analyzed
    imports: Imports,
}

impl Module {
    fn new(source: TranslationUnit, path: ModulePath) -> Self {
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
    #[allow(unused)]
    fn used_idents(&self) -> impl Iterator<Item = &Ident> {
        self.treated_idents.iter()
    }
}

pub(crate) struct Resolutions {
    modules: Modules,
    order: Vec<ModulePath>,
}

impl Resolutions {
    pub(crate) fn root_path(&self) -> &ModulePath {
        self.order.first().unwrap() // safety: new() guarantees that there is always a root module
    }
    pub(crate) fn modules(&self) -> impl Iterator<Item = Ref<Module>> {
        self.order.iter().map(|res| self.modules[res].borrow())
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
pub fn resolve_lazy(
    root: TranslationUnit,
    path: &ModulePath,
    keep: HashSet<Ident>,
    resolver: &impl Resolver,
) -> Result<Resolutions, E> {
    fn load_module(
        path: &ModulePath,
        local_decls: &mut HashSet<usize>,
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
                .enumerate()
                .filter_map(|(i, decl)| decl.is_const_assert().then_some(i));
            local_decls.extend(const_asserts);

            let module = Rc::new(RefCell::new(module));
            resolutions.push_module(path.clone(), module.clone());

            Ok(module)
        }
    }

    fn resolve_ty(
        mod_path: &ModulePath,
        mod_imports: &Imports,
        mod_idents: &HashMap<Ident, usize>,
        mod_treated_idents: &HashSet<Ident>,
        ty: &mut TypeExpression,
        local_decls: &mut HashSet<usize>,
        extern_decls: &mut Decls,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), E> {
        for ty in Visit::<TypeExpression>::visit_mut(ty) {
            resolve_ty(
                mod_path,
                mod_imports,
                mod_idents,
                mod_treated_idents,
                ty,
                local_decls,
                extern_decls,
                resolutions,
                resolver,
            )?;
        }

        if mod_treated_idents.contains(&ty.ident) {
            return Ok(());
        }

        // get the the module path associated with the type, if it points to a decl in another module.
        let (ext_path, ext_id) = if let Some(path) = &ty.path {
            let ext_path = resolve_inline_path(path, mod_path, mod_imports);
            (ext_path, ty.ident.clone())
        } else if let Some((path, id)) = mod_imports.get(&ty.ident) {
            (path.clone(), id.clone())
        } else {
            // points to a local decl, we stop here.
            if let Some(decl) = mod_idents.get(&ty.ident) {
                local_decls.insert(*decl);
            }
            return Ok(());
        };

        // if the import path points to a local decl, we stop here
        if &ext_path == mod_path {
            if let Some(decl) = mod_idents.get(&ty.ident) {
                local_decls.insert(*decl);
                return Ok(());
            } else {
                return Err(E::MissingDecl(ext_path, ty.ident.name().to_string()));
            }
        }

        // get or load the external module
        let ext_mod = load_module(&ext_path, &mut HashSet::new(), resolutions, resolver)?;
        let mut ext_mod = ext_mod
            .try_borrow_mut()
            .map_err(|_| E::CircularDependency(mod_path.clone()))?;
        let ext_mod = ext_mod.deref_mut();

        // get the ident of the external declaration pointed to by the type
        let (ext_id, ext_decl) = ext_mod
            .idents
            .iter()
            .find(|(id, _)| *id.name() == *ext_id.name())
            .map(|(id, decl)| (id.clone(), *decl))
            .ok_or_else(|| E::MissingDecl(ext_path.clone(), ext_id.to_string()))?;

        if !ext_mod.treated_idents.contains(&ext_id) {
            extern_decls.entry(ext_path).or_default().insert(ext_decl);
        }

        ty.path = None;
        ty.ident = ext_id;
        Ok(())
    }

    fn resolve_decl(
        module: &mut Module,
        decl: usize,
        local_decls: &mut HashSet<usize>,
        extern_decls: &mut Decls,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), E> {
        let decl = module.source.global_declarations.get_mut(decl).unwrap();

        if let Some(id) = decl.ident() {
            if !module.treated_idents.insert(id.clone()) {
                return Ok(());
            }
        }

        for ty in Visit::<TypeExpression>::visit_mut(decl) {
            resolve_ty(
                &module.path,
                &module.imports,
                &module.idents,
                &module.treated_idents,
                ty,
                local_decls,
                extern_decls,
                resolutions,
                resolver,
            )?;
        }

        Ok(())
    }

    fn resolve_decls(
        path: &ModulePath,
        local_decls: &mut HashSet<usize>,
        extern_decls: &mut Decls,
        resolver: &impl Resolver,
        resolutions: &mut Resolutions,
    ) -> Result<(), E> {
        let module = load_module(path, &mut HashSet::new(), resolutions, resolver)?;
        let mut module = module
            .try_borrow_mut()
            .map_err(|_| E::CircularDependency(path.clone()))?;
        let module = module.deref_mut();

        let mut next_decls = HashSet::new();

        while !local_decls.is_empty() {
            for decl in local_decls.iter() {
                resolve_decl(
                    module,
                    *decl,
                    &mut next_decls,
                    extern_decls,
                    resolutions,
                    resolver,
                )?;
            }

            std::mem::swap(local_decls, &mut next_decls);
            next_decls.clear();
        }

        Ok(())
    }

    let mut resolutions = Resolutions::new();
    let module = Module::new(root, path.clone());

    let mut keep_decls: HashSet<usize> = keep
        .iter()
        .map(|id| {
            module
                .idents
                .get(id)
                .copied()
                .ok_or_else(|| E::MissingDecl(path.clone(), id.to_string()))
        })
        .try_collect()?;

    // const_asserts of used modules must be included.
    // https://github.com/wgsl-tooling-wg/wesl-spec/issues/66
    let const_asserts = module
        .source
        .global_declarations
        .iter()
        .enumerate()
        .filter_map(|(i, decl)| decl.is_const_assert().then_some(i));
    keep_decls.extend(const_asserts);

    let mut decls = Decls::new();
    let mut next_decls = Decls::new();
    decls.insert(path.clone(), keep_decls);

    let module = Rc::new(RefCell::new(module));
    resolutions.push_module(path.clone(), module.clone());

    while !decls.is_empty() {
        for (path, decls) in &mut decls {
            resolve_decls(path, decls, &mut next_decls, resolver, &mut resolutions)?;
        }
        std::mem::swap(&mut decls, &mut next_decls);
        next_decls.clear();
    }

    Ok(resolutions)
}

pub fn resolve_eager(
    root: TranslationUnit,
    path: &ModulePath,
    resolver: &impl Resolver,
) -> Result<Resolutions, E> {
    let mut resolutions = Resolutions::new();

    let module = Module::new(root, path.clone());

    let module = Rc::new(RefCell::new(module));
    resolutions.push_module(path.clone(), module.clone());

    fn resolve_module(
        module: &mut Module,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), E> {
        for (path, _) in module.imports.values() {
            if !resolutions.modules.contains_key(path) {
                let source = resolver.resolve_module(path)?;
                let module = Module::new(source, path.clone());
                let module = Rc::new(RefCell::new(module));
                resolutions.push_module(path.clone(), module.clone());
                resolve_module(module.borrow_mut().deref_mut(), resolutions, resolver)?;
            }
        }

        for ty in Visit::<TypeExpression>::visit_mut(&mut module.source) {
            let (ext_res, ext_id) = if let Some(path) = &ty.path {
                let res = resolve_inline_path(path, &module.path, &module.imports);
                (res, ty.ident.clone())
            } else if let Some((path, ident)) = module.imports.get(&ty.ident) {
                (path.clone(), ident.clone())
            } else {
                // points to a local decl, we stop here.
                continue;
            };

            // if the import path points to a local decl, we stop here
            if ext_res == module.path {
                if module.idents.contains_key(&ty.ident) {
                    continue;
                } else {
                    return Err(E::MissingDecl(ext_res, ty.ident.name().to_string()));
                }
            }

            // load the external module for this external ident
            let ext_mod = if let Some(module) = resolutions.modules.get(&ext_res) {
                module.clone()
            } else {
                let source = resolver.resolve_module(&ext_res)?;
                let module = Module::new(source, ext_res.clone());
                let module = Rc::new(RefCell::new(module));
                resolutions.push_module(ext_res.clone(), module.clone());
                resolve_module(module.borrow_mut().deref_mut(), resolutions, resolver)?;
                module
            };

            // get the ident of the external declaration pointed to by the type
            let ext_id = ext_mod
                .borrow() // safety: only 1 module is borrowed at a time, the current one.
                .idents
                .iter()
                .find(|(id, _)| *id.name() == *ext_id.name())
                .map(|(id, _)| id.clone())
                .ok_or_else(|| E::MissingDecl(ext_res.clone(), ext_id.to_string()))?;

            ty.path = None;
            ty.ident = ext_id;
        }

        Ok(())
    }

    resolve_module(module.borrow_mut().deref_mut(), &mut resolutions, resolver)?;

    Ok(resolutions)
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

fn mangle_decls<'a>(wgsl: &'a mut TranslationUnit, path: &'a ModulePath, mangler: &impl Mangler) {
    wgsl.global_declarations
        .iter_mut()
        .filter_map(|decl| decl.ident_mut())
        .for_each(|ident| {
            let new_name = mangler.mangle(path, &ident.name());
            ident.rename(new_name.clone());
        })
}

impl Resolutions {
    fn new() -> Self {
        Resolutions {
            modules: Default::default(),
            order: Default::default(),
        }
    }
    fn push_module(&mut self, path: ModulePath, module: Rc<RefCell<Module>>) {
        self.modules.insert(path.clone(), module);
        self.order.push(path);
    }
    pub fn mangle(&mut self, mangler: &impl Mangler) -> Result<(), E> {
        let root_path = self.root_path().clone();
        for (path, module) in self.modules.iter_mut() {
            if path != &root_path {
                let mut module = module.borrow_mut();
                mangle_decls(&mut module.source, path, mangler);
            }
        }
        Ok(())
    }

    pub fn assemble(&self, strip: bool) -> TranslationUnit {
        let mut wesl = TranslationUnit::default();
        for module in self.modules() {
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
                                    .is_some_and(|id| module.treated_idents.contains(id))
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
