use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use itertools::Itertools;
use wgsl_parse::{
    SyntaxNode,
    syntax::{
        GlobalDeclaration, Ident, Import, ImportContent, ImportItem, ImportStatement, ModulePath,
        PathOrigin, TranslationUnit, TypeExpression,
    },
};

use crate::{Diagnostic, Error, Mangler, ResolveError, Resolver, SyntaxUtil, visit::Visit};

type Imports = HashMap<Ident, ImportedItem>;
type Modules = HashMap<ModulePath, Rc<RefCell<Module>>>;

#[derive(Clone, Debug)]
struct ImportedItem {
    path: ModulePath,
    ident: Ident, // this is the ident's original name before `as` renaming.
    public: bool,
}

#[derive(Clone, Debug, Default)]
struct FlattenedImports {
    idents: Imports,
    wildcards: Vec<ModulePath>,
}

/// Error produced during import resolution.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ImportError {
    #[error("duplicate declaration of `{0}`")]
    DuplicateSymbol(String),
    #[error("{0}")]
    ResolveError(#[from] ResolveError),
    #[error("module `{0}` has no declaration `{1}`")]
    MissingDecl(ModulePath, String),
    #[error(
        "import of `{0}` in module `{1}` is not `@publish`, but another module tried to import it"
    )]
    Private(String, ModulePath),
}

type E = ImportError;

#[derive(Debug)]
pub(crate) struct Module {
    pub(crate) source: TranslationUnit,
    pub(crate) path: ModulePath,
    idents: HashMap<Ident, usize>,        // lookup (ident, decl_index)
    used_idents: RefCell<HashSet<Ident>>, // used idents that have already been usage-analyzed
    imports: FlattenedImports,
}

impl Module {
    pub(crate) fn new(source: TranslationUnit, path: ModulePath) -> Result<Self, E> {
        let idents = source
            .global_declarations
            .iter()
            .enumerate()
            .filter_map(|(i, decl)| decl.ident().map(|id| (id, i)))
            .collect::<HashMap<_, _>>();
        let imports = flatten_imports(&source.imports, &path)?;

        Ok(Self {
            source,
            path,
            idents,
            used_idents: Default::default(),
            imports,
        })
    }

    fn find_decl(&self, name: &str) -> Option<(&Ident, &usize)> {
        self.idents.iter().find(|(id, _)| *id.name() == name)
    }
    fn find_import(&self, name: &str) -> Option<(&Ident, &ImportedItem)> {
        self.imports
            .idents
            .iter()
            .find(|(id, _)| *id.name() == name)
    }
}

#[derive(Debug)]
pub(crate) struct Resolutions {
    modules: Modules,
    order: Vec<ModulePath>,
}

impl Resolutions {
    pub(crate) fn new(root_module: Module) -> Self {
        let mut resol = Resolutions {
            modules: Default::default(),
            order: Default::default(),
        };

        resol.push_module(root_module);
        resol
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

/// get or load a module with the resolver, while resolving idents and wildcard imports
fn load_module<R: Resolver>(
    path: &ModulePath,
    resolutions: &mut Resolutions,
    resolver: &R,
    onload: &impl Fn(&Module, &mut Resolutions, &R) -> Result<(), Error>,
) -> Result<Rc<RefCell<Module>>, Error> {
    if let Some(module) = resolutions.modules.get(path) {
        return Ok(module.clone());
    }

    let mut source = resolver.resolve_module(path)?;

    let imports = flatten_imports(&source.imports, path)?;

    // load wildcard imports.
    // we add an import statement containing all items after loading the ext module.
    // this guarantees that it behaves identically as importing all items.
    // in particular, `retarget_idents` will be able to track the used external idents.
    // it's hacky, but it works alright.
    for path in &imports.wildcards {
        let ext_mod = load_module(path, resolutions, resolver, onload)?;
        source.imports.push(ImportStatement {
            attributes: vec![],
            path: Some(path.clone()),
            content: ImportContent::Collection(
                ext_mod
                    .borrow()
                    .idents
                    .keys()
                    .map(|id| Import {
                        path: vec![],
                        content: ImportContent::Item(ImportItem {
                            ident: id.clone(),
                            rename: None,
                        }),
                    })
                    .collect(),
            ),
        });
    }

    source.retarget_idents();
    let module = Module::new(source, path.clone())?;
    let module = resolutions.push_module(module);

    {
        let module = module.borrow();
        onload(&module, resolutions, resolver)
            .map_err(|e| err_with_module(e, &module, resolver))?;
    }

    Ok(module)
}

// XXX: it's quite messy.
/// Load all modules "used" transitively by the root module. Make external idents point at
/// the right declaration in the external module.
///
/// It is "lazy" because external modules are loaded only if used by the `keep` declarations
/// or module-scope `const_assert`s.
///
/// This approach is only valid when stripping is enabled. Otherwise, unused declarations
/// may refer to declarations in unused modules, and mangling will panic.
///
/// "used": used declarations in the root module are the `keep` parameter. Used declarations
/// in other modules are those reached by `keep` declarations, recursively.
/// Module-scope `const_assert`s are always included.
///
/// Returns a list of [`Module`]s with the list of their "used" idents.
///
/// See also: [`resolve_eager`]
pub fn resolve_lazy<'a>(
    keep: impl IntoIterator<Item = &'a Ident>,
    root_module: Module,
    resolver: &impl Resolver,
) -> Result<Resolutions, Error> {
    fn resolve_module(
        module: &Module,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), Error> {
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

        Ok(())
    }

    fn resolve_ident(
        module: &Module,
        ident: &Ident,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), Error> {
        if let Some((_, n)) = module.find_decl(&ident.name()) {
            let decl = module.source.global_declarations.get(*n).unwrap();
            resolve_decl(module, decl, resolutions, resolver)
        } else if let Some((_, item)) = module.find_import(&ident.name()) {
            if item.public {
                // load the external module for this imported item
                let ext_mod = load_module(&item.path, resolutions, resolver, &resolve_module)?;
                let ext_mod = ext_mod.borrow();
                resolve_ident(&ext_mod, &item.ident, resolutions, resolver)
                    .map_err(|e| err_with_module(e, &ext_mod, resolver))
            } else {
                Err(E::Private(ident.to_string(), module.path.clone()).into())
            }
        } else {
            Err(E::MissingDecl(module.path.clone(), ident.to_string()).into())
        }
    }

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
            let path = resolve_inline_path(path, &module.path, &module.imports.idents);
            (path, ty.ident.clone())
        } else if let Some(item) = module.imports.idents.get(&ty.ident) {
            (item.path.clone(), item.ident.clone())
        } else {
            // points to a local decl, we stop here.
            if let Some(n) = module.idents.get(&ty.ident) {
                let decl = module.source.global_declarations.get(*n).unwrap();
                return resolve_decl(module, decl, resolutions, resolver);
            } else {
                return Ok(());
            };
        };

        // if the import path points to a local decl, we stop here
        if ext_path == module.path {
            return resolve_ident(module, &ext_id, resolutions, resolver);
        }

        // load the external module for this imported item
        let ext_mod = load_module(&ext_path, resolutions, resolver, &resolve_module)?;
        let ext_mod = ext_mod.borrow();
        resolve_ident(&ext_mod, &ext_id, resolutions, resolver)
            .map_err(|e| err_with_module(e, &ext_mod, resolver))
    }

    fn resolve_decl(
        module: &Module,
        decl: &GlobalDeclaration,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), Error> {
        if let Some(ident) = decl.ident() {
            if module.used_idents.borrow().contains(&ident) {
                return Ok(());
            }
            module.used_idents.borrow_mut().insert(ident);
        }

        for ty in Visit::<TypeExpression>::visit(decl) {
            resolve_ty(module, ty, resolutions, resolver)?;
        }

        Ok(())
    }

    let mut resolutions = Resolutions::new(root_module);
    let module = resolutions.root_module();

    {
        let module = module.borrow();
        resolve_module(&module, &mut resolutions, resolver)
            .map_err(|e| err_with_module(e, &module, resolver))?;

        for id in keep {
            resolve_ident(&module, id, &mut resolutions, resolver)
                .map_err(|e| err_with_module(e, &module, resolver))?;
        }
    }

    resolutions.retarget();
    Ok(resolutions)
}

/// Load all [`Module`]s referenced by the root module.
pub fn resolve_eager(root_module: Module, resolver: &impl Resolver) -> Result<Resolutions, Error> {
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
            let res = resolve_inline_path(path, &module.path, &module.imports.idents);
            (res, ty.ident.clone())
        } else if let Some(item) = module.imports.idents.get(&ty.ident) {
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

        // load the external module for this imported item
        let ext_mod = load_module(&ext_path, resolutions, resolver, &resolve_module)?;
        let ext_mod = ext_mod.borrow();

        // check if the imported item exists
        if ext_mod.find_decl(&ext_id.name()).is_none() {
            if let Some((_, item)) = ext_mod.find_import(&ext_id.name()) {
                if !item.public {
                    return Err(E::Private(ext_id.to_string(), ext_path.clone()).into());
                }
            } else {
                return Err(E::MissingDecl(ext_path.clone(), ext_id.to_string()).into());
            }
        }

        Ok(())
    }

    fn resolve_module(
        module: &Module,
        resolutions: &mut Resolutions,
        resolver: &impl Resolver,
    ) -> Result<(), Error> {
        // resolve all module imports
        for item in module.imports.idents.values() {
            load_module(&item.path, resolutions, resolver, &resolve_module)?;
        }

        for ty in Visit::<TypeExpression>::visit(&module.source) {
            resolve_ty(&module, ty, resolutions, resolver)?;
        }

        Ok(())
    }

    let mut resolutions = Resolutions::new(root_module);
    let module = resolutions.root_module();
    {
        let module = module.borrow();
        resolve_module(&module, &mut resolutions, resolver)
            .map_err(|e| err_with_module(e, &module, resolver))?;
    }
    resolutions.retarget();
    Ok(resolutions)
}

/// Flatten imports to a list of module paths.
// TODO: split into flatten_imports / flatten_wildcards which are not used at the same time.
fn flatten_imports(
    imports: &[ImportStatement],
    parent_path: &ModulePath,
) -> Result<FlattenedImports, E> {
    fn rec(
        content: &ImportContent,
        path: ModulePath,
        public: bool,
        res: &mut FlattenedImports,
    ) -> Result<(), E> {
        match content {
            ImportContent::Item(item) => {
                let ident = item.rename.as_ref().unwrap_or(&item.ident).clone();
                res.idents.insert(
                    ident,
                    ImportedItem {
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
            ImportContent::Wildcard => {
                res.wildcards.push(path);
            }
        }
        Ok(())
    }

    let mut res = FlattenedImports::default();

    for import in imports {
        let public = import.attributes.iter().any(|attr| attr.is_publish());
        match &import.path {
            Some(import_path) => {
                let path = parent_path.join_path(import_path);
                rec(&import.content, path, public, &mut res)?;
            }
            None => {
                // this covers two cases: `import foo;` and `import {foo, ..};`.
                // COMBAK: these edge-cases smell
                match &import.content {
                    ImportContent::Item(_) => {
                        // `import foo`, this import statement does nothing currently.
                        // In the future, it may become a visibility/re-export mechanism.
                    }
                    ImportContent::Collection(coll) => {
                        for import in coll {
                            let mut components = import.path.iter().cloned();
                            match components.next() {
                                Some(pkg_name) => {
                                    // `import {foo::bar}`, foo becomes the package name.
                                    let path = ModulePath::new(
                                        PathOrigin::Package(pkg_name),
                                        components.collect_vec(),
                                    );
                                    rec(&import.content, path, public, &mut res)?;
                                }
                                None => {
                                    // `import {foo}`, this does nothing, same as above.
                                }
                            }
                        }
                    }
                    ImportContent::Wildcard => {
                        // TODO
                        // `import *`, this should be invalid? We just ignore it for now.
                    }
                }
            }
        }
    }
    Ok(res)
}

/// Finds the normalized module path for an inline import.
///
/// Inline imports differ from import statements only in case of package imports:
/// the package component may refer to a local import shadowing the package name.
fn resolve_inline_path(
    path: &ModulePath,
    parent_path: &ModulePath,
    imports: &Imports,
) -> ModulePath {
    match &path.origin {
        PathOrigin::Package(pkg_name) => {
            // the path could be either a package, of referencing an imported module alias.
            let imported_item = imports.iter().find(|(ident, _)| *ident.name() == *pkg_name);

            if let Some((_, ext_item)) = imported_item {
                // this inline path references an imported item. Example:
                // import a::b::c as foo; foo::bar::baz() => a::b::c::bar::baz()
                let mut res = ext_item.path.clone(); // a::b
                res.push(&ext_item.ident.name()); // c
                res.join(path.components.iter().cloned())
            } else {
                parent_path.join_path(path)
            }
        }
        _ => parent_path.join_path(path),
    }
}

pub(crate) fn mangle_decls<'a>(
    wgsl: &'a mut TranslationUnit,
    path: &'a ModulePath,
    mangler: &impl Mangler,
) {
    wgsl.global_declarations
        .iter_mut()
        .filter_map(|decl| decl.ident())
        .for_each(|mut ident| {
            let new_name = mangler.mangle(path, &ident.name());
            ident.rename(new_name.clone());
        })
}

impl Resolutions {
    /// Retarget identifiers to point at the corresponding declaration.
    ///
    /// Panics
    /// * if an identifier has no corresponding declaration.
    /// * if a module is already borrowed.
    pub(crate) fn retarget(&mut self) {
        fn find_ext_ident(
            modules: &Modules,
            src_path: &ModulePath,
            src_id: &Ident,
        ) -> Option<Ident> {
            // load the external module for this external ident
            let module = modules.get(src_path)?;
            // SAFETY: since this is an external ident, it cannot be in the currently
            // borrowed module.
            let module = module.borrow();

            module
                .idents
                .iter()
                .find(|(id, _)| *id.name() == *src_id.name())
                .map(|(id, _)| id.clone())
                .or_else(|| {
                    // or it could be a re-exported import with `@publish`
                    module
                        .imports
                        .idents
                        .iter()
                        .find(|(id, _)| *id.name() == *src_id.name())
                        .and_then(|(_, item)| find_ext_ident(modules, &item.path, &item.ident))
                })
        }

        for module in self.modules.values() {
            let mut module = module.borrow_mut();
            let module = &mut *module;
            Visit::<TypeExpression>::visit_rec_mut(&mut module.source, &mut |ty| {
                let (ext_path, ext_id) = if let Some(path) = &ty.path {
                    let res = resolve_inline_path(path, &module.path, &module.imports.idents);
                    (res, ty.ident.clone())
                } else if let Some(item) = module.imports.idents.get(&ty.ident) {
                    (item.path.clone(), item.ident.clone())
                } else {
                    // points to a local decl, we stop here.
                    return;
                };

                // if the import path points to a local decl
                if ext_path == module.path {
                    let local_id = module
                        .idents
                        .iter()
                        .find(|(id, _)| *id.name() == *ext_id.name())
                        .map(|(id, _)| id.clone())
                        .expect("missing local declaration");
                    ty.path = None;
                    ty.ident = local_id;
                }
                // get the ident of the external declaration pointed to by the type
                else if let Some(ext_id) = find_ext_ident(&self.modules, &ext_path, &ext_id) {
                    ty.path = None;
                    ty.ident = ext_id;
                }
                // the ident has no declaration
                else {
                    eprintln!("could not find declaration for ident {ext_path}::{ext_id}");
                    panic!("missing declaration");
                }
            });
        }
    }

    /// Mangle all declarations in all modules. Should be called after [`Self::retarget`].
    ///
    /// Panics if a module is already borrowed.
    pub(crate) fn mangle(&mut self, mangler: &impl Mangler, mangle_root: bool) {
        let root_path = self.root_path().clone();
        for (path, module) in self.modules.iter_mut() {
            if mangle_root || path != &root_path {
                let mut module = module.borrow_mut();
                mangle_decls(&mut module.source, path, mangler);
            }
        }
    }

    /// Merge all declarations into a single module. If the `strip` flag is set, it will
    /// copy over only used declarations.
    pub(crate) fn assemble(&self, strip: bool) -> TranslationUnit {
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
                                    .is_some_and(|id| module.used_idents.borrow().contains(&id))
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
