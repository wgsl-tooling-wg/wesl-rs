use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use crate::{Diagnostic, Error, ModulePath, SyntaxUtil, validate::validate_wesl};
use quote::{format_ident, quote};
use wgsl_parse::syntax::{PathOrigin, TranslationUnit};

/// A builder that generates code for WESL packages.
///
/// It is designed to be used in a build script (`build.rs` file). Add `wesl` to the
/// build-dependencies of your project and enable the `package` feature flag.
///
/// ```ignore
/// // in build.rs
/// fn main() {
///    wesl::PkgBuilder::new("my_package")
///        // read all wesl files in the directory "src/shaders"
///        .scan_root("src/shaders")
///        .expect("failed to scan WESL files")
///        // validation is optional
///        .validate()
///        .map_err(|e| eprintln!("{e}"))
///        .expect("validation error")
///        // write "my_package.rs" in OUT_DIR
///        .build_artifact()
///        .expect("failed to build artifact");
/// }
/// ```
/// Then, in your `lib.rs` file, expose the generated module with the [`crate::wesl_pkg`] macro.
/// ```ignore
/// // in src/lib.rs
/// wesl::wesl_pkg!(my_package);
/// ```
///
/// The package name must be a valid rust identifier, E.g. it must not contain dashes `-`.
/// Dashes are replaced with underscores `_`.
pub struct PkgBuilder {
    name: String,
    dependencies: Vec<&'static crate::Pkg>,
}

pub struct Pkg {
    crate_name: String,
    root: Module,
    dependencies: Vec<&'static crate::Pkg>,
}

#[derive(Debug, thiserror::Error)]
pub enum ScanDirectoryError {
    #[error("Package root was not found: `{0}`")]
    RootNotFound(PathBuf),
    #[error("I/O error while scanning package root: {0}")]
    Io(#[from] std::io::Error),
}

impl PkgBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.replace('-', "_"),
            dependencies: Vec::new(),
        }
    }

    /// Add a package dependency.
    ///
    /// Learn more about packages in [`PkgBuilder`].
    pub fn add_package(mut self, pkg: &'static crate::Pkg) -> Self {
        self.dependencies.push(pkg);
        self
    }

    /// Add several package dependencies.
    ///
    /// Learn more about packages in [`PkgBuilder`].
    pub fn add_packages(mut self, pkgs: impl IntoIterator<Item = &'static crate::Pkg>) -> Self {
        for pkg in pkgs {
            self = self.add_package(pkg);
        }
        self
    }

    /// Reads all files to include in the package, starting from the root module.
    ///
    /// The input path must point at the root file or folder. The package will include
    /// all .wesl and .wgsl files reachable from the root module, recursively.
    /// The name or the root file is ignored, instead the name of the package is used.
    pub fn scan_root(self, path: impl AsRef<Path>) -> Result<Pkg, ScanDirectoryError> {
        fn process_path(path: &Path) -> Result<Option<Module>, std::io::Error> {
            let path_with_ext_wesl = path.with_extension("wesl");
            let path_with_ext_wgsl = path.with_extension("wgsl");
            let path_without_ext = path.with_extension("");

            // check for source file
            let source = if path_with_ext_wesl.is_file() {
                std::fs::read_to_string(&path_with_ext_wesl)?
            } else if path_with_ext_wgsl.is_file() {
                std::fs::read_to_string(&path_with_ext_wgsl)?
            } else {
                String::new()
            };

            // check for submodules
            let mut submodules = Vec::new();
            if path_without_ext.is_dir() {
                // use hashset to avoid duplicate entries
                let mut unique_submodules = HashSet::new();
                for entry in std::fs::read_dir(&path_without_ext)? {
                    let Ok(entry) = entry else { continue };
                    let submodule_path = entry.path().with_extension("");
                    unique_submodules.insert(submodule_path);
                }
                for entry in unique_submodules {
                    // errors in the top module should be returned
                    // other errors should only be logged
                    match process_path(&entry) {
                        Ok(Some(module)) => submodules.push(module),
                        Ok(None) => {
                            eprintln!("INFO: found non shader/dir at {:?}: ignoring", entry)
                        }
                        Err(err) => {
                            eprintln!("WARN: error processing submodule {:?}: {}", entry, err)
                        }
                    }
                }
            };

            // check for empty module
            if source.is_empty() && submodules.is_empty() {
                return Ok(None);
            }

            let path_filename = path_without_ext
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .replace('-', "_");
            let module = Module {
                name: path_filename,
                source,
                submodules,
            };

            Ok(Some(module))
        }

        let root_path = path.as_ref().to_path_buf();
        let potential_module = process_path(&root_path)?;
        let Some(mut module) = potential_module else {
            return Err(ScanDirectoryError::RootNotFound(root_path));
        };
        // top level module should be named by package builder and not file path
        module.name = self.name;

        let crate_name = std::env::var("CARGO_PKG_NAME")
            .expect("CARGO_PKG_NAME environment variable is not defined")
            .to_string();

        Ok(Pkg {
            crate_name,
            root: module,
            dependencies: self.dependencies,
        })
    }
}

#[derive(Debug)]
pub struct Module {
    name: String,
    source: String,
    submodules: Vec<Module>,
}

impl Module {
    fn codegen(&self) -> proc_macro2::TokenStream {
        let mod_ident = format_ident!("{}", self.name);
        let name = &self.name;
        let source = &self.source;

        let submodules = self.submodules.iter().map(|submod| {
            let name = &submod.name;
            let ident = format_ident!("{}", name);
            quote! { &#ident::MODULE }
        });

        let submods = self.submodules.iter().map(|submod| submod.codegen());

        quote! {
            #[allow(clippy::all)]
            pub mod #mod_ident {
                use super::PkgModule;
                pub const MODULE: PkgModule = PkgModule {
                    name: #name,
                    source: #source,
                    submodules: &[#(#submodules),*]
                };

                #(#submods)*
            }
        }
    }

    fn validate(&self, path: ModulePath) -> Result<(), Error> {
        let mut wesl: TranslationUnit = self.source.parse().map_err(|e| {
            Diagnostic::from(e)
                .with_module_path(path.clone(), None)
                .with_source(self.source.clone())
        })?;
        wesl.retarget_idents();
        validate_wesl(&wesl)?;
        for module in &self.submodules {
            let mut path = path.clone();
            path.push(&self.name);
            module.validate(path)?;
        }
        Ok(())
    }
}

impl Pkg {
    /// generate the rust code that holds the packaged wesl files.
    /// you probably want to use [`Self::build_artifact`] instead.
    pub fn codegen(&self) -> std::io::Result<String> {
        let deps = self.dependencies.iter().map(|dep| {
            let crate_name = format_ident!("{}", dep.crate_name);
            let mod_name = format_ident!("{}", dep.root.name);
            quote! { &#crate_name::#mod_name::PACKAGE }
        });

        let crate_name = &self.crate_name;
        let root = format_ident!("{}", self.root.name);
        let root_mod = self.root.codegen();

        let tokens = quote! {
            pub const PACKAGE: Pkg = Pkg {
                crate_name: #crate_name,
                root: &#root::MODULE,
                dependencies: &[#(#deps),*],
            };

            #root_mod
        };

        Ok(tokens.to_string())
    }

    /// run validation checks on each of the scanned files.
    pub fn validate(self) -> Result<Self, Error> {
        let path = ModulePath::new(PathOrigin::Absolute, vec![self.root.name.clone()]);
        self.root.validate(path)?;
        Ok(self)
    }

    /// generate the build artifact that can then be exposed by the [`super::wesl_pkg`] macro.
    ///
    /// this function must be called from a `build.rs` file. Refer to the crate documentation
    /// for more details.
    ///
    /// # Panics
    /// panics if the OUT_DIR environment variable is not set. This should not happen if
    /// ran from a `build.rs` file.
    pub fn build_artifact(&self) -> std::io::Result<()> {
        let code = self.codegen()?;
        let out_dir = std::path::Path::new(
            &std::env::var_os("OUT_DIR").expect("OUT_DIR environment variable is not defined"),
        )
        .join(format!("{}.rs", self.root.name));
        std::fs::write(&out_dir, code)?;
        Ok(())
    }
}
