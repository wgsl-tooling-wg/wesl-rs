use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use crate::{Diagnostic, Error, ModulePath, SyntaxUtil, validate::validate_wesl};
use proc_macro2::TokenStream;
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
        }
    }

    /// Reads all files to include in the package, starting from the root module.
    ///
    /// The input path must point at the root file or folder. The package will include
    /// all .wesl and .wgsl files reachable from the root module, recursively.
    /// The name or the root file is ignored, instead the name of the package is used.
    pub fn scan_root(self, path: impl AsRef<Path>) -> Result<Module, ScanDirectoryError> {
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

        Ok(module)
    }
}

#[derive(Debug)]
pub struct Module {
    name: String,
    source: String,
    submodules: Vec<Module>,
}

impl Module {
    /// generate the rust code that holds the packaged wesl files.
    /// you probably want to use [`Self::build`] instead.
    pub fn codegen(&self) -> std::io::Result<String> {
        fn codegen_module(module: &Module) -> TokenStream {
            let name = &module.name;
            let source = &module.source;

            let submodules = module.submodules.iter().map(|submod| {
                let name = &submod.name;
                let ident = format_ident!("{}", name);
                quote! {
                    &#ident::Mod,
                }
            });

            let match_arms = module.submodules.iter().map(|submod| {
                let name = &submod.name;
                let ident = format_ident!("{}", name);
                quote! {
                    #name => Some(&#ident::Mod),
                }
            });

            let subquotes = module.submodules.iter().map(|submod| {
                let ident = format_ident!("{}", submod.name);
                let module = codegen_module(submod);
                quote! {
                    pub mod #ident {
                        use super::PkgModule;
                        #module
                    }
                }
            });

            quote! {
                pub struct Mod;

                impl PkgModule for Mod {
                    fn name(&self) -> &'static str {
                        #name
                    }
                    fn source(&self) -> &'static str {
                        #source
                    }
                    fn submodules(&self) -> &[&dyn PkgModule] {
                        static SUBMODULES: &[&dyn PkgModule] = &[
                            #(#submodules)*
                        ];
                        SUBMODULES
                    }
                    fn submodule(&self, name: &str) -> Option<&'static dyn PkgModule> {
                        #[allow(clippy::match_single_binding)]
                        match name {
                            #(#match_arms)*
                            _ => None,
                        }
                    }
                }

                #(#subquotes)*
            }
        }

        let tokens = codegen_module(self);
        Ok(tokens.to_string())
    }

    /// run validation checks on each of the scanned files.
    pub fn validate(self) -> Result<Self, Error> {
        fn validate_module(module: &Module, path: ModulePath) -> Result<(), Error> {
            let mut wesl: TranslationUnit = module.source.parse().map_err(|e| {
                Diagnostic::from(e)
                    .with_module_path(path.clone(), None)
                    .with_source(module.source.clone())
            })?;
            wesl.retarget_idents();
            validate_wesl(&wesl)?;
            for module in &module.submodules {
                let mut path = path.clone();
                path.push(&module.name);
                validate_module(module, path)?;
            }
            Ok(())
        }
        let path = ModulePath::new(PathOrigin::Package, vec![self.name.clone()]);
        validate_module(&self, path)?;
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
        .join(format!("{}.rs", self.name));
        std::fs::write(&out_dir, code)?;
        Ok(())
    }
}
