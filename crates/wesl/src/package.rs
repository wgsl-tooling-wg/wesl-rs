use std::path::Path;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use wgsl_parse::syntax::{PathOrigin, TranslationUnit};

use crate::{Diagnostic, Error, ModulePath, SyntaxUtil, validate::validate_wesl};

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
///        .scan_directory("src/shaders")
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

pub struct Module {
    name: String,
    source: String,
    submodules: Vec<Module>,
}

impl PkgBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.replace('-', "_"),
        }
    }

    /// Reads all files in a directory to build the package.
    pub fn scan_directory(self, path: impl AsRef<Path>) -> std::io::Result<Module> {
        fn process_dir(lib_path: std::path::PathBuf) -> std::io::Result<Module> {
            let module_name = lib_path
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .replace('-', "_");

            let mut module = Module {
                name: module_name,
                source: String::new(),
                submodules: Vec::new(),
            };

            if lib_path.is_file() {
                // if path is wesl/wgsl use it as source
                let source_wesl = lib_path.with_extension("wesl");
                let source_wgsl = lib_path.with_extension("wgsl");
                if source_wesl.is_file() {
                    module.source = std::fs::read_to_string(source_wesl)?;
                } else if source_wgsl.is_file() {
                    module.source = std::fs::read_to_string(source_wgsl)?;
                }
            }

            if lib_path.is_dir() {
                // check if folder contains same named file
                let source_wesl = lib_path.join(&module.name).with_extension("wesl");
                let source_wgsl = lib_path.join(&module.name).with_extension("wgsl");

                if source_wesl.is_file() {
                    module.source = std::fs::read_to_string(&source_wesl)?
                } else if source_wgsl.is_file() {
                    module.source = std::fs::read_to_string(&source_wgsl)?
                }

                // add submodules
                for file in std::fs::read_dir(&lib_path)? {
                    let submodule_path = file?.path();

                    if submodule_path == source_wesl || submodule_path == source_wgsl {
                        continue;
                    }

                    if let Ok(submodule) = process_dir(submodule_path) {
                        module.submodules.push(submodule);
                    }
                }
            }

            Ok(module)
        }

        let mut module = Module {
            name: self.name.clone(),
            source: String::new(),
            submodules: Vec::new(),
        };

        let lib_path = path.as_ref().to_path_buf();

        // If path is directory, treat all files/dir as submodules
        // ignores files with same name as pkg name (see below)
        if lib_path.is_dir() {
            module.submodules = process_dir(lib_path.clone())?.submodules;
        }

        // If file with same name as pkg exist, use content as source
        let source_wesl = lib_path.join(self.name.clone()).with_extension("wesl");
        let source_wgsl = lib_path.join(self.name.clone()).with_extension("wgsl");
        if source_wesl.is_file() {
            module.source = std::fs::read_to_string(source_wesl)?
        } else if source_wgsl.is_file() {
            module.source = std::fs::read_to_string(source_wgsl)?
        }

        Ok(module)
    }
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
        let out_dir = Path::new(
            &std::env::var_os("OUT_DIR").expect("OUT_DIR environment variable is not defined"),
        )
        .join(format!("{}.rs", self.name));
        std::fs::write(&out_dir, code)?;
        Ok(())
    }
}
