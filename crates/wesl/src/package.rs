use std::path::Path;

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
    dependencies: Vec<&'static crate::Pkg>,
}

pub struct Pkg {
    crate_name: String,
    root: Module,
    dependencies: Vec<&'static crate::Pkg>,
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

    /// Reads all files in a directory to build the package.
    pub fn scan_directory(self, path: impl AsRef<Path>) -> std::io::Result<Pkg> {
        let dir = path.as_ref().to_path_buf();
        // we look for a file with the same name as the dir in the same directory
        let mut lib_path = dir.clone();
        lib_path.set_extension("wesl");

        let source = if lib_path.is_file() {
            std::fs::read_to_string(&lib_path)?
        } else {
            lib_path.set_extension("wgsl");
            if lib_path.is_file() {
                std::fs::read_to_string(&lib_path)?
            } else {
                String::from("")
            }
        };

        let mut root = Module {
            name: self.name.clone(),
            source,
            submodules: Vec::new(),
        };

        fn process_dir(module: &mut Module, dir: &Path) -> std::io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file()
                    && path
                        .extension()
                        .is_some_and(|ext| ext == "wesl" || ext == "wgsl")
                {
                    let source = std::fs::read_to_string(&path)?;
                    let name = path
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .replace('-', "_");
                    // we look for a dir with the same name as the file in the same directory
                    let mut subdir = dir.to_path_buf();
                    subdir.push(&name);

                    let mut submod = Module {
                        name,
                        source,
                        submodules: Vec::new(),
                    };

                    if subdir.is_dir() {
                        process_dir(&mut submod, &subdir)?;
                    }

                    module.submodules.push(submod);
                }
            }

            Ok(())
        }

        if dir.is_dir() {
            process_dir(&mut root, &dir)?;
        }

        let crate_name = std::env::var("CARGO_PKG_NAME")
            .expect("CARGO_PKG_NAME environment variable is not defined")
            .to_string();

        Ok(Pkg {
            crate_name,
            root,
            dependencies: self.dependencies,
        })
    }
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
        let out_dir = Path::new(
            &std::env::var_os("OUT_DIR").expect("OUT_DIR environment variable is not defined"),
        )
        .join(format!("{}.rs", self.root.name));
        std::fs::write(&out_dir, code)?;
        Ok(())
    }
}
