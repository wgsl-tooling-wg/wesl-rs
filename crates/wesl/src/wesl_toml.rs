//! wesl.toml configuration parsing and file scanning.
//!
//! This module handles reading wesl.toml configuration files and building
//! module hierarchies from glob patterns.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::package::{Module, RESERVED_MOD_NAMES, is_mod_ident};

/// Parsed wesl.toml configuration.
#[derive(Debug)]
pub struct WeslToml {
    /// Package configuration.
    pub package: PackageConfig,
    /// The [dependencies] section.
    pub dependencies: DependencyConfig,
}

/// Package configuration fields.
#[derive(Debug)]
pub struct PackageConfig {
    /// WESL edition (required).
    pub edition: String,
    /// Package manager: "npm" or "cargo". Auto-detected if not specified.
    pub package_manager: Option<String>,
    /// Root folder for package:: syntax. Default: "./shaders/"
    pub root: String,
    /// Glob patterns for files to include. Default: all .wesl/.wgsl in root.
    pub include: Option<Vec<String>>,
    /// Glob patterns for files to exclude. Default: empty.
    pub exclude: Vec<String>,
}

/// The [dependencies] section of wesl.toml.
#[derive(Debug, Default, Deserialize)]
#[serde(untagged)]
pub enum DependencyConfig {
    /// No dependencies specified.
    #[default]
    None,
    /// Automatic dependency detection: `dependencies = "auto"`
    Auto(String),
    /// Explicit dependency map.
    Map(HashMap<String, DependencySpec>),
}

/// A single dependency specification.
#[derive(Debug, Deserialize)]
pub struct DependencySpec {
    /// Package name (for renaming): `{ package = "actual_name" }`
    #[serde(default)]
    pub package: Option<String>,
    /// Local path: `{ path = "../lib" }`
    #[serde(default)]
    pub path: Option<String>,
}

/// Warning emitted during file scanning (non-fatal).
#[derive(Debug, Clone)]
pub enum ScanWarning {
    /// A path component is not a valid WGSL identifier; file was skipped.
    InvalidIdentifier { component: String, file: PathBuf },
    /// A path component is a reserved module name; file was skipped.
    ReservedModName { name: String, file: PathBuf },
}

impl std::fmt::Display for ScanWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidIdentifier { component, file } => {
                write!(
                    f,
                    "skipped file: component `{component}` is not a valid WGSL identifier in {file:?}"
                )
            }
            Self::ReservedModName { name, file } => {
                write!(
                    f,
                    "skipped file: module name `{name}` is reserved in {file:?}"
                )
            }
        }
    }
}

/// Result of scanning files, including any non-fatal warnings.
#[derive(Debug)]
pub struct ScanResult {
    /// The root module containing the scanned file hierarchy.
    pub module: Module,
    /// Warnings encountered during scanning.
    pub warnings: Vec<ScanWarning>,
}

#[derive(Debug, thiserror::Error)]
pub enum ScanTomlError {
    #[error("wesl.toml not found at `{0}`")]
    TomlNotFound(PathBuf),
    #[error("Failed to parse wesl.toml: {0}")]
    TomlParse(#[from] toml::de::Error),
    #[error("missing required field `edition`")]
    MissingEdition,
    #[error("Invalid glob pattern `{0}`: {1}")]
    InvalidGlob(String, glob::PatternError),
    #[error("File `{0}` is outside root `{1}`")]
    FileOutsideRoot(PathBuf, PathBuf),
    #[error("I/O error: {0}")]
    Io(std::io::Error),
    #[error("No source files matched the include patterns")]
    NoFilesMatched,
    #[error("Multiple files map to module `{0}`: {1:?}")]
    ConflictingFiles(String, Vec<PathBuf>),
}

// Raw deserialization type for wesl.toml
#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct RawWeslToml {
    // Package fields at root level (Option<T> is already optional in serde)
    edition: Option<String>,
    package_manager: Option<String>,
    root: Option<String>,
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
    // [dependencies] section
    #[serde(default)]
    dependencies: DependencyConfig,
}

impl WeslToml {
    /// Parse a wesl.toml file from a path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ScanTomlError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(ScanTomlError::Io)?;
        Self::parse_str(&content)
    }

    /// Parse a wesl.toml from string content.
    pub fn parse_str(content: &str) -> Result<Self, ScanTomlError> {
        let raw: RawWeslToml = toml::from_str(content).map_err(ScanTomlError::TomlParse)?;

        let edition = raw.edition.ok_or(ScanTomlError::MissingEdition)?;
        let root = raw.root.unwrap_or_else(|| "./shaders/".to_string());
        let exclude = raw.exclude.unwrap_or_default();

        Ok(WeslToml {
            package: PackageConfig {
                edition,
                package_manager: raw.package_manager,
                root,
                include: raw.include,
                exclude,
            },
            dependencies: raw.dependencies,
        })
    }
}

/// Scan files based on a WeslToml and build a Module hierarchy.
///
/// This is the main entry point called by `PkgBuilder::scan_toml`.
pub fn scan_from_config(
    name: &str,
    base_dir: &Path,
    config: &WeslToml,
) -> Result<ScanResult, ScanTomlError> {
    let root_str = config.package.root.trim_start_matches("./");
    let root_path = base_dir.join(root_str);

    let include_patterns = config
        .package
        .include
        .clone()
        .unwrap_or_else(|| vec![format!("{}/**/*.w[eg]sl", root_str)]);
    let matched_files = collect_glob_filtered(base_dir, &include_patterns, |p| p.is_file())?;
    let exclude_paths = collect_glob_filtered(base_dir, &config.package.exclude, |_| true)?;

    let matched_files: HashSet<_> = matched_files
        .into_iter()
        .filter(|file| !is_excluded(file, &exclude_paths))
        .collect();

    if matched_files.is_empty() {
        return Err(ScanTomlError::NoFilesMatched);
    }

    let (module, warnings) = build_module_hierarchy(name, &matched_files, &root_path)?;
    Ok(ScanResult { module, warnings })
}

/// Collect paths matching glob patterns, filtered by a predicate.
fn collect_glob_filtered(
    base_dir: &Path,
    patterns: &[String],
    filter: impl Fn(&Path) -> bool,
) -> Result<HashSet<PathBuf>, ScanTomlError> {
    let mut paths = HashSet::new();

    for pattern in patterns {
        let pattern_path = pattern
            .strip_prefix("./")
            .map_or_else(|| base_dir.join(pattern), |s| base_dir.join(s));

        let pattern_str = pattern_path.to_string_lossy();
        let glob_iter =
            glob::glob(&pattern_str).map_err(|e| ScanTomlError::InvalidGlob(pattern.clone(), e))?;

        paths.extend(glob_iter.flatten().filter(|p| filter(p)));
    }

    Ok(paths)
}

/// Check if a file should be excluded.
///
/// A file is excluded if it matches an exclude path directly,
/// or if it's inside an excluded directory.
fn is_excluded(file: &Path, exclude_paths: &HashSet<PathBuf>) -> bool {
    exclude_paths.contains(file)
        || exclude_paths
            .iter()
            .any(|excl| excl.is_dir() && file.starts_with(excl))
}

struct FileEntry {
    path: PathBuf,
    module_components: Vec<String>,
}

/// Build a Module hierarchy from a flat list of files.
fn build_module_hierarchy(
    root_name: &str,
    files: &HashSet<PathBuf>,
    root_path: &Path,
) -> Result<(Module, Vec<ScanWarning>), ScanTomlError> {
    let (entries, warnings) = derive_module_paths(files, root_path)?;
    let module = build_module_tree(root_name, entries)?;
    Ok((module, warnings))
}

/// Extract module path components from a relative file path.
///
/// Combines the parent directory components with the file stem.
fn path_to_components(relative: &Path) -> Vec<String> {
    let mut components = Vec::new();
    if let Some(parent) = relative.parent() {
        components.extend(parent.iter().map(|c| c.to_string_lossy().to_string()));
    }
    if let Some(stem) = relative.file_stem() {
        components.push(stem.to_string_lossy().to_string());
    }
    components
}

/// Validate module path components.
///
/// Returns `None` if valid, or `Some(warning)` if the file should be skipped.
fn validate_components(components: &[String], file_path: &Path) -> Option<ScanWarning> {
    for comp in components {
        if RESERVED_MOD_NAMES.contains(&comp.as_str()) {
            return Some(ScanWarning::ReservedModName {
                name: comp.clone(),
                file: file_path.to_path_buf(),
            });
        }
        if !is_mod_ident(comp) {
            return Some(ScanWarning::InvalidIdentifier {
                component: comp.clone(),
                file: file_path.to_path_buf(),
            });
        }
    }
    None
}

/// Derive module path components from file paths by stripping the root prefix.
fn derive_module_paths(
    files: &HashSet<PathBuf>,
    root_path: &Path,
) -> Result<(Vec<FileEntry>, Vec<ScanWarning>), ScanTomlError> {
    let canonical_root = root_path
        .canonicalize()
        .unwrap_or_else(|_| root_path.to_path_buf());

    let mut entries = Vec::new();
    let mut warnings = Vec::new();
    for file_path in files {
        let canonical_file = file_path
            .canonicalize()
            .unwrap_or_else(|_| file_path.clone());

        let relative = canonical_file.strip_prefix(&canonical_root).map_err(|_| {
            ScanTomlError::FileOutsideRoot(file_path.clone(), root_path.to_path_buf())
        })?;

        let components = path_to_components(relative);

        if !components.is_empty() {
            if let Some(warning) = validate_components(&components, file_path) {
                warnings.push(warning);
            } else {
                entries.push(FileEntry {
                    path: file_path.clone(),
                    module_components: components,
                });
            }
        }
    }

    Ok((entries, warnings))
}

/// Intermediate tree node used while building the module hierarchy.
///
/// The `path` field tracks which file was assigned to detect conflicts
/// (e.g., both `main.wesl` and `main.wgsl` mapping to the same module).
struct ModuleNode {
    path: Option<PathBuf>,
    source: String,
    children: HashMap<String, ModuleNode>,
}

impl ModuleNode {
    fn new() -> Self {
        Self {
            path: None,
            source: String::new(),
            children: HashMap::new(),
        }
    }

    fn into_module(self, name: String) -> Module {
        let submodules = self
            .children
            .into_iter()
            .map(|(name, node)| node.into_module(name))
            .collect();

        Module {
            name,
            source: self.source,
            submodules,
        }
    }
}

/// Build a tree of Modules from flat file entries.
fn build_module_tree(root_name: &str, entries: Vec<FileEntry>) -> Result<Module, ScanTomlError> {
    let mut root = ModuleNode::new();

    for entry in entries {
        let Some((leaf, parents)) = entry.module_components.split_last() else {
            continue;
        };

        // Traverse/create intermediate nodes for parent components
        let mut current = &mut root;
        for comp in parents {
            current = current
                .children
                .entry(comp.clone())
                .or_insert_with(ModuleNode::new);
        }

        // Create/update the leaf node for the actual module
        let node = current
            .children
            .entry(leaf.clone())
            .or_insert_with(ModuleNode::new);

        if let Some(existing_path) = &node.path {
            let module_name = entry.module_components.join("::");
            return Err(ScanTomlError::ConflictingFiles(
                module_name,
                vec![existing_path.clone(), entry.path.clone()],
            ));
        }

        node.path = Some(entry.path.clone());
        node.source = std::fs::read_to_string(&entry.path).map_err(ScanTomlError::Io)?;
    }

    Ok(root.into_module(root_name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_config_parsing() {
        // Basic config with defaults
        let basic = WeslToml::parse_str("edition = \"2026_pre\"").unwrap();
        assert_eq!(basic.package.edition, "2026_pre");
        assert_eq!(basic.package.root, "./shaders/");

        // Config with custom root
        let with_root = WeslToml::parse_str("edition = \"2026_pre\"\nroot = \"./src/\"").unwrap();
        assert_eq!(with_root.package.edition, "2026_pre");
        assert_eq!(with_root.package.root, "./src/");

        // Missing edition
        let missing = WeslToml::parse_str("root = \"./shaders/\"");
        assert!(matches!(missing, Err(ScanTomlError::MissingEdition)));

        // Dependencies variants
        let with_deps =
            WeslToml::parse_str("edition = \"2026_pre\"\n\n[dependencies]\nfoo = {}").unwrap();
        assert!(matches!(with_deps.dependencies, DependencyConfig::Map(_)));

        let auto_deps =
            WeslToml::parse_str("edition = \"2026_pre\"\ndependencies = \"auto\"").unwrap();
        assert!(matches!(auto_deps.dependencies, DependencyConfig::Auto(_)));
    }

    #[test]
    fn test_scan_from_config() {
        let temp_dir = tempfile::tempdir().unwrap();
        let base = temp_dir.path();

        fs::create_dir_all(base.join("shaders/utils")).unwrap();
        fs::write(base.join("shaders/main.wesl"), "// main").unwrap();
        fs::write(base.join("shaders/utils/math.wesl"), "fn add() {}").unwrap();

        let config =
            WeslToml::parse_str("edition = \"2026_pre\"\nroot = \"./shaders/\"").unwrap();
        let result = scan_from_config("my_pkg", base, &config).unwrap();

        assert_eq!(result.module.name, "my_pkg");
        assert_eq!(result.module.submodules.len(), 2);
        assert!(result.warnings.is_empty());

        let main_mod = result
            .module
            .submodules
            .iter()
            .find(|m| m.name == "main")
            .unwrap();
        assert_eq!(main_mod.source, "// main");

        let utils = result
            .module
            .submodules
            .iter()
            .find(|m| m.name == "utils")
            .unwrap();
        assert_eq!(utils.submodules[0].name, "math");
    }

    #[test]
    fn test_conflicting_files_error() {
        let temp_dir = tempfile::tempdir().unwrap();
        let base = temp_dir.path();

        fs::create_dir_all(base.join("shaders")).unwrap();
        fs::write(base.join("shaders/main.wesl"), "// wesl").unwrap();
        fs::write(base.join("shaders/main.wgsl"), "// wgsl").unwrap();

        let config =
            WeslToml::parse_str("edition = \"2026_pre\"\nroot = \"./shaders/\"").unwrap();
        let result = scan_from_config("my_pkg", base, &config);

        assert!(matches!(result, Err(ScanTomlError::ConflictingFiles(_, _))));
    }

    #[test]
    fn test_exclude_directory() {
        let temp_dir = tempfile::tempdir().unwrap();
        let base = temp_dir.path();

        fs::create_dir_all(base.join("shaders/test")).unwrap();
        fs::write(base.join("shaders/main.wesl"), "// main").unwrap();
        fs::write(base.join("shaders/test/fixture.wesl"), "// fixture").unwrap();

        // Exclude the test directory
        let config = WeslToml::parse_str(
            r#"
            edition = "2026_pre"
            root = "./shaders/"
            exclude = ["**/test"]
            "#,
        )
        .unwrap();

        let result = scan_from_config("my_pkg", base, &config).unwrap();

        // Should only have main, not the test/fixture
        assert_eq!(result.module.submodules.len(), 1);
        assert_eq!(result.module.submodules[0].name, "main");
    }
}
