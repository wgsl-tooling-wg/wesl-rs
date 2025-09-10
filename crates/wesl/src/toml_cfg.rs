#![cfg(feature = "serde")]
//! The WESL TOML file format.
//!
//! ```toml
//! [package]
//! # Version of WESL used in this project.
//! edition = "unstable_2025"
//!
//! # Optional, can be auto-inferred from the existence of a package manager file.
//! # Inclusion of this field is encouraged.
//! package-manager = "npm"
//!
//! # Where are the shaders located. This is the path of `package::`.
//! root = "./shaders"
//!
//! # Optional
//! include = [ "shaders/**/*.wesl", "shaders/**/*.wgsl" ]
//!
//! # Optional.
//! # Some projects have large folders that we shouldn't react to.
//! exclude = [ "**/test" ]
//!
//! # Lists all used dependencies
//! [dependencies]
//! # Shorthand for `foolib = { package = "foolib" }`
//! foolib = {}
//! # Can be used for renaming packages. Now bevy in my code is called "cute_bevy".
//! cute_bevy = { package = "bevy" }
//! # File path to a folder with a wesl.toml. Simplest kind of dependency.
//! mylib = { path = "../mylib" }
//! ```

use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename = "snake_case")]
pub enum WeslEdition {
    Unstable2025,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, Serialize, Deserialize)]
#[serde(rename = "snake_case")]
pub enum PackageManager {
    // TODO: the spec says: "can be inferred from the existence of certain files"
    #[default]
    Cargo,
    Npm,
}

fn default_root() -> PathBuf {
    PathBuf::from("shaders")
}

fn default_include() -> Vec<String> {
    vec!["**/*.wesl".to_string(), "**/*.wgsl".to_string()]
}

fn default_exclude() -> Vec<String> {
    vec![]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WeslTomlPackage {
    edition: WeslEdition,
    #[serde(default)]
    package_manager: PackageManager,
    #[serde(default = "default_root")]
    root: PathBuf,
    #[serde(default = "default_include")]
    include: Vec<String>,
    #[serde(default = "default_exclude")]
    exclude: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum WeslTomlDependency {
    Auto {},
    Package { package: String },
    Path { path: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum WeslTomlDependencies {
    Auto,
    Manual(HashMap<String, WeslTomlDependency>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WeslToml {
    package: WeslTomlPackage,
    dependencies: WeslTomlDependencies,
}
