//! Crystal module persistence: save/load `.mod` files to/from disk.

use crate::crystal::module::CrystalModule;
use std::fs;
use std::path::{Path, PathBuf};

/// Save a module to `<dir>/module_<id>.mod`. Returns the path written.
pub fn save_module(module: &CrystalModule, dir: &Path) -> std::io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("module_{:08}.mod", module.id));
    fs::write(&path, module.to_bytes())?;
    Ok(path)
}

/// Load every `.mod` file in `dir`, sorted by filename (i.e. by id).
/// Silently skips malformed files but logs them.
pub fn load_modules(dir: &Path) -> std::io::Result<Vec<CrystalModule>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "mod") {
            paths.push(path);
        }
    }
    paths.sort();

    let mut modules = Vec::with_capacity(paths.len());
    for path in paths {
        match fs::read(&path).and_then(|b| CrystalModule::from_bytes(&b)) {
            Ok(m) => modules.push(m),
            Err(e) => eprintln!("[crystal] skipping {:?}: {}", path, e),
        }
    }
    Ok(modules)
}
