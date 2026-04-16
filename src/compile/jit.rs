//! JIT module loader — dlopen compiled modules and hot-swap them.

use std::path::Path;

/// A compiled module loaded via dlopen.
pub struct CompiledModule {
    /// The dlopen handle.
    _lib: *mut libc::c_void,
    /// Function pointer: extern "C" fn(input: *const f32, output: *mut f32, len: usize)
    forward_fn: unsafe extern "C" fn(*const f32, *mut f32, usize),
    /// Path to the .so file.
    pub path: std::path::PathBuf,
    /// Module ID.
    pub module_id: u64,
}

// Safety: the function pointer and lib handle are valid for the lifetime of the struct.
// The .so file must remain on disk.
unsafe impl Send for CompiledModule {}
unsafe impl Sync for CompiledModule {}

impl CompiledModule {
    /// Load a compiled module from a .so file.
    pub fn load(path: &Path, module_id: u64) -> Result<Self, String> {
        let c_path = std::ffi::CString::new(path.to_str().unwrap())
            .map_err(|e| format!("invalid path: {}", e))?;

        unsafe {
            let lib = libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW);
            if lib.is_null() {
                let err = libc::dlerror();
                let msg = if err.is_null() {
                    "unknown error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned()
                };
                return Err(format!("dlopen failed: {}", msg));
            }

            let sym_name = std::ffi::CString::new("module_forward").unwrap();
            let sym = libc::dlsym(lib, sym_name.as_ptr());
            if sym.is_null() {
                libc::dlclose(lib);
                return Err("dlsym(module_forward) returned null".into());
            }

            let forward_fn: unsafe extern "C" fn(*const f32, *mut f32, usize) =
                std::mem::transmute(sym);

            Ok(Self {
                _lib: lib,
                forward_fn,
                path: path.to_path_buf(),
                module_id,
            })
        }
    }

    /// Execute the compiled module.
    ///
    /// # Safety
    /// input must point to at least `len` f32 values.
    /// output must point to at least `len` f32 values.
    pub unsafe fn forward(&self, input: &[f32], output: &mut [f32]) {
        (self.forward_fn)(input.as_ptr(), output.as_mut_ptr(), input.len());
    }
}

impl Drop for CompiledModule {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self._lib);
        }
    }
}

/// Compile a crystal module to native code and load it.
pub fn compile_and_load(
    ternary_mat: &crate::tensor::ternary::TernaryMatrix,
    module_id: u64,
    modules_dir: &Path,
) -> Result<CompiledModule, String> {
    // Lower to IR
    let ir = crate::compile::lower::lower_ternary(ternary_mat);
    eprintln!("[compile] {}", ir.stats());

    // Emit x86-64
    let machine_code = crate::compile::x86::emit_x86(&ir);
    eprintln!("[compile] Emitted {} bytes of x86-64", machine_code.text_size);

    // Write ELF .so
    std::fs::create_dir_all(modules_dir)
        .map_err(|e| format!("mkdir: {}", e))?;
    let so_path = modules_dir.join(format!("module_{}.so", module_id));
    crate::compile::elf::write_elf(&machine_code, &so_path)
        .map_err(|e| format!("write_elf: {}", e))?;

    eprintln!("[compile] Wrote {:?}", so_path);

    // dlopen
    CompiledModule::load(&so_path, module_id)
}
