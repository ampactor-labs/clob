//! Build script: capture the git commit hash into CLOB_GIT_COMMIT so the
//! runtime manifest can record it. Falls back to "unknown" if the command
//! fails (e.g., building from a tarball outside a git repo).

use std::process::Command;

fn main() {
    let commit = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .and_then(|out| if out.status.success() {
            Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
        } else {
            None
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=CLOB_GIT_COMMIT={}", commit);
    // Re-run if HEAD moves. This is best-effort — `HEAD` inside `.git` is
    // what changes on branch switches or commits.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads");
}
