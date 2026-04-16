//! Spore packaging — self-extracting archive for kernel propagation.
//!
//! A spore contains everything needed to boot on a bare Linux machine:
//! the binary, a seed model, and a bootstrap script.

use std::io::Write;
use std::path::Path;

/// Package the current kernel into a spore.
pub fn package_spore(
    binary_path: &Path,
    model_path: &Path,
    output_path: &Path,
) -> std::io::Result<SporeInfo> {
    let binary = std::fs::read(binary_path)?;
    let model = std::fs::read(model_path)?;

    let bootstrap = format!(
        "#!/bin/sh\n\
         set -e\n\
         echo '=== The Kernel: Spore Extraction ==='\n\
         OFFSET=$(awk '/^__PAYLOAD__$/{{print NR + 1; exit 0;}}' \"$0\")\n\
         EXTRACT_DIR=$(mktemp -d /tmp/clob.XXXXXX)\n\
         tail -n+$OFFSET \"$0\" | tar xz -C \"$EXTRACT_DIR\"\n\
         chmod +x \"$EXTRACT_DIR/clob\"\n\
         echo \"Booting kernel from $EXTRACT_DIR\"\n\
         exec \"$EXTRACT_DIR/clob\" boot \\\n\
           --model \"$EXTRACT_DIR/seed.clob\" \\\n\
           --memory-dir \"$EXTRACT_DIR/episodes\" \\\n\
           --modules-dir \"$EXTRACT_DIR/modules\"\n\
         __PAYLOAD__\n"
    );

    // Create tar.gz payload
    let payload = create_tar_gz(&binary, &model)?;

    // Write self-extracting script
    let mut file = std::fs::File::create(output_path)?;
    file.write_all(bootstrap.as_bytes())?;
    file.write_all(&payload)?;
    file.flush()?;

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(output_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(output_path, perms)?;
    }

    let total_size = bootstrap.len() + payload.len();
    Ok(SporeInfo {
        binary_size: binary.len(),
        model_size: model.len(),
        total_size,
    })
}

/// Spore packaging info.
pub struct SporeInfo {
    pub binary_size: usize,
    pub model_size: usize,
    pub total_size: usize,
}

impl SporeInfo {
    pub fn summary(&self) -> String {
        format!(
            "binary={:.2}MB, model={:.2}MB, total={:.2}MB",
            self.binary_size as f64 / (1024.0 * 1024.0),
            self.model_size as f64 / (1024.0 * 1024.0),
            self.total_size as f64 / (1024.0 * 1024.0),
        )
    }
}

/// Create a tar.gz containing the binary and model.
fn create_tar_gz(binary: &[u8], model: &[u8]) -> std::io::Result<Vec<u8>> {
    // Minimal tar format (no compression for now — just tar)
    let mut tar = Vec::new();

    write_tar_entry(&mut tar, "clob", binary)?;
    write_tar_entry(&mut tar, "seed.clob", model)?;

    // End-of-archive: two 512-byte zero blocks
    tar.extend_from_slice(&[0u8; 1024]);

    Ok(tar)
}

/// Write a single tar entry (simplified POSIX tar).
fn write_tar_entry(tar: &mut Vec<u8>, name: &str, data: &[u8]) -> std::io::Result<()> {
    let mut header = [0u8; 512];

    // Name (100 bytes)
    let name_bytes = name.as_bytes();
    header[..name_bytes.len().min(100)].copy_from_slice(&name_bytes[..name_bytes.len().min(100)]);

    // Mode (8 bytes, octal string)
    header[100..107].copy_from_slice(b"0000755");

    // Size (12 bytes, octal string)
    let size_str = format!("{:011o}", data.len());
    header[124..135].copy_from_slice(size_str.as_bytes());

    // Modification time (12 bytes, octal)
    let mtime = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mtime_str = format!("{:011o}", mtime);
    header[136..147].copy_from_slice(mtime_str.as_bytes());

    // Type flag ('0' = regular file)
    header[156] = b'0';

    // Magic
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");

    // Checksum (8 bytes at offset 148, sum of all bytes with checksum field as spaces)
    header[148..156].copy_from_slice(b"        ");
    let checksum: u32 = header.iter().map(|&b| b as u32).sum();
    let cksum_str = format!("{:06o}\0 ", checksum);
    header[148..156].copy_from_slice(cksum_str.as_bytes());

    tar.extend_from_slice(&header);
    tar.extend_from_slice(data);

    // Pad to 512-byte boundary
    let padding = (512 - (data.len() % 512)) % 512;
    tar.extend_from_slice(&vec![0u8; padding]);

    Ok(())
}
