//! Minimal ELF .so emitter.
//!
//! Writes a valid ELF shared object with a single exported function.

use crate::compile::x86::MachineCode;
use std::io::Write;

/// ELF header constants.
const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const ET_DYN: u16 = 3;
const EM_X86_64: u16 = 62;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const STB_GLOBAL: u8 = 1;
const STT_FUNC: u8 = 2;
const PT_LOAD: u32 = 1;
const PF_X: u32 = 0x1;
const PF_R: u32 = 0x4;

/// Write a minimal ELF .so containing the module's machine code.
///
/// The exported symbol is `module_forward` with signature:
///   extern "C" fn(input: *const f32, output: *mut f32, len: usize)
pub fn write_elf(code: &MachineCode, path: &std::path::Path) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    // Layout:
    // 0x0000: ELF header (64 bytes)
    // 0x0040: Program header (56 bytes)
    // 0x0078: Section header table start (padding to 0x0080)
    // 0x0080: .text section (machine code)
    // 0x0080+text_size: .strtab section
    // after strtab: .symtab section
    // after symtab: section header table

    let text_offset: u64 = 0x0080;
    let text_size = code.text_size as u64;
    let text_addr: u64 = 0x1000; // Virtual address

    // String table: \0 + ".text\0" + ".strtab\0" + ".symtab\0" + "module_forward\0"
    let strtab: Vec<u8> = {
        let mut s = Vec::new();
        s.push(0); // null string
        s.extend_from_slice(b".text\0");    // offset 1
        s.extend_from_slice(b".strtab\0");  // offset 7
        s.extend_from_slice(b".symtab\0");  // offset 15
        s.extend_from_slice(b"module_forward\0"); // offset 23
        s
    };

    let strtab_offset = text_offset + text_size;
    let strtab_size = strtab.len() as u64;

    // Symbol table: null symbol + module_forward
    let symtab_offset = strtab_offset + strtab_size;
    let sym_entry_size: u64 = 24; // sizeof(Elf64_Sym)
    let symtab_size = sym_entry_size * 2; // null + module_forward

    // Section header table
    let shdr_offset = symtab_offset + symtab_size;
    let shdr_entry_size: u16 = 64; // sizeof(Elf64_Shdr)
    let num_sections: u16 = 4; // null, .text, .strtab, .symtab

    // ELF header
    let mut ehdr = [0u8; 64];
    ehdr[0..4].copy_from_slice(&ELF_MAGIC);
    ehdr[4] = ELFCLASS64;
    ehdr[5] = ELFDATA2LSB;
    ehdr[6] = 1; // EV_CURRENT
    ehdr[16..18].copy_from_slice(&ET_DYN.to_le_bytes());
    ehdr[18..20].copy_from_slice(&EM_X86_64.to_le_bytes());
    ehdr[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
    ehdr[24..32].copy_from_slice(&(text_addr).to_le_bytes()); // e_entry
    ehdr[32..40].copy_from_slice(&0x40u64.to_le_bytes()); // e_phoff
    ehdr[40..48].copy_from_slice(&shdr_offset.to_le_bytes()); // e_shoff
    ehdr[52..54].copy_from_slice(&64u16.to_le_bytes()); // e_ehsize
    ehdr[54..56].copy_from_slice(&56u16.to_le_bytes()); // e_phentsize
    ehdr[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum
    ehdr[58..60].copy_from_slice(&shdr_entry_size.to_le_bytes()); // e_shentsize
    ehdr[60..62].copy_from_slice(&num_sections.to_le_bytes()); // e_shnum
    ehdr[62..64].copy_from_slice(&2u16.to_le_bytes()); // e_shstrndx (.strtab)

    file.write_all(&ehdr)?;

    // Program header (LOAD segment for .text)
    let mut phdr = [0u8; 56];
    phdr[0..4].copy_from_slice(&PT_LOAD.to_le_bytes());
    phdr[4..8].copy_from_slice(&(PF_R | PF_X).to_le_bytes()); // flags
    phdr[8..16].copy_from_slice(&text_offset.to_le_bytes()); // p_offset
    phdr[16..24].copy_from_slice(&text_addr.to_le_bytes()); // p_vaddr
    phdr[24..32].copy_from_slice(&text_addr.to_le_bytes()); // p_paddr
    phdr[32..40].copy_from_slice(&text_size.to_le_bytes()); // p_filesz
    phdr[40..48].copy_from_slice(&text_size.to_le_bytes()); // p_memsz
    phdr[48..56].copy_from_slice(&0x1000u64.to_le_bytes()); // p_align

    file.write_all(&phdr)?;

    // Pad to text_offset
    let pad_len = text_offset as usize - 64 - 56;
    file.write_all(&vec![0u8; pad_len])?;

    // .text section
    file.write_all(&code.code)?;

    // .strtab section
    file.write_all(&strtab)?;

    // .symtab section
    // Null symbol
    file.write_all(&[0u8; 24])?;
    // module_forward symbol
    let mut sym = [0u8; 24];
    sym[0..4].copy_from_slice(&23u32.to_le_bytes()); // st_name offset
    sym[4] = (STB_GLOBAL << 4) | STT_FUNC; // st_info
    sym[5] = 0; // st_other
    sym[6..8].copy_from_slice(&1u16.to_le_bytes()); // st_shndx (.text)
    sym[8..16].copy_from_slice(&text_addr.to_le_bytes()); // st_value
    sym[16..24].copy_from_slice(&text_size.to_le_bytes()); // st_size
    file.write_all(&sym)?;

    // Section header table
    // [0] null section
    file.write_all(&[0u8; 64])?;
    // [1] .text
    let mut shdr = [0u8; 64];
    shdr[0..4].copy_from_slice(&1u32.to_le_bytes()); // sh_name
    shdr[4..8].copy_from_slice(&SHT_PROGBITS.to_le_bytes());
    shdr[8..16].copy_from_slice(&(SHF_ALLOC | SHF_EXECINSTR).to_le_bytes());
    shdr[16..24].copy_from_slice(&text_addr.to_le_bytes()); // sh_addr
    shdr[24..32].copy_from_slice(&text_offset.to_le_bytes());
    shdr[32..40].copy_from_slice(&text_size.to_le_bytes());
    shdr[48..56].copy_from_slice(&16u64.to_le_bytes()); // sh_addralign
    file.write_all(&shdr)?;
    // [2] .strtab
    let mut shdr = [0u8; 64];
    shdr[0..4].copy_from_slice(&7u32.to_le_bytes());
    shdr[4..8].copy_from_slice(&SHT_STRTAB.to_le_bytes());
    shdr[24..32].copy_from_slice(&strtab_offset.to_le_bytes());
    shdr[32..40].copy_from_slice(&strtab_size.to_le_bytes());
    file.write_all(&shdr)?;
    // [3] .symtab
    let mut shdr = [0u8; 64];
    shdr[0..4].copy_from_slice(&15u32.to_le_bytes());
    shdr[4..8].copy_from_slice(&SHT_SYMTAB.to_le_bytes());
    shdr[24..32].copy_from_slice(&symtab_offset.to_le_bytes());
    shdr[32..40].copy_from_slice(&symtab_size.to_le_bytes());
    shdr[40..44].copy_from_slice(&2u32.to_le_bytes()); // sh_link (.strtab)
    shdr[44..48].copy_from_slice(&1u32.to_le_bytes()); // sh_info
    shdr[56..64].copy_from_slice(&sym_entry_size.to_le_bytes()); // sh_entsize
    file.write_all(&shdr)?;

    file.flush()?;
    Ok(())
}
