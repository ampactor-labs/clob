//! Integration test for train/holdout split enforcement (Phase L data gate).
//!
//! Runs in CI without any downloaded corpus: the in-memory cases always
//! execute. When a real corpus has been acquired (`data/corpus/{train,
//! holdout}.txt` present), it additionally asserts the real split is clean.

use clob::util::split_check::find_overlap;
use std::path::Path;

#[test]
fn injected_overlap_between_train_and_holdout_is_caught() {
    let shared =
        "A planted sentence long enough to exceed the minimum word threshold for detection.";
    let train = format!("Training preamble unrelated to anything else at all. {shared}");
    let holdout = format!("{shared} A holdout tail that differs from the training side entirely.");
    let overlap = find_overlap(&train, &holdout)
        .expect("an injected shared sentence must be reported as contamination");
    assert!(overlap.count >= 1);
}

#[test]
fn clean_split_passes() {
    let train =
        "Ternary cores avoid multiplication by restricting weights to minus one, zero, and one.";
    let holdout =
        "The crystallization engine clusters prediction errors before distilling them to modules.";
    assert!(find_overlap(train, holdout).is_none());
}

#[test]
fn real_corpus_split_is_clean_when_present() {
    let train_p = Path::new("data/corpus/train.txt");
    let holdout_p = Path::new("data/corpus/holdout.txt");
    if !train_p.exists() || !holdout_p.exists() {
        eprintln!("[data_split] real corpus absent — skipping (run scripts/acquire_corpus.sh)");
        return;
    }
    let train = std::fs::read_to_string(train_p).unwrap();
    let holdout = std::fs::read_to_string(holdout_p).unwrap();
    if let Some(o) = find_overlap(&train, &holdout) {
        panic!(
            "train/holdout contamination: {} shared sentences, e.g. {:?}",
            o.count, o.examples
        );
    }
}
