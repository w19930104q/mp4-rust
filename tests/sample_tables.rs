//! Fixed test slices for the box parser: compares the structure tree and the
//! sample index against frozen expectations, and records peak memory and
//! parse (read-call) counts.
//!
//! The fixed slice is a deterministic in-memory MP4 with large stts/ctts/stsz/
//! stco/stss tables (50k samples, run-length encoding defeated on purpose).
//!
//! Run with `cargo test --test sample_tables -- --nocapture` to see the
//! recorded metrics.

use mp4::{
    AvcConfig, BoxHeader, BoxType, Bytes, MdhdBox, MediaConfig, Mp4Config, Mp4Reader, Mp4Sample,
    Mp4Writer, MvhdBox, ReadBox, TrackConfig, TrackType, WriteBox,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::convert::TryInto;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Peak-memory recording allocator.
// ---------------------------------------------------------------------------

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct PeakAlloc;

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let cur = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(cur, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc;

// Serializes tests in this binary so allocator statistics are not perturbed
// by concurrently running tests.
static TEST_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Read-call counting reader ("parse counts").
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ReadStats {
    calls: AtomicUsize,
    bytes: AtomicUsize,
}

struct CountingReader<R> {
    inner: R,
    stats: Arc<ReadStats>,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes.fetch_add(n, Ordering::Relaxed);
        Ok(n)
    }
}

impl<R: Seek> Seek for CountingReader<R> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

// ---------------------------------------------------------------------------
// Fixed test slice.
// ---------------------------------------------------------------------------

const N_SAMPLES: u32 = 50_000;

/// Deterministic sample parameters, shared by the writer and the
/// ground-truth checks below.
fn sample_duration(id: u32) -> u32 {
    512 + (id & 1) // 513, 512, 513, ... defeats stts run-length encoding
}

fn sample_rendering_offset(id: u32) -> i32 {
    (id % 3) as i32 - 1 // 0, 1, -1, 0, 1, -1, ... defeats ctts RLE
}

fn sample_size(id: u32) -> usize {
    4 + (id % 5) as usize // 4..=8, defeats fixed-size stsz
}

fn sample_is_sync(id: u32) -> bool {
    id.is_multiple_of(7)
}

fn sample_start_time(id: u32) -> u64 {
    // Pairs of (513 + 512) = 1025 per two samples.
    ((id - 1) / 2) as u64 * 1025 + if id.is_multiple_of(2) { 513 } else { 0 }
}

fn sample_bytes(id: u32) -> Vec<u8> {
    vec![(id % 256) as u8; sample_size(id)]
}

/// Builds the fixed test slice: one AVC video track with N_SAMPLES samples
/// whose stts/ctts/stsz/stco/stss tables do not collapse.
fn build_fixed_slice() -> Vec<u8> {
    let config = Mp4Config {
        major_brand: str::parse("isom").unwrap(),
        minor_version: 512,
        compatible_brands: vec![
            str::parse("isom").unwrap(),
            str::parse("iso2").unwrap(),
            str::parse("avc1").unwrap(),
            str::parse("mp41").unwrap(),
        ],
        timescale: 1000,
    };
    let track_config = TrackConfig {
        track_type: TrackType::Video,
        timescale: 1000,
        language: String::from("und"),
        media_conf: MediaConfig::AvcConfig(AvcConfig {
            width: 320,
            height: 240,
            seq_param_set: vec![0x67, 0x64, 0x00, 0x1f],
            pic_param_set: vec![0x68, 0xee, 0x3c, 0x80],
        }),
    };

    let mut writer = Mp4Writer::write_start(Cursor::new(Vec::new()), &config).unwrap();
    writer.add_track(&track_config).unwrap();
    for id in 1..=N_SAMPLES {
        writer
            .write_sample(
                1,
                &Mp4Sample {
                    start_time: 0, // computed by the writer from durations
                    duration: sample_duration(id),
                    rendering_offset: sample_rendering_offset(id),
                    is_sync: sample_is_sync(id),
                    bytes: Bytes::from(sample_bytes(id)),
                },
            )
            .unwrap();
    }
    writer.write_end().unwrap();
    writer.into_writer().into_inner()
}

// ---------------------------------------------------------------------------
// Fingerprints (FNV-1a 64 over canonical field dumps).
// ---------------------------------------------------------------------------

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf29ce484222325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }

    fn i32(&mut self, v: i32) {
        self.bytes(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }

    fn tag(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }

    fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// Canonical fingerprint of the parsed box structure tree.
fn tree_fingerprint<R>(mp4: &Mp4Reader<R>) -> String {
    let mut h = Fnv::new();
    h.tag("mvhd");
    h.u32(mp4.moov.mvhd.version as u32);
    h.u32(mp4.moov.mvhd.timescale);
    h.u64(mp4.moov.mvhd.duration);
    for trak in &mp4.moov.traks {
        h.tag("trak");
        h.u32(trak.tkhd.track_id);
        h.u32(trak.tkhd.version as u32);
        h.u64(trak.tkhd.duration);
        h.tag("mdhd");
        h.u32(trak.mdia.mdhd.version as u32);
        h.u32(trak.mdia.mdhd.timescale);
        h.u64(trak.mdia.mdhd.duration);
        h.tag(&trak.mdia.hdlr.handler_type.to_string());

        let stbl = &trak.mdia.minf.stbl;
        h.tag("stsd");
        h.u32(stbl.stsd.avc1.is_some() as u32);
        h.u32(stbl.stsd.hev1.is_some() as u32);
        h.u32(stbl.stsd.mp4a.is_some() as u32);
        h.u32(stbl.stsd.vp09.is_some() as u32);
        h.u32(stbl.stsd.tx3g.is_some() as u32);

        h.tag("stts");
        h.u32(stbl.stts.entries.len() as u32);
        for e in stbl.stts.entries.iter() {
            h.u32(e.sample_count);
            h.u32(e.sample_delta);
        }

        h.tag("ctts");
        match &stbl.ctts {
            Some(ctts) => {
                h.u32(ctts.entries.len() as u32);
                for e in ctts.entries.iter() {
                    h.u32(e.sample_count);
                    h.i32(e.sample_offset);
                }
            }
            None => h.tag("none"),
        }

        h.tag("stsc");
        h.u32(stbl.stsc.entries.len() as u32);
        for e in stbl.stsc.entries.iter() {
            h.u32(e.first_chunk);
            h.u32(e.samples_per_chunk);
            h.u32(e.sample_description_index);
            h.u32(e.first_sample);
        }

        h.tag("stsz");
        h.u32(stbl.stsz.sample_size);
        h.u32(stbl.stsz.sample_count);
        for s in stbl.stsz.sample_sizes.iter() {
            h.u32(*s);
        }

        h.tag("stss");
        match &stbl.stss {
            Some(stss) => {
                h.u32(stss.entries.len() as u32);
                for s in stss.entries.iter() {
                    h.u32(*s);
                }
            }
            None => h.tag("none"),
        }

        match (&stbl.stco, &stbl.co64) {
            (Some(stco), None) => {
                h.tag("stco");
                h.u32(stco.entries.len() as u32);
                for o in stco.entries.iter() {
                    h.u32(*o);
                }
            }
            (None, Some(co64)) => {
                h.tag("co64");
                h.u32(co64.entries.len() as u32);
                for o in co64.entries.iter() {
                    h.u64(*o);
                }
            }
            _ => h.tag("bad-offset-table"),
        }
    }
    h.tag("moofs");
    h.u32(mp4.moofs.len() as u32);
    h.hex()
}

/// Canonical fingerprint of the sample index as exposed by the public
/// read-only view (`sample_offset` + `read_sample`).
fn sample_index_fingerprint<R: Read + Seek>(mp4: &mut Mp4Reader<R>, track_id: u32) -> String {
    let mut h = Fnv::new();
    let n = mp4.sample_count(track_id).unwrap();
    h.u32(n);
    for id in 1..=n {
        h.u64(mp4.sample_offset(track_id, id).unwrap());
        let s = mp4.read_sample(track_id, id).unwrap().unwrap();
        h.u64(s.start_time);
        h.u32(s.duration);
        h.i32(s.rendering_offset);
        h.u32(s.is_sync as u32);
        h.u32(s.bytes.len() as u32);
    }
    h.hex()
}

fn parse(data: &[u8]) -> Mp4Reader<Cursor<Vec<u8>>> {
    Mp4Reader::read_header(Cursor::new(data.to_vec()), data.len() as u64).unwrap()
}

// ---------------------------------------------------------------------------
// Tests: structure tree and sample index on the fixed slice.
// ---------------------------------------------------------------------------

#[test]
fn structure_tree_matches_frozen_expectation() {
    let _g = TEST_LOCK.lock().unwrap();
    let data = build_fixed_slice();
    let mp4 = parse(&data);

    // Table shapes derived from the fixed generator parameters.
    let stbl = &mp4.moov.traks[0].mdia.minf.stbl;
    assert_eq!(stbl.stts.entries.len(), N_SAMPLES as usize);
    assert_eq!(stbl.ctts.as_ref().unwrap().entries.len(), N_SAMPLES as usize);
    assert_eq!(stbl.stsz.sample_sizes.len(), N_SAMPLES as usize);
    assert_eq!(stbl.stss.as_ref().unwrap().entries.len(), 7142); // every 7th
    assert_eq!(stbl.stco.as_ref().unwrap().entries.len(), 25000); // 2 samples/chunk
    assert_eq!(stbl.stsc.entries.len(), 1);

    assert_eq!(tree_fingerprint(&mp4), EXPECTED_TREE);
}

#[test]
fn sample_index_matches_frozen_expectation() {
    let _g = TEST_LOCK.lock().unwrap();
    let data = build_fixed_slice();
    let mut mp4 = parse(&data);

    // Ground truth from the generator formulas, independent of the parser.
    for &id in &[1u32, 2, 3, 7, 25_000, 49_999, 50_000] {
        let s = mp4.read_sample(1, id).unwrap().unwrap();
        assert_eq!(s.start_time, sample_start_time(id), "start_time id={}", id);
        assert_eq!(s.duration, sample_duration(id), "duration id={}", id);
        assert_eq!(
            s.rendering_offset,
            sample_rendering_offset(id),
            "rendering_offset id={}",
            id
        );
        assert_eq!(s.is_sync, sample_is_sync(id), "is_sync id={}", id);
        assert_eq!(&s.bytes[..], &sample_bytes(id)[..], "bytes id={}", id);
    }
    // Monotonic, gapless sample offsets inside the mdat.
    let mut prev_end = None;
    for id in (1..=N_SAMPLES).step_by(997) {
        let off = mp4.sample_offset(1, id).unwrap();
        if let Some(prev) = prev_end {
            assert!(off >= prev, "sample offsets must be monotonic");
        }
        prev_end = Some(off + sample_size(id) as u64);
    }

    assert_eq!(sample_index_fingerprint(&mut mp4, 1), EXPECTED_INDEX);
}

#[test]
fn out_of_order_seek_reads_are_stable() {
    let _g = TEST_LOCK.lock().unwrap();
    let data = build_fixed_slice();
    let mut mp4 = parse(&data);

    // Deterministic shuffled order exercises seek backwards and forwards.
    let mut ids: Vec<u32> = vec![N_SAMPLES, 1, N_SAMPLES / 2, 2, N_SAMPLES - 1];
    for i in 0..64u32 {
        ids.push((i.wrapping_mul(9973) % N_SAMPLES) + 1);
    }
    for &id in &ids {
        let s = mp4.read_sample(1, id).unwrap().unwrap();
        assert_eq!(s.start_time, sample_start_time(id), "id={}", id);
        assert_eq!(s.duration, sample_duration(id), "id={}", id);
        assert_eq!(s.rendering_offset, sample_rendering_offset(id), "id={}", id);
        assert_eq!(&s.bytes[..], &sample_bytes(id)[..], "id={}", id);
        // Re-reading the same sample must give identical results.
        let again = mp4.read_sample(1, id).unwrap().unwrap();
        assert_eq!(s, again);
    }
}

// ---------------------------------------------------------------------------
// Tests: unknown box skipping.
// ---------------------------------------------------------------------------

/// Splices an unknown top-level box and a free box between ftyp and moov.
fn splice_unknown_top_level(data: &[u8]) -> Vec<u8> {
    let ftyp_size = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
    assert_eq!(&data[4..8], b"ftyp");
    let mut out = Vec::with_capacity(data.len() + 24);
    out.extend_from_slice(&data[..ftyp_size]);
    out.extend_from_slice(&16u32.to_be_bytes()); // unknown box, 8 bytes payload
    out.extend_from_slice(b"abcd");
    out.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef, 0, 1, 2, 3]);
    out.extend_from_slice(&8u32.to_be_bytes()); // empty free box
    out.extend_from_slice(b"free");
    out.extend_from_slice(&data[ftyp_size..]);
    out
}

/// Offset of the first top-level box with the given fourcc.
fn find_top_level_box(data: &[u8], fourcc: &[u8; 4]) -> usize {
    let mut off = 0usize;
    while off + 8 <= data.len() {
        let size = u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        assert!(size >= 8, "invalid box size at {}", off);
        if &data[off + 4..off + 8] == fourcc {
            return off;
        }
        off += size;
    }
    panic!("box {:?} not found", fourcc);
}

/// Splices an unknown box inside moov (before its first child).
fn splice_unknown_inside_moov(data: &[u8]) -> Vec<u8> {
    let moov_off = find_top_level_box(data, b"moov");
    let moov_size = u32::from_be_bytes(data[moov_off..moov_off + 4].try_into().unwrap());
    let ins = moov_off + 8;
    let mut out = Vec::with_capacity(data.len() + 16);
    out.extend_from_slice(&data[..ins]);
    out.extend_from_slice(&16u32.to_be_bytes());
    out.extend_from_slice(b"zzzz");
    out.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    out.extend_from_slice(&data[ins..]);
    let new_size = moov_size + 16;
    out[moov_off..moov_off + 4].copy_from_slice(&new_size.to_be_bytes());
    out
}

#[test]
fn unknown_boxes_are_skipped() {
    let _g = TEST_LOCK.lock().unwrap();
    let data = build_fixed_slice();
    let want_tree = tree_fingerprint(&parse(&data));

    let top = splice_unknown_top_level(&data);
    let mp4 = parse(&top);
    assert_eq!(tree_fingerprint(&mp4), want_tree, "top-level unknown box");
    assert_eq!(mp4.size(), top.len() as u64);

    let nested = splice_unknown_inside_moov(&data);
    let mp4 = parse(&nested);
    assert_eq!(tree_fingerprint(&mp4), want_tree, "unknown box inside moov");
}

// ---------------------------------------------------------------------------
// Tests: 32/64-bit duration semantics.
// ---------------------------------------------------------------------------

#[test]
fn duration_32_64_bit_semantics() {
    let _g = TEST_LOCK.lock().unwrap();

    // mdhd version 0 (32-bit duration) and version 1 (64-bit duration)
    // must expose the same u64 duration field after parsing.
    for (version, duration) in [(0u8, 3_043_936u64), (1, 3_043_936), (1, 5_000_000_000)] {
        let src = MdhdBox {
            version,
            timescale: 48_000,
            duration,
            ..Default::default()
        };
        let mut buf = Vec::new();
        src.write_box(&mut buf).unwrap();
        let mut reader = Cursor::new(&buf);
        let header = BoxHeader::read(&mut reader).unwrap();
        assert_eq!(header.name, BoxType::MdhdBox);
        let dst = MdhdBox::read_box(&mut reader, header.size).unwrap();
        assert_eq!(dst.version, version);
        assert_eq!(dst.duration, duration, "mdhd v{} duration", version);
    }

    // mvhd version 0/1 round-trip with a duration exceeding u32::MAX.
    for (version, duration) in [(0u8, 62u64), (1, 5_000_000_000)] {
        let src = MvhdBox {
            version,
            timescale: 1000,
            duration,
            ..Default::default()
        };
        let mut buf = Vec::new();
        src.write_box(&mut buf).unwrap();
        let mut reader = Cursor::new(&buf);
        let header = BoxHeader::read(&mut reader).unwrap();
        assert_eq!(header.name, BoxType::MvhdBox);
        let dst = MvhdBox::read_box(&mut reader, header.size).unwrap();
        assert_eq!(dst.duration, duration, "mvhd v{} duration", version);
    }

    // File level: track/movie durations derive from the 32-bit (v0) fields.
    let data = build_fixed_slice();
    let mp4 = parse(&data);
    let track = mp4.tracks().get(&1).unwrap();
    let mdhd = &track.trak.mdia.mdhd;
    assert_eq!(mdhd.version, 0);
    assert_eq!(
        track.duration(),
        Duration::from_micros(mdhd.duration * 1_000_000 / u64::from(mdhd.timescale))
    );
    assert_eq!(
        mp4.duration(),
        Duration::from_millis(mp4.moov.mvhd.duration * 1000 / u64::from(mp4.moov.mvhd.timescale))
    );
}

// ---------------------------------------------------------------------------
// Metrics: peak memory and parse (read-call) counts.
// ---------------------------------------------------------------------------

#[test]
fn report_peak_memory_and_parse_counts() {
    let _g = TEST_LOCK.lock().unwrap();
    let data = build_fixed_slice();
    let stats = Arc::new(ReadStats::default());
    let reader = CountingReader {
        inner: Cursor::new(&data[..]),
        stats: Arc::clone(&stats),
    };
    let size = data.len() as u64;

    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);

    let mut mp4 = Mp4Reader::read_header(reader, size).unwrap();

    let retained = CURRENT.load(Ordering::SeqCst) - base;
    let peak = PEAK.load(Ordering::SeqCst) - base;
    let reads = stats.calls.load(Ordering::Relaxed);
    let read_bytes = stats.bytes.load(Ordering::Relaxed);
    let tree = tree_fingerprint(&mp4);
    let index = sample_index_fingerprint(&mut mp4, 1);

    println!("METRICS file_bytes={}", size);
    println!("METRICS read_calls={} read_bytes={}", reads, read_bytes);
    println!("METRICS retained_bytes={} peak_bytes={}", retained, peak);
    println!("METRICS tree={}", tree);
    println!("METRICS index={}", index);

    drop(mp4);
    println!(
        "METRICS retained_after_drop={}",
        CURRENT.load(Ordering::SeqCst) - base
    );

    assert_eq!(tree, EXPECTED_TREE);
    assert_eq!(index, EXPECTED_INDEX);

    // Budgets recorded after the copy-reduction refactor; the pre-refactor
    // parser exceeds them by an order of magnitude (see PR description).
    assert!(
        reads <= MAX_READ_CALLS,
        "parse issued {} read calls, budget {}",
        reads,
        MAX_READ_CALLS
    );
    assert!(
        retained <= MAX_RETAINED_BYTES,
        "parse retained {} bytes, budget {}",
        retained,
        MAX_RETAINED_BYTES
    );
    assert!(
        peak <= MAX_PEAK_BYTES,
        "parse peaked at {} bytes, budget {}",
        peak,
        MAX_PEAK_BYTES
    );
}

// Frozen expectations for the fixed slice (recorded from the pre-refactor
// parser; the refactor must reproduce them exactly).
const EXPECTED_TREE: &str = "4371fc00ee90686d";
const EXPECTED_INDEX: &str = "5e00245c36951fd3";

// Metric budgets (post-refactor, debug build):
//   read_calls=192, retained=1136152, peak=1145249.
// Baseline values at 35560e9 (pre-refactor, same harness):
//   read_calls=282288, retained=2265392, peak=2265392.
// Budgets sit far below the baseline so a regression to per-entry reads or
// double-stored sample tables trips them.
const MAX_READ_CALLS: usize = 1_000;
const MAX_RETAINED_BYTES: usize = 1_400_000;
const MAX_PEAK_BYTES: usize = 1_500_000;
