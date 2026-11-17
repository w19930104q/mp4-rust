use criterion::BenchmarkId;
use criterion::{criterion_group, criterion_main, Criterion};

use std::fs::File;
use std::io::Cursor;

use mp4::{
    AvcConfig, Bytes, MediaConfig, Mp4Config, Mp4Reader, Mp4Sample, Mp4Writer, TrackConfig,
    TrackType,
};

fn read_mp4(filename: &str) -> u64 {
    let f = File::open(filename).unwrap();
    let m = mp4::read_mp4(f).unwrap();

    m.size()
}

/// Deterministic in-memory MP4 with 50k samples and non-collapsing
/// stts/ctts/stsz/stco/stss tables (same shape as tests/sample_tables.rs).
fn build_big_table_slice(n: u32) -> Vec<u8> {
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
    for id in 1..=n {
        writer
            .write_sample(
                1,
                &Mp4Sample {
                    start_time: 0,
                    duration: 512 + (id & 1),
                    rendering_offset: (id % 3) as i32 - 1,
                    is_sync: id % 7 == 0,
                    bytes: Bytes::from(vec![(id % 256) as u8; 4 + (id % 5) as usize]),
                },
            )
            .unwrap();
    }
    writer.write_end().unwrap();
    writer.into_writer().into_inner()
}

fn criterion_benchmark(c: &mut Criterion) {
    let filename = "tests/samples/minimal.mp4";

    c.bench_with_input(
        BenchmarkId::new("input_example", filename),
        &filename,
        |b, &s| {
            b.iter(|| read_mp4(s));
        },
    );

    let slice = build_big_table_slice(50_000);
    let size = slice.len() as u64;
    c.bench_with_input(
        BenchmarkId::new("parse_header", "50k_sample_tables"),
        &slice,
        |b, data| {
            b.iter(|| Mp4Reader::read_header(Cursor::new(&data[..]), size).unwrap());
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
