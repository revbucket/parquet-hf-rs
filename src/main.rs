use std::sync::Arc;




use std::time::Instant;
use std::io::BufRead;


use serde_json;

use serde_json::Value;
use anyhow::Error;

use clap::Parser;
use std::path::PathBuf;
use crate::io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, get_output_filename};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};


use arrow::array::{ArrayRef, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use regex::Regex;

pub mod s3;
pub mod io;




/*=================================================================
=                                  ARGS                           =
=================================================================*/



#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct ArgParser {
    #[arg(long, required=true, num_args=1..)]
    input: Vec<PathBuf>,

    #[arg(long, required=true)]
    output: PathBuf,

    #[arg(long, default_value_t=1_000_000)]
    reservoir_size: usize,

    #[arg(long, default_value_t=0)]
    threads: usize,
}


/*=================================================================
=                             UTILITIES.                          =
=================================================================*/

fn build_pbar(num_items: usize, units: &str) -> ProgressBar {
    let mut template = String::from(units);
    template.push_str(" {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]");
    let pbar = ProgressBar::new(num_items as u64)
        .with_style(
            ProgressStyle::with_template(&template).unwrap()
        );

    pbar.inc(0);
    pbar
}


fn replace_extension(path: &PathBuf) -> PathBuf {
    let path = path.clone();
    let regex = Regex::new(r"\.jsonl?\.(?:zstd|gz)$").unwrap();
    let path_str = path.to_str().unwrap();
    
    let output_path = if regex.is_match(path_str) {
        let new_path = regex.replace(path_str, ".parquet");
        let path = PathBuf::from(new_path.into_owned());
        path 
    } else {
        path
    };
    output_path
}


fn _build_schema() -> arrow::datatypes::Schema {
    let schema: arrow::datatypes::Schema = Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("url", DataType::Utf8, false),
        Field::new("warcinfo", DataType::Utf8, false),
        Field::new("metadata", DataType::Utf8, false)]);
    schema 
}


fn jsonl_to_parquet(input_path: &PathBuf, output_path: &PathBuf) -> Result<(), Error> {

    let contents = read_pathbuf_to_mem(input_path).unwrap();

    let mut text_builder = StringBuilder::new();
    let mut url_builder = StringBuilder::new();
    let mut warcinfo_builder = StringBuilder::new();
    let mut metadata_builder = StringBuilder::new();

    for line in contents.lines() {
        let line = line.unwrap();
        let json : Value = serde_json::from_str(&line).unwrap();

        text_builder.append_value(json["text"].as_str().unwrap());
        url_builder.append_value(json["url"].as_str().unwrap());
        warcinfo_builder.append_value(json["warcinfo"].as_str().unwrap());
        metadata_builder.append_value(serde_json::to_string(&json["metadata"]).unwrap());
    }
    let text_array : ArrayRef = Arc::new(text_builder.finish());
    let url_array : ArrayRef = Arc::new(url_builder.finish());
    let warcinfo_array : ArrayRef = Arc::new(warcinfo_builder.finish());
    let metadata_array : ArrayRef = Arc::new(metadata_builder.finish());


    let schema = _build_schema();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![text_array, url_array, warcinfo_array, metadata_array],
    )?;    

    let mut buf = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buf, Arc::new(schema), None)?;
        writer.write(&batch)?;
        writer.close()?;
    }    

    write_mem_to_pathbuf(&buf, output_path).unwrap();
    Ok(())
}



/*=================================================================
=                                  MAIN                           =
=================================================================*/

fn main() {
    let start_main = Instant::now();
    let args = ArgParser::parse();

    let paths = expand_dirs(args.input.clone(), None).unwrap();
    let pbar = build_pbar(paths.len(), "Paths");

    paths.par_iter()
        .for_each(|p| {
            let output_path = get_output_filename(&args.input, p, &args.output);
            let output_path = replace_extension(&output_path);
            jsonl_to_parquet(p, &output_path).unwrap();
            pbar.inc(1);
        });


    println!("-------------------------");
    println!("Finishing parquet creation in {:?} seconds", start_main.elapsed().as_secs());    
}